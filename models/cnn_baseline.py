import math
from typing import Union, List, Optional

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
import einops
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity


class CNNBaseline(pl.LightningModule):
    def __init__(self, in_channels: int, labels: Union[int, List[str]],
                 sampling_rate: int,
                 window_embedding_dim: int = 512,
                 learning_rate: float = 1e-3):
        super().__init__()
        assert isinstance(in_channels, int) and in_channels >= 1
        self.in_channels = in_channels

        assert isinstance(labels, int) or isinstance(labels, list)
        if isinstance(labels, list):
            assert False not in [isinstance(label, str) for label in labels], f"the names of the labels must be strings"
            self.labels = labels
        else:
            self.labels = [f"label_{i}" for i in range(labels)]

        assert isinstance(sampling_rate, int) and sampling_rate >= 1
        self.eeg_sampling_rate = sampling_rate

        assert isinstance(window_embedding_dim, int) and window_embedding_dim >= 1
        self.window_embedding_dim = window_embedding_dim

        assert isinstance(learning_rate, float) and learning_rate > 0
        self.learning_rate = learning_rate

        self.cnn = nn.Sequential(
            nn.Conv1d(self.in_channels, 64, kernel_size=1, stride=1),

            ResidualBlock(in_channels=64, out_channels=64, reduce_output=True),
            ResidualBlock(in_channels=64, out_channels=64, reduce_output=False),
            ResidualBlock(in_channels=64, out_channels=128, reduce_output=True),

            ResidualBlock(in_channels=128, out_channels=128, reduce_output=True),
            ResidualBlock(in_channels=128, out_channels=128, reduce_output=False),
            ResidualBlock(in_channels=128, out_channels=256, reduce_output=True),

            ResidualBlock(in_channels=256, out_channels=256, reduce_output=False),
            ResidualBlock(in_channels=256, out_channels=self.window_embedding_dim, reduce_output=True),
            nn.AdaptiveAvgPool1d(output_size=(1,)),
            nn.Flatten(start_dim=1),
        )
        self.special_tokens = {
            token: i_token
            for i_token, token in enumerate(["mask"])
        }
        self.tokens_embedder = nn.Embedding(len(self.special_tokens), self.window_embedding_dim)

        self.classification = nn.ModuleList()
        for label in self.labels:
            self.classification.add_module(label,
                                           nn.Sequential(
                                               nn.Linear(in_features=self.window_embedding_dim, out_features=1024),
                                               nn.Sigmoid(),
                                               nn.BatchNorm1d(num_features=1024),

                                               nn.Linear(in_features=1024, out_features=128),
                                               nn.Sigmoid(),
                                               nn.BatchNorm1d(num_features=128),

                                               nn.Linear(in_features=128, out_features=2),
                                           ))
        self.float()
        self.save_hyperparameters()

    def forward(self, eeg):
        assert eeg.shape[-1] == self.in_channels
        x = eeg  # (b s c)
        with profiler.record_function("cnns"):
            x = einops.rearrange(x, "b s c -> b c s")
            x = self.cnn(x)

        with profiler.record_function("predictions"):
            labels_pred = torch.stack([net(x)
                                       for i_label, net in enumerate(self.classification)],
                                      dim=1)  # (b l d)
            assert labels_pred.shape[1] == len(self.labels)

        return labels_pred

    def training_step(self, batch, batch_idx):
        eeg, labels = [e.to(self.device) for e in batch]  # (b s c), (b l)
        labels_pred = self(eeg)  # (b l d)
        losses = [F.cross_entropy(labels_pred[:, i_label, :], labels[:, i_label])
                  for i_label in range(labels.shape[-1])]
        accs = [torchmetrics.functional.accuracy(F.softmax(labels_pred[:, i_label, :], dim=1),
                                                 labels[:, i_label], average="micro")
                for i_label in range(labels.shape[-1])]
        for i_label, label in enumerate(self.labels):
            self.log(f"{label}_acc_train", accs[i_label], prog_bar=False)
        loss, acc = sum(losses), (sum(accs) / len(accs))
        self.log(f"loss", loss)
        self.log(f"acc_train", acc, prog_bar=True)
        self.log("training", 1.0, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        eeg, labels = [e.to(self.device) for e in batch]  # (b s c), (b l)
        labels_pred = self(eeg)  # (b l)
        losses = [F.cross_entropy(labels_pred[:, i_label, :], labels[:, i_label])
                  for i_label in range(labels.shape[-1])]
        accs = [torchmetrics.functional.accuracy(F.softmax(labels_pred[:, i_label, :], dim=1),
                                                 labels[:, i_label], average="micro")
                for i_label in range(labels.shape[-1])]
        for i_label, label in enumerate(self.labels):
            self.log(f"{label}_acc_val", accs[i_label], prog_bar=False)
        loss, acc = sum(losses), (sum(accs) / len(accs))
        self.log(f"loss_val", loss, prog_bar=True)
        self.log(f"acc_val", acc, prog_bar=True)
        self.log("training", 0.0, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None,
                 reduce_output: bool = False):
        super(ResidualBlock, self).__init__()
        assert isinstance(in_channels, int) and in_channels >= 1
        self.in_channels = in_channels

        assert out_channels is None or (isinstance(in_channels, int) and in_channels >= 1)
        if out_channels is None:
            self.out_channels = self.in_channels
        else:
            self.out_channels = out_channels

        assert isinstance(reduce_output, bool)
        self.reduce_output = reduce_output

        self.reduction_stream = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1),
            nn.BatchNorm1d(num_features=self.in_channels),
            nn.ReLU(),

            nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels,
                      kernel_size=3, stride=2 if self.reduce_output else 1, padding=1),
            nn.BatchNorm1d(num_features=self.in_channels),
            nn.ReLU(),

            nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1),
            nn.BatchNorm1d(num_features=self.out_channels),
        )
        self.projection_stream = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels,
                      kernel_size=1, stride=2 if self.reduce_output else 1),
            nn.BatchNorm1d(num_features=self.out_channels),
        )
        self.normalization_stream = nn.Sequential(
            nn.ReLU()
        )

    def forward(self, x):
        x_transformed = self.reduction_stream(x)
        if self.in_channels != self.out_channels or self.reduce_output:
            x = self.projection_stream(x)
        x = x_transformed + x
        out = self.normalization_stream(x)
        return out


if __name__ == "__main__":
    model = CNNBaseline(in_channels=32, labels=4, sampling_rate=128)
    x = torch.randn(64, 128, 32)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            out = model(x)
    print(prof.key_averages().table(sort_by="cpu_time", row_limit=5))
