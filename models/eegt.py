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


class EEGT(pl.LightningModule):
    def __init__(self, in_channels: int, labels: Union[int, List[str]],
                 sampling_rate: int, windows_length: float = 0.1,
                 max_sequence_length: int = 5000,
                 mask_perc_min: float = 0.05, mask_perc_max: float = 0.15,
                 num_encoders: int = 4, num_decoders: int = 4,
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
        assert windows_length > 0
        self.windows_length = math.floor(self.eeg_sampling_rate * windows_length)

        assert isinstance(max_sequence_length, int) and max_sequence_length >= 1
        self.max_sequence_length = max_sequence_length

        assert isinstance(mask_perc_min, float) and 0 <= mask_perc_min < 1
        self.mask_perc_min = mask_perc_min

        assert isinstance(mask_perc_max, float) and 0 <= mask_perc_max < 1 and mask_perc_max >= mask_perc_min
        self.mask_perc_max = mask_perc_max

        assert isinstance(num_encoders, int) and num_encoders >= 1
        self.num_encoders = num_encoders

        assert isinstance(num_decoders, int) and num_decoders >= 1
        self.num_decoders = num_decoders

        assert isinstance(window_embedding_dim, int) and window_embedding_dim >= 1
        self.window_embedding_dim = window_embedding_dim

        assert isinstance(learning_rate, float) and learning_rate > 0
        self.learning_rate = learning_rate

        self.cnn = nn.Sequential(
            ResidualBlock(in_channels=self.in_channels, out_channels=256, reduce_output=True),
            # ResidualBlock(in_channels=64, out_channels=256, reduce_output=True),
            ResidualBlock(in_channels=256, out_channels=self.window_embedding_dim, reduce_output=True),
        )
        self.cnn_pre = nn.Sequential(
            # ResidualBlock(in_channels=self.in_channels, out_channels=64, reduce_output=True),
            # ResidualBlock(in_channels=64, out_channels=256, reduce_output=True),
            # ResidualBlock(in_channels=256, out_channels=self.window_embedding_dim, reduce_output=True),
            ResidualBlock(in_channels=self.in_channels, out_channels=self.window_embedding_dim, reduce_output=True),
            nn.AdaptiveMaxPool1d(output_size=(1,)),
            nn.Flatten(start_dim=1),
        )
        # self.linear_pre = nn.Sequential(
        #     nn.Flatten(start_dim=1),
        #     nn.Linear(in_features=self.windows_length * self.in_channels, out_features=1024),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(num_features=1024),
        #     nn.Linear(in_features=1024, out_features=512),
        # )
        self.special_tokens = {
            token: i_token
            for i_token, token in enumerate(["start", "end", "mask"])
        }
        self.tokens_embedder = nn.Embedding(len(self.special_tokens), self.window_embedding_dim)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.window_embedding_dim, nhead=8, batch_first=True),
            num_layers=self.num_encoders)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.window_embedding_dim, nhead=8, batch_first=True),
            num_layers=self.num_decoders)
        self.target_embedder = nn.Embedding(len(self.labels), self.window_embedding_dim)

        self.classification = nn.ModuleList()
        for label in self.labels:
            self.classification.add_module(label,
                                           nn.Sequential(
                                               nn.Linear(in_features=self.window_embedding_dim, out_features=2),
                                           ))
        self.float()
        self.save_hyperparameters()

    def forward(self, eeg):
        assert eeg.shape[-1] == self.in_channels
        x = eeg  # (b s c)
        with profiler.record_function("cnns"):
            timestamps = list(range(self.windows_length, x.shape[1] + 1, self.windows_length//2))
            if timestamps[-1] != x.shape[1]:
                timestamps += [x.shape[1]]
            x = einops.rearrange(x, "b s c -> b c s")  # (b c s)
            latent_representations = []
            for t in timestamps:
                latent_representations += [self.cnn_pre(x[:, :, t - self.windows_length:t])]
            x = torch.stack(latent_representations, dim=1)

        with profiler.record_function("transformer encoder"):
            # adds special tokens
            start_token, end_token, mask_token = self.tokens_embedder(torch.as_tensor([
                self.special_tokens["start"],
                self.special_tokens["end"],
                self.special_tokens["mask"],
            ], device=self.device))
            if self.training:
                # masks a percentage of tokens
                for i_batch, batch in enumerate(x):
                    masked_no = int(((self.mask_perc_min - self.mask_perc_max) * torch.rand(1) + self.mask_perc_max) \
                                    * batch.shape[0])
                    mask_ixs = torch.randperm(batch.shape[0])[:int(masked_no)]
                    x[i_batch, mask_ixs] = mask_token
            # adds start and end token
            x = torch.cat([start_token.repeat(x.shape[0], 1, 1),
                           x,
                           end_token.repeat(x.shape[0], 1, 1)], dim=1)
            # adds positional embeddings
            x = self.add_positional_encodings(x)
            # encoder_mask = self.generate_square_subsequent_mask(sequence_length=x.shape[1])
            # encoder_mask = torch.zeros((x.shape[1], x.shape[1]), device=self.device, requires_grad=False)
            # print(self.training, x.shape, encoder_mask.shape)
            # x = self.transformer_encoder(x, encoder_mask)
            x = self.transformer_encoder(x)

        with profiler.record_function("transformer decoder"):
            # transformer decoder
            e = self.target_embedder(torch.arange(len(self.labels), device=self.device, dtype=torch.int)) \
                .repeat(x.shape[0], 1, 1)
            x = self.transformer_decoder(e, x)

        with profiler.record_function("predictions"):
            labels_pred = torch.stack([net(x[:, i_label, :])
                                       for i_label, net in enumerate(self.classification)],
                                      dim=1)  # (b l d)
            # labels_pred = torch.cat([net(x[:, i_label, :])
            #                          for i_label, net in enumerate(self.classification)],
            #                         dim=-1)  # (b l)
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
        self.log("training", True, prog_bar=False)
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
        self.log("training", False, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def generate_square_subsequent_mask(self, sequence_length: int):
        assert isinstance(sequence_length, int) and sequence_length >= 1
        # mask = torch.triu(torch.full((sequence_length, sequence_length), float('-inf'),
        #                              device=self.device, dtype=torch.float32),
        #                   diagonal=1)
        mask = torch.triu(torch.full((sequence_length, sequence_length), float("-inf"),
                                     device=self.device, dtype=torch.float32),
                          diagonal=1)
        # mask = mask < 0
        return mask

    def add_positional_encodings(self, x):
        position = torch.arange(self.max_sequence_length, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, x.shape[-1], 2, device=self.device, dtype=torch.float32)
                             * (-math.log(10000.0) / x.shape[-1]))
        pe = torch.zeros(self.max_sequence_length, 1, x.shape[-1], device=self.device, dtype=torch.float32)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        x = einops.rearrange(x, "b s e -> s b e")
        x = x + pe[:x.size(0)]
        x = einops.rearrange(x, "s b e -> b s e")
        if self.training:
            x = nn.Dropout(p=0.1)(x)
        return x


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
            # nn.BatchNorm1d(num_features=self.in_channels),
            nn.ReLU(),

            nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels,
                      kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(num_features=self.in_channels),
            nn.ReLU(),

            nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels,
                      kernel_size=1, stride=2 if self.reduce_output else 1),
            # nn.BatchNorm1d(num_features=self.out_channels),
        )
        self.projection_stream = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels,
                      kernel_size=1, stride=2 if self.reduce_output else 1),
        )

    def forward(self, x):
        x_transformed = self.reduction_stream(x)
        x_projection = self.projection_stream(x)
        out = x_transformed + x_projection
        return out


if __name__ == "__main__":
    eegt = EEGT(in_channels=32, labels=4, sampling_rate=128, windows_length=0.1,
                mask_perc_min=0.1, mask_perc_max=0.3)
    x = torch.randn(64, 128, 32)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            out = eegt(x)
    print(prof.key_averages().table(sort_by="cpu_time", row_limit=5))
