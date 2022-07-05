import math
from typing import Union, List, Optional

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
import einops


class EEGEmotionRecognitionTransformer(pl.LightningModule):
    def __init__(self, in_channels: int, labels: Union[int, List[str]]):
        super().__init__()
        assert isinstance(in_channels, int) and in_channels >= 1
        self.in_channels = in_channels

        assert isinstance(labels, int) or isinstance(labels, list)
        if isinstance(labels, list):
            assert False not in [isinstance(label, str) for label in labels], f"the names of the labels must be strings"
            self.labels = labels
        else:
            self.labels = [f"label_{i}" for i in range(labels)]

        self.cnn_pre = nn.Sequential(
            ResidualBlock(in_channels=self.in_channels, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=128),

            ResidualBlock(in_channels=128, out_channels=128),
            ResidualBlock(in_channels=128, out_channels=128),
            ResidualBlock(in_channels=128, out_channels=256),

            ResidualBlock(in_channels=256, out_channels=256),
            ResidualBlock(in_channels=256, out_channels=256),
            ResidualBlock(in_channels=256, out_channels=512),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
            num_layers=8)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True),
            num_layers=8)
        self.target_embeddings = nn.Embedding(len(self.labels), 512)

        self.reshape_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(start_dim=1),
        )
        self.classification = nn.ModuleList()
        for label in self.labels:
            self.classification.add_module(label,
                                           nn.Sequential(
                                               nn.Linear(in_features=512, out_features=256),
                                               nn.ReLU(),
                                               nn.BatchNorm1d(256),
                                               nn.Linear(in_features=256, out_features=2),
                                           ))
        self.float()
        self.save_hyperparameters()

    def forward(self, eeg):
        x = eeg.to(self.device)  # (b s c)

        # latent space
        x = einops.rearrange(x, "b s c -> b c s")
        x = self.cnn_pre(x)

        # transformer encoder
        x = einops.rearrange(x, "b c s -> b s c")
        x = PositionalEncoding(x.shape[-1]).to(self.device)(x)
        # mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1])
        # mask = torch.triu(torch.full((x.shape[1], x.shape[1]), float('-inf')), diagonal=1).to(self.device).repeat(x.shape[0], 1, 1)
        # print(mask.shape, x.shape)
        # x = self.transformer_encoder(x, mask)
        x = self.transformer_encoder(x)

        # transformer decoder
        e = self.target_embeddings(torch.arange(len(self.labels), device=self.device)) \
            .unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = self.transformer_decoder(e, x)

        labels_pred = torch.stack([net(x[:, i_label, :])
                                   for i_label, net in enumerate(self.classification)],
                                  dim=1)  # (b l d)
        return labels_pred

    def training_step(self, batch, batch_idx):
        eeg, labels = batch  # (b s c), (b l)
        labels_pred = self(eeg)  # (b l d)
        losses = [F.cross_entropy(labels_pred[:, i_label, :], labels[:, i_label])
                  for i_label in range(labels.shape[-1])]
        accs = [torchmetrics.functional.accuracy(labels_pred[:, i_label, :], labels[:, i_label], average="micro")
                for i_label in range(labels.shape[-1])]
        for i_label, label in enumerate(self.labels):
            self.log(f"{label}_loss_train", losses[i_label], prog_bar=False)
            self.log(f"{label}_acc_train", accs[i_label], prog_bar=False)
        loss, acc = sum(losses), (sum(accs) / len(accs))
        self.log(f"loss", loss, prog_bar=True)
        self.log(f"acc_train", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        eeg, labels = batch  # (b s c), (b l)
        # print(eeg.shape, labels.shape)
        labels_pred = self(eeg)  # (b l d)
        losses = [F.cross_entropy(labels_pred[:, i_label, :], labels[:, i_label])
                  for i_label in range(labels.shape[-1])]
        accs = [torchmetrics.functional.accuracy(labels_pred[:, i_label, :], labels[:, i_label], average="micro")
                for i_label in range(labels.shape[-1])]
        for i_label, label in enumerate(self.labels):
            self.log(f"{label}_loss_val", losses[i_label], prog_bar=False)
            self.log(f"{label}_acc_val", accs[i_label], prog_bar=False)
        loss, acc = sum(losses), (sum(accs) / len(accs))
        self.log(f"loss_val", loss, prog_bar=True)
        self.log(f"acc_val", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
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
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=self.in_channels),
            nn.ReLU(),

            nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels,
                      kernel_size=1, stride=2 if self.reduce_output else 1),
            nn.BatchNorm1d(num_features=self.out_channels),
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


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = einops.rearrange(x, "b s e -> s b e")
        x = x + self.pe[:x.size(0)]
        x = einops.rearrange(x, "s b e -> b s e")
        return self.dropout(x)
