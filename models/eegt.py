import math
from typing import Union, List

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

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32, nhead=8, batch_first=True),
            num_layers=2)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=32, nhead=8, batch_first=True),
            num_layers=2)
        self.target_embeddings = nn.Embedding(len(self.labels), 32)

        self.reshape_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(start_dim=1),
        )
        self.classification = nn.ModuleList()
        for label in self.labels:
            self.classification.add_module(label,
                                           nn.Sequential(nn.Linear(in_features=32, out_features=32),
                                                         nn.ReLU(),
                                                         nn.Linear(in_features=32, out_features=2)))
        self.float()
        self.save_hyperparameters()

    def forward(self, eeg):
        x = eeg.to(self.device)  # (b s c)

        # transformer encoder
        x = PositionalEncoding(x.shape[-1])(x)
        # mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1])
        # x = self.transformer_encoder(x, mask)
        x = self.transformer_encoder(x)

        # transformer decoder
        e = self.target_embeddings(torch.arange(len(self.labels))).unsqueeze(0).repeat(x.shape[0], 1, 1)
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
        self.log(f"loss_train", loss, prog_bar=True)
        self.log(f"acc_train", acc, prog_bar=True)

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
