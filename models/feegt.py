import math
from typing import Union, List, Optional, Dict, Any

import functorch
import numpy as np
import torch
from einops.layers.torch import Rearrange
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
import einops
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity
from torchaudio import transforms


class FEEGT(pl.LightningModule):
    def __init__(self, in_channels: int, labels: Union[int, List[str]],
                 sampling_rate: int, windows_length: float = 0.1,
                 max_sequence_length: int = 5000,
                 mask_perc_min: float = 0.05, mask_perc_max: float = 0.15,
                 num_encoders: int = 4, num_decoders: int = 4,
                 window_embedding_dim: int = 512,
                 learning_rate: float = 1e-3,
                 dropout: float = 0.1,
                 mels: int = 4,
                 device: Optional[str] = None):
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

        assert 0 <= dropout < 1
        self.dropout = dropout

        assert isinstance(mels, int) and mels >= 1
        self.mels = mels

        self.cnn_bands = nn.ModuleList()
        self.normalization = nn.Sequential(
            Rearrange("b s c m -> b c s m"),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=(1, 1), stride=(1, 1)),
            Rearrange("b c s m -> b s m c"),
            nn.LayerNorm(self.in_channels),
            Rearrange("b s m c -> b s c m"),
        )
        self.cnn_merge = nn.Sequential(
            Rearrange("b s c m -> b c s m"),
            nn.Conv2d(self.in_channels, 64, kernel_size=9, stride=1, padding="same"),

            nn.Conv2d(64, 128, kernel_size=7, stride=1,
                      padding="same"),
            nn.ReLU(),
            # nn.BatchNorm1d(num_features=128),
            nn.Dropout(p=self.dropout),

            nn.Conv2d(128, 256, kernel_size=5, stride=1,
                      padding="same"),
            nn.ReLU(),
            # nn.BatchNorm1d(num_features=256),
            nn.Dropout(p=self.dropout),

            nn.Conv2d(256, self.window_embedding_dim, kernel_size=3, stride=1,
                      padding="same"),
            nn.ReLU(),
            # nn.BatchNorm1d(num_features=512),
            nn.Dropout(p=self.dropout),
            Rearrange("b c s m -> b s c m"),

            nn.AdaptiveAvgPool2d(output_size=(self.window_embedding_dim, 1)),
            nn.Flatten(start_dim=2),
        )

        self.fnet_encoders = nn.Sequential(
            FNetEncoderBlock(d_in=512, d_hidden=512, d_out=512),
            FNetEncoderBlock(d_in=512, d_hidden=512, d_out=512),
        )
        self.special_tokens = {
            token: i_token
            for i_token, token in enumerate(["start", "end", "mask"])
        }
        self.tokens_embedder = nn.Embedding(len(self.special_tokens), self.window_embedding_dim)

        self.classification = nn.ModuleList()
        for label in self.labels:
            self.classification.add_module(label,
                                           nn.Sequential(
                                               nn.Linear(in_features=self.window_embedding_dim, out_features=1024),
                                               nn.ReLU(),
                                               nn.Dropout(p=self.dropout),

                                               nn.Linear(in_features=1024, out_features=128),
                                               nn.ReLU(),
                                               nn.Dropout(p=self.dropout),

                                               nn.Linear(in_features=128, out_features=2),
                                           ))
        self.float()
        assert device is None or device in {"cuda", "cpu"}
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        self.save_hyperparameters()

    def forward(self, eeg):
        assert eeg.shape[-1] == self.in_channels
        x = eeg  # (b s c)
        # cast from microvolts to volts
        x *= 1e6

        with profiler.record_function("decomposition"):
            x = self.get_mel_spectrogram(x, sampling_rate=self.eeg_sampling_rate,
                                         mels=self.mels,
                                         window_size=9, window_stride=5)  # (b s c m)
            # print(x.max(), x.min())
        with profiler.record_function("preparation"):
            x = self.normalization(x)  # (b s c m)
            # print(x.max(), x.min(), x[0, 0, :4])
            # print(x.max(), x.min())
            x = self.cnn_merge(x)
            # print(x.max(), x.min())
            # exit()

        with profiler.record_function("transformer encoder"):
            # adds special tokens
            # start_token, end_token, mask_token = self.tokens_embedder(torch.as_tensor([
            #     self.special_tokens["start"],
            #     self.special_tokens["end"],
            #     self.special_tokens["mask"],
            # ], device=self.device))
            # if self.training:
            #     # masks a percentage of tokens
            #     for i_batch, batch in enumerate(x):
            #         masked_no = int(((self.mask_perc_min - self.mask_perc_max) * torch.rand(1) + self.mask_perc_max) \
            #                         * batch.shape[0])
            #         mask_ixs = torch.randperm(batch.shape[0])[:int(masked_no)]
            #         x[i_batch, mask_ixs] = mask_token
            # adds start and end token
            # x = torch.cat([start_token.repeat(x.shape[0], 1, 1),
            #                x,
            #                end_token.repeat(x.shape[0], 1, 1)], dim=1)
            # adds positional embeddings
            x += self.get_positional_encodings(length=x.shape[1])
            # x = self.add_positional_encodings(x)
            # encoder_mask = self.generate_square_subsequent_mask(sequence_length=x.shape[1])
            # encoder_mask = torch.zeros((x.shape[1], x.shape[1]), device=self.device, requires_grad=False)
            # print(self.training, x.shape, encoder_mask.shape)
            # x = self.transformer_encoder(x, encoder_mask)
            # x = self.transformer_encoder(x)
            x = self.fnet_encoders(x)

        # with profiler.record_function("transformer decoder"):
        #     # transformer decoder
        #     e = self.target_embedder(torch.arange(len(self.labels), device=self.device, dtype=torch.int)) \
        #         .repeat(x.shape[0], 1, 1)
        #     x = self.transformer_decoder(e, x)

        with profiler.record_function("predictions"):
            labels_pred = torch.stack([net(x[:, 0, :])
                                       for net in self.classification],
                                      dim=1)  # (b l d)
            assert labels_pred.shape[1] == len(self.labels)
            assert len(labels_pred.shape) == 3

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
        self.log(f"acc_train", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("training", 1.0, prog_bar=False, on_step=False, on_epoch=True)
        return {
            "loss": loss,
        }

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
        self.log(f"loss_val", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"acc_val", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("training", 0.0, prog_bar=False)
        return {
            "loss": loss,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_positional_encodings(self, length: int):
        pe = torch.zeros(length, self.window_embedding_dim, device=self.device)
        position = torch.arange(0, length, device=self.device).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, self.window_embedding_dim, 2, dtype=torch.float, device=self.device) *
                              -(math.log(10000.0) / self.window_embedding_dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe

    @staticmethod
    def get_mel_spectrogram(x, sampling_rate: Union[int, float], mels: int,
                            window_size: Optional[int] = None, window_stride: int = 1):
        assert isinstance(x, torch.Tensor) and len(x.shape) in {2, 3}
        assert sampling_rate > 0
        assert window_size is None or isinstance(window_size, int) and window_size >= 1
        if window_size is None:
            window_size = sampling_rate // 2
        assert isinstance(window_stride, int) and window_stride >= 1
        mel_fn = transforms.MelSpectrogram(sample_rate=sampling_rate, f_min=0, f_max=100, n_mels=mels, center=True,
                                           n_fft=x.shape[-1], normalized=True, power=1,
                                           win_length=window_size, hop_length=window_stride).to(x.device)
        mel_spectrogram = mel_fn(
            einops.rearrange(x, "s c -> c s" if len(x.shape) == 2 else "b s c -> b c s"))  # (b c m s)
        mel_spectrogram = einops.rearrange(mel_spectrogram, "b c m s -> b s c m")
        return mel_spectrogram

    @staticmethod
    def plot_mel_spectrogram(spectrogram: torch.Tensor, scale: int = 2):
        assert len(spectrogram.shape) == 3  # s c m
        import matplotlib.pyplot as plt
        import seaborn as sns
        lines = int(np.ceil(np.sqrt(spectrogram.shape[1])))
        fig, axs = plt.subplots(nrows=lines, ncols=lines, figsize=(lines * scale * 1.5, lines * scale),
                                tight_layout=True)
        for i_ax, ax in enumerate(axs.flat):
            if i_ax < spectrogram.shape[1]:
                sns.heatmap(einops.rearrange(spectrogram[:, i_ax, :], "s m -> m s"),
                            vmin=0, vmax=spectrogram.max(),
                            ax=ax)
                ax.set_title(f"Electrode {i_ax}")
                ax.set_xlabel("time")
                ax.set_ylabel("frequency")
            else:
                ax.set_visible(False)
        plt.show(block=False)


class FourierTransformLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = functorch.vmap(torch.fft.fftn)(x).real
        # out = torch.stack([torch.fft.fftn(x[b, :, :]).real
        #                    for b in range(x.shape[0])],
        #                   dim=0)
        return out


class FNetEncoderBlock(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fourier_layer = FourierTransformLayer()
        self.feed_forward_layer = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
            nn.Dropout(p=0.1),
        )

    def forward(self, x):
        x_mixed = self.fourier_layer(x)
        x = x + x_mixed
        x = nn.LayerNorm(x.shape[-1], device=x.device)(x)

        x_feed_forwarded = torch.stack([self.feed_forward_layer(x[:, s, :])
                                        for s in range(x.shape[1])],
                                       dim=1)
        x = x + x_feed_forwarded
        x = nn.LayerNorm(x.shape[-1], device=x.device)(x)
        return x


if __name__ == "__main__":
    eegt = FEEGT(in_channels=32, labels=4, sampling_rate=128, windows_length=0.1,
                 mask_perc_min=0.1, mask_perc_max=0.3)
    x = torch.randn(64, 128, 32)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            out = eegt(x)
    print(prof.key_averages().table(sort_by="cpu_time", row_limit=5))
