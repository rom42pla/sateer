import math
from collections import OrderedDict
from pprint import pprint
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
    def __init__(self,
                 in_channels: int,
                 labels: Union[int, List[str]],

                 window_embedding_dim: int = 512,
                 num_encoders: int = 1,
                 dropout: float = 0.1,

                 learning_rate: float = 1e-3,

                 use_masking: bool = True, mask_perc_min: float = 0.05, mask_perc_max: float = 0.15,

                 mels: int = 16,

                 device: Optional[str] = None):
        super().__init__()

        # metas
        assert isinstance(in_channels, int) and in_channels >= 1
        self.in_channels = in_channels
        assert isinstance(labels, int) or isinstance(labels, list), \
            f"the labels must be a list of strings or a positive integer, not {labels}"
        if isinstance(labels, list):
            assert all([isinstance(label, str) for label in labels]), \
                f"if the name of the labels are given ({labels}), they must all be strings"
            self.labels = labels
        else:
            assert labels > 0, \
                f"there must be a positive number of labels, not {labels}"
            self.labels = [f"label_{i}" for i in range(labels)]

        # preprocessing
        assert isinstance(mels, int) and mels >= 1, \
            f"the spectrogram must contain at least one mel bank"
        self.mels = mels

        # masking
        assert isinstance(use_masking, bool)
        self.use_masking = use_masking
        if self.use_masking is True:
            assert isinstance(mask_perc_min, float) and 0 <= mask_perc_min < 1
            assert isinstance(mask_perc_max, float) and 0 <= mask_perc_max < 1 and mask_perc_max >= mask_perc_min
            self.mask_perc_max, self.mask_perc_min = mask_perc_max, mask_perc_min

        # model architecture
        assert isinstance(num_encoders, int) and num_encoders >= 1
        self.num_encoders = num_encoders
        assert isinstance(window_embedding_dim, int) and window_embedding_dim >= 1
        self.window_embedding_dim = window_embedding_dim
        assert 0 <= dropout < 1
        self.dropout = dropout

        # optimization
        assert isinstance(learning_rate, float) and learning_rate > 0
        self.learning_rate = learning_rate

        # layers
        self.normalization = nn.Sequential(OrderedDict([
            ("reshaping_1", Rearrange("b s c m -> b c s m")),
            ("conv", nn.Conv2d(self.in_channels, self.in_channels,
                               kernel_size=(1, self.mels * 2 + 1), stride=1, padding=(0, self.mels))),
            ("activation", nn.GELU()),
            ("reshaping_2", Rearrange("b c s m -> b s m c")),
            ("normalization", nn.LayerNorm(self.in_channels)),
            ("reshaping_3", Rearrange("b s m c -> b s c m")),
        ]))
        # self.scale = nn.Parameter(torch.ones(1))

        self.cnn_merge = nn.Sequential(
            Rearrange("b s c m -> b c s m"),
            ResidualBlock(in_channels=self.in_channels, out_channels=256, reduce_output=True),
            ResidualBlock(in_channels=256, out_channels=self.window_embedding_dim, reduce_output=True),

            # nn.Conv2d(self.in_channels, 64,
            #           kernel_size=(9, self.mels // 2 + 1), stride=(2, 1), padding=(4, self.mels // 4)),
            #
            # nn.Conv2d(64, 128,
            #           kernel_size=(7, self.mels // 2 + 1), stride=(2, 1), padding=(3, self.mels // 4)),
            # nn.ReLU(),
            # nn.Dropout(p=self.dropout),
            #
            # nn.Conv2d(128, 256,
            #           kernel_size=(5, self.mels // 2 + 1), stride=(2, 1), padding=(2, self.mels // 4)),
            # nn.ReLU(),
            # nn.Dropout(p=self.dropout),
            #
            # nn.Conv2d(256, self.window_embedding_dim,
            #           kernel_size=(3, self.mels // 2 + 1), stride=(2, 1), padding=(1, self.mels // 4)),
            # nn.ReLU(),
            # nn.Dropout(p=self.dropout),
            Rearrange("b c s m -> b s c m"),

            nn.AdaptiveAvgPool2d(output_size=(self.window_embedding_dim, 1)),
            nn.Flatten(start_dim=2),
        )

        self.fnet_encoders = nn.Sequential(OrderedDict([
            (f"encoder_{i}", FNetEncoderBlock(in_features=self.window_embedding_dim,
                                              mid_features=self.window_embedding_dim,
                                              out_features=self.window_embedding_dim))
            for i in range(self.num_encoders)
        ]))
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

    def forward(self, eegs: torch.Tensor, sampling_rates: Union[float, int, torch.Tensor]):
        assert eegs.shape[-1] == self.in_channels
        x = eegs.to(self.device)  # (b s c)
        # cast from microvolts to volts
        x *= 1e6

        with profiler.record_function("decomposition"):
            x = self.get_mel_spectrogram(x, sampling_rate=sampling_rates,
                                         mels=self.mels,
                                         window_size=0.1, window_stride=None)  # (b s c m)

        with profiler.record_function("preparation"):
            x = self.normalization(x)  # (b s c m)
            # x = x * self.scale
            x = self.cnn_merge(x)

        with profiler.record_function("transformer encoder"):
            # adds special tokens
            start_token, end_token, mask_token = self.tokens_embedder(torch.as_tensor([
                self.special_tokens["start"],
                self.special_tokens["end"],
                self.special_tokens["mask"],
            ], device=self.device))
            if self.training and self.use_masking:
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
            x += self.get_positional_encodings(length=x.shape[1])
            x = self.fnet_encoders(x)

        with profiler.record_function("predictions"):
            labels_pred = torch.stack([net(x[:, 0, :])
                                       for net in self.classification],
                                      dim=1)  # (b l d)
            assert labels_pred.shape[1] == len(self.labels)
            assert len(labels_pred.shape) == 3
        return labels_pred

    def training_step(self, batch, batch_idx):
        # self.log("batch_size", float(len(batch)), prog_bar=False, on_step=True, batch_size=len(batch))
        eegs: torch.Tensor = batch["eegs"]
        labels: torch.Tensor = batch["labels"]
        sampling_rates: torch.Tensor = batch["sampling_rates"]
        labels_pred = self(eegs=eegs, sampling_rates=sampling_rates)  # (b l)
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
        # self.log("batch_size", float(len(batch)), prog_bar=False, on_step=True, batch_size=len(batch))
        # eeg, labels = [e.to(self.device) for e in batch]  # (b s c), (b l)
        eegs: torch.Tensor = batch["eegs"]
        labels: torch.Tensor = batch["labels"]
        sampling_rates: torch.Tensor = batch["sampling_rates"]
        labels_pred = self(eegs=eegs, sampling_rates=sampling_rates)  # (b l)
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
    def get_mel_spectrogram(x, sampling_rate: Union[int, float, torch.Tensor], mels: int,
                            window_size: Union[int, float] = 1,
                            window_stride: Optional[Union[int, float]] = None):
        assert isinstance(x, torch.Tensor) and len(x.shape) in {2, 3}
        assert any([isinstance(sampling_rate, t) for t in (int, float, torch.Tensor)])
        # sets the sampling rate
        if isinstance(sampling_rate, torch.Tensor):
            if not torch.allclose(sampling_rate, sampling_rate[0]):
                raise NotImplementedError(f"all the elements in a batch must have the same sampling rate")
            sampling_rate = sampling_rate[0].item()
        # sets the window
        assert window_size > 0
        window_size = math.floor(window_size * sampling_rate)
        assert window_stride is None or window_stride > 0
        if window_stride is not None:
            window_stride = math.floor(window_stride * sampling_rate)
        else:
            window_stride = 1
        mel_fn = transforms.MelSpectrogram(sample_rate=sampling_rate, f_min=0, f_max=50, n_mels=mels, center=True,
                                           n_fft=x.shape[-1], normalized=True, power=2,
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
        min_value, max_value = spectrogram.min(), spectrogram.max()
        for i_ax, ax in enumerate(axs.flat):
            if i_ax < spectrogram.shape[1]:
                sns.heatmap(einops.rearrange(spectrogram[:, i_ax, :], "s m -> m s"),
                            vmin=min_value, vmax=max_value,
                            ax=ax)
                ax.set_title(f"Electrode {i_ax}")
                ax.set_xlabel("time")
                ax.set_ylabel("mel")
                ax.invert_yaxis()
            else:
                ax.set_visible(False)
        plt.show(block=False)


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
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.in_channels),
            nn.GELU(),

            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                      kernel_size=7, stride=2 if self.reduce_output else 1, padding=3),
            nn.BatchNorm2d(num_features=self.in_channels),
            nn.GELU(),

            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
        )
        self.projection_stream = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                      kernel_size=3, stride=2 if self.reduce_output else 1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
        )
        self.normalization_stream = nn.Sequential(
            nn.GELU()
        )

    def forward(self, x):
        x_transformed = self.reduction_stream(x)
        if self.in_channels != self.out_channels or self.reduce_output:
            x = self.projection_stream(x)
        x = x_transformed + x
        out = self.normalization_stream(x)
        return out


class FNetEncoderBlock(nn.Module):

    def __init__(self, in_features: int, mid_features: int, out_features: int,
                 dropout_p: Union[int, float] = 0):
        super().__init__()

        self.fourier_layer = FourierTransformLayer()
        self.feed_forward_layer = FeedForwardLayer(in_features=in_features,
                                                   mid_features=mid_features,
                                                   out_features=out_features,
                                                   dropout_p=dropout_p)

    def forward(self, x):
        x_mixed = self.fourier_layer(x)
        x = x + x_mixed
        x = F.layer_norm(x, (x.shape[-1],))

        x_feed_forwarded = torch.stack([self.feed_forward_layer(x[:, s, :])
                                        for s in range(x.shape[1])],
                                       dim=1)
        x = x + x_feed_forwarded
        x = F.layer_norm(x, (x.shape[-1],))
        return x


class FourierTransformLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = functorch.vmap(torch.fft.fftn)(x).real
        return x


class FeedForwardLayer(nn.Module):
    def __init__(self, in_features: int, mid_features: Optional[int] = None, out_features: Optional[int] = None,
                 dropout_p: Union[int, float] = 0.1):
        super().__init__()
        assert isinstance(in_features, int) and in_features >= 1
        assert out_features is None or isinstance(out_features, int) and out_features >= 1
        assert mid_features is None or isinstance(mid_features, int) and mid_features >= 1
        self.in_features = in_features
        self.mid_features = mid_features if mid_features is not None else self.in_features
        self.out_features = out_features if out_features is not None else self.in_features
        assert 0 <= dropout_p < 1
        self.dropout_p = dropout_p

        self.linear_1 = nn.Linear(self.in_features, self.mid_features)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(self.mid_features, self.out_features)
        self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    model = FEEGT(in_channels=32, labels=4,
                  mask_perc_min=0.1, mask_perc_max=0.3)
    print(model)
    eegs = torch.randn(256, 128, 32)
    sampling_rates = torch.zeros(256) + 128
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            out = model(eegs, sampling_rates)
    print(prof.key_averages().table(sort_by="cpu_time", row_limit=10))
