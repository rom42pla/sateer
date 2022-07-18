import math
import time
from collections import OrderedDict
from pprint import pprint
from typing import Union, List, Optional, Dict, Any, Tuple

import functorch
import torch
import torchvision
from einops.layers.torch import Rearrange
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

                 use_masking: bool = True,
                 mask_perc_min: float = 0.05,
                 mask_perc_max: float = 0.15,

                 mels: int = 8,

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

        self.normalize = nn.Sequential(OrderedDict([
            ("reshape1", Rearrange("b s c m -> b c s m")),
            ("conv", nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0, bias=False)),
            ("bn", nn.BatchNorm2d(num_features=self.in_channels)),
            ("activation", nn.GELU()),
            ("reshape2", Rearrange("b c s m -> b s c m")),
        ]))

        self.merge_mels = nn.Sequential(OrderedDict([
            ("reshape", Rearrange("b s c m -> b s (c m)")),
            (f"encoder", FNetEncoderBlock(in_features=self.in_channels * self.mels,
                                          mid_features=self.window_embedding_dim,
                                          out_features=self.window_embedding_dim))
        ]))

        self.fnet_encoders = nn.Sequential(OrderedDict([*[(f"encoder_{i}",
                                                           FNetEncoderBlock(in_features=self.window_embedding_dim,
                                                                            mid_features=self.window_embedding_dim,
                                                                            out_features=self.window_embedding_dim))
                                                          for i in range(self.num_encoders)],
                                                        ("pooler", nn.Linear(in_features=self.window_embedding_dim,
                                                                             out_features=self.window_embedding_dim)),
                                                        ("activation", nn.GELU()),
                                                        ]))
        self.special_tokens = {
            token: i_token
            for i_token, token in enumerate(["start", "end", "mask"])
        }
        self.tokens_embedder = nn.Sequential(
            nn.Embedding(len(self.special_tokens), self.window_embedding_dim),
        )

        self.classification = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(in_features=self.window_embedding_dim,
                                 out_features=len(self.labels) * 2)),
            ("reshape", Rearrange("b (c d) -> b c d", c=len(self.labels))),
        ]))
        self.float()
        assert device is None or device in {"cuda", "cpu"}
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        self.save_hyperparameters()

    def forward(self,
                eegs: torch.Tensor,
                sampling_rates: Union[float, int, torch.Tensor]):
        assert eegs.shape[-1] == self.in_channels
        if eegs.device != self.device:
            eegs = eegs.to(self.device)  # (b s c)
        # cast from microvolts to volts
        eegs *= 1e6

        with profiler.record_function("spectrogram"):
            spectrogram = Spectrogram(sampling_rate=sampling_rates,
                                      min_freq=0, max_freq=40, mels=self.mels,
                                      window_size=1, window_stride=0.1)(eegs)
        # self.plot_mel_spectrogram(spectrogram[0])

        with profiler.record_function("preparation"):
            x = self.normalize(spectrogram)
            x = self.merge_mels(x)  # (b s c)
            # print("sequence", x.shape)

        # generates special tokens
        start_token, end_token, mask_token = self.tokens_embedder(torch.as_tensor([
            self.special_tokens["start"],
            self.special_tokens["end"],
            self.special_tokens["mask"],
        ], device=self.device))
        if self.training and self.use_masking:
            with profiler.record_function("masking"):
                mask_rand = torch.rand(x.shape[:2], dtype=x.dtype, device=self.device)
                mask = (mask_rand >= self.mask_perc_min) * (mask_rand <= self.mask_perc_max)
                x[mask] = mask_token
        # adds start and end token
        x = torch.cat([start_token.repeat(x.shape[0], 1, 1),
                       x,
                       end_token.repeat(x.shape[0], 1, 1)], dim=1)
        # adds positional embeddings
        x += self.get_positional_encodings(length=x.shape[1])

        with profiler.record_function("transformer encoder"):
            x = self.fnet_encoders(x)

        with profiler.record_function("predictions"):
            labels_pred = self.classification(x[:, 0, :])
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
        losses = [F.cross_entropy(labels_pred[:, i_label, :], labels[:, i_label], label_smoothing=0.1)
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

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.learning_rate, amsgrad=False)
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
    def plot_mel_spectrogram(spectrogram: torch.Tensor, scale: int = 2):
        assert len(spectrogram.shape) == 3  # s c m
        import matplotlib.pyplot as plt
        lines = int(math.ceil(math.sqrt(spectrogram.shape[1])))
        fig, axs = plt.subplots(nrows=lines, ncols=lines, figsize=(lines * scale * 1.5, lines * scale),
                                tight_layout=True)
        min_value, max_value = spectrogram.min(), spectrogram.max()

        for i_ax, ax in enumerate(axs.flat):
            if i_ax < spectrogram.shape[1]:
                im = ax.imshow(einops.rearrange(spectrogram[:, i_ax, :], "s m -> m s"),
                               vmin=min_value, vmax=max_value, aspect="auto", cmap=plt.get_cmap("hot"))
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax, orientation="vertical")
                ax.set_title(f"electrode {i_ax}")
                ax.set_xlabel("sample")
                ax.set_ylabel("mels")
                ax.invert_yaxis()
            else:
                ax.set_visible(False)
        plt.show(block=False)


class Spectrogram(nn.Module):
    def __init__(self,
                 sampling_rate: Union[int, float, torch.Tensor],
                 window_size: Union[int, float] = 1,
                 window_stride: Optional[Union[int, float]] = None,
                 min_freq: int = 0,
                 max_freq: int = 50,
                 mels: int = 8,
                 ):
        super().__init__()
        # sampling rate
        assert any([isinstance(sampling_rate, t) for t in (int, float, torch.Tensor)])
        if isinstance(sampling_rate, torch.Tensor):
            if not torch.allclose(sampling_rate, sampling_rate[0]):
                raise NotImplementedError(f"all the elements in a batch must have the same sampling rate")
            sampling_rate = sampling_rate[0].detach().item()
        self.sampling_rate = sampling_rate
        # frequencies
        assert isinstance(min_freq, int) and isinstance(max_freq, int) and \
               0 <= min_freq <= max_freq
        self.min_freq: int = min_freq
        self.max_freq: int = max_freq
        assert isinstance(mels, int) \
               and mels > 0
        self.mels = mels

        # window
        assert window_size > 0
        self.window_size = math.floor(window_size * self.sampling_rate)
        assert window_stride is None or window_stride > 0
        if window_stride is not None:
            self.window_stride = math.floor(window_stride * self.sampling_rate)
        else:
            self.window_stride = 1

    def forward(self, eegs: torch.Tensor):
        assert isinstance(eegs, torch.Tensor) and len(eegs.shape) in {2, 3}
        eegs = einops.rearrange(eegs, "s c -> c s" if len(eegs.shape) == 2 else "b s c -> b c s")
        mel_fn = transforms.MelSpectrogram(
            sample_rate=self.sampling_rate,
            f_min=self.min_freq, f_max=self.max_freq,
            n_mels=self.mels, center=True,
            n_fft=eegs.shape[-1],
            normalized=True, power=2,
            win_length=self.window_size,
            hop_length=self.window_stride,
            # pad=self.window_stride // 2,
            pad=0,
        ).to(eegs.device)
        spectrogram = mel_fn(eegs)  # (b c m s)
        spectrogram = einops.rearrange(spectrogram, "b c m s -> b s c m")
        return spectrogram


class FNetEncoderBlock(nn.Module):

    def __init__(self, in_features: int, mid_features: int, out_features: int,
                 dropout_p: Union[int, float] = 0):
        super().__init__()
        assert isinstance(in_features, int) and in_features >= 1
        assert isinstance(mid_features, int) and mid_features >= 1
        assert isinstance(out_features, int) and out_features >= 1
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = out_features
        assert 0 <= dropout_p < 1
        self.dropout_p = dropout_p

        self.fourier_layer = FourierTransformLayer()
        self.feed_forward_layer = FeedForwardLayer(in_features=self.in_features,
                                                   mid_features=self.mid_features,
                                                   out_features=self.out_features,
                                                   dropout_p=dropout_p)

    def forward(self, x):
        # fourier pass
        x_fourier = self.fourier_layer(x)
        x = F.layer_norm(x + x_fourier,
                         (x.shape[-1],))
        # fc pass
        # x_forwarded = torch.stack([self.feed_forward_layer(x[:, s, :])
        #                            for s in range(x.shape[1])],
        #                           dim=1)
        x_forwarded = self.feed_forward_layer(x)
        x = x_forwarded if self.in_features != self.out_features else (x + x_forwarded)
        x = F.layer_norm(x, (x.shape[-1],))
        return x


class FourierTransformLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = functorch.vmap(torch.fft.fftn)(x).real
        return x


class FeedForwardLayer(nn.Module):
    def __init__(self, in_features: int, mid_features: int, out_features: int,
                 dropout_p: Union[int, float] = 0.1):
        super().__init__()
        assert isinstance(in_features, int) and in_features >= 1
        assert isinstance(mid_features, int) and mid_features >= 1
        assert isinstance(out_features, int) and out_features >= 1
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = out_features
        assert 0 <= dropout_p < 1
        self.dropout_p = dropout_p

        self.linear_1 = nn.Linear(self.in_features, self.mid_features)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(self.mid_features, self.out_features)
        self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        # x = self.activation(x)
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    # torch.backends.cudnn.benchmark = True
    model = FEEGT(in_channels=32, labels=4, window_embedding_dim=512,
                  num_encoders=2, use_masking=True,
                  mask_perc_min=0.1, mask_perc_max=0.3)
    batch_size = 2048
    sampling_rate = 128
    seconds = 1
    batch = {
        "eegs": torch.randn(batch_size, seconds * sampling_rate, 32, dtype=torch.float32),
        "labels": torch.ones(batch_size, 4, dtype=torch.long),
        "sampling_rates": torch.zeros(batch_size, dtype=torch.long) + sampling_rate,
    }
    model.training = True

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            # out = model(batch["eegs"], batch["sampling_rates"])
            model.training_step(batch, 0)
    print(prof.key_averages(group_by_input_shape=False).table(sort_by="cpu_time", row_limit=10))
    # print(model)
    # print(torchvision.models.resnet18())
