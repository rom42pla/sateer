import math
from typing import Union, List, Optional

# import julius
import numpy as np
import torch
import torchaudio
import torchvision.models
from einops.layers.torch import Rearrange
from scipy import signal
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
import einops
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity
from torchaudio import transforms


class CNNBaseline(pl.LightningModule):
    def __init__(self, in_channels: int, labels: Union[int, List[str]],
                 sampling_rate: int,
                 mels: int = 8,
                 window_embedding_dim: int = 512,
                 learning_rate: float = 1e-3,
                 dropout: float = 0.1,
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

        assert isinstance(window_embedding_dim, int) and window_embedding_dim >= 1
        self.window_embedding_dim = window_embedding_dim

        assert isinstance(learning_rate, float) and learning_rate > 0
        self.learning_rate = learning_rate

        assert 0 <= dropout < 1
        self.dropout = dropout

        assert isinstance(mels, int) and mels >= 1
        self.mels = 8

        self.cnn_bands = nn.ModuleList()
        for i_band in range(self.mels):
            self.cnn_bands.add_module(f"band_{i_band}",
                                      nn.Sequential(
                                          nn.LayerNorm(self.in_channels),
                                          Rearrange("b s c -> b c s"),

                                          nn.Conv1d(self.in_channels, 64, kernel_size=9, stride=1),

                                          nn.Conv1d(64, 128, kernel_size=7, stride=1),
                                          nn.ReLU(),
                                          # nn.BatchNorm1d(num_features=128),
                                          nn.Dropout(p=self.dropout),

                                          nn.Conv1d(128, 256, kernel_size=5, stride=1),
                                          nn.ReLU(),
                                          # nn.BatchNorm1d(num_features=256),
                                          nn.Dropout(p=self.dropout),

                                          nn.Conv1d(256, self.window_embedding_dim, kernel_size=3, stride=1),
                                          nn.ReLU(),
                                          # nn.BatchNorm1d(num_features=512),
                                          nn.Dropout(p=self.dropout),

                                          nn.AdaptiveAvgPool1d(output_size=(1,)),
                                          nn.Flatten(start_dim=1),
                                      ))
        self.bands_reduction = nn.Sequential(
            nn.Flatten(start_dim=1),

            nn.Linear(in_features=self.window_embedding_dim * self.mels, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),

            nn.Linear(in_features=1024, out_features=self.window_embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )

        self.classification = nn.ModuleList()
        for label in self.labels:
            self.classification.add_module(label,
                                           nn.Sequential(
                                               nn.Linear(in_features=self.window_embedding_dim, out_features=1024),
                                               nn.ReLU(),
                                               nn.BatchNorm1d(num_features=1024),
                                               nn.Dropout(p=self.dropout),

                                               nn.Linear(in_features=1024, out_features=128),
                                               nn.ReLU(),
                                               nn.BatchNorm1d(num_features=128),
                                               nn.Dropout(p=self.dropout),

                                               nn.Linear(in_features=128, out_features=2),
                                           ))
        self.float()
        assert device is None or device in {"cuda", "cpu"}
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        self.save_hyperparameters()

    # def split_in_bands(self, x):
    #     x_bands = julius.split_bands(einops.rearrange(x, "b s c -> b c s").contiguous(),
    #                                  sample_rate=self.eeg_sampling_rate,
    #                                  cutoffs=[4, 8, 13, 30, 50])[1:]  # (n b c s)
    #     x_bands = einops.rearrange(x_bands, "n b c s -> b n s c")
    #     return x_bands

    def forward(self, eeg):
        assert eeg.shape[-1] == self.in_channels
        x = eeg  # (b s c)

        with profiler.record_function("decomposition"):
            x = self.get_mel_spectrogram(x, sampling_rate=self.eeg_sampling_rate,
                                         mels=self.mels,
                                         window_size=9, window_stride=5)  # (b s c m)
            # self.plot_mel_spectrogram(x[0])

        with profiler.record_function("cnns"):
            x = torch.stack([net(x[:, :, :, i_mel])
                             for i_mel, net in enumerate(self.cnn_bands)],
                            dim=1)  # (b n d)
            x = self.bands_reduction(x)

        with profiler.record_function("predictions"):
            labels_pred = torch.stack([net(x)
                                       for i_label, net in enumerate(self.classification)],
                                      dim=1)  # (b l d)
            assert labels_pred.shape[1] == len(self.labels)

        return labels_pred

    @staticmethod
    def get_mel_spectrogram(x, sampling_rate: Union[int, float], mels: int,
                            window_size: Optional[int] = None, window_stride: int = 1):
        assert isinstance(x, torch.Tensor) and len(x.shape) in {2, 3}
        assert sampling_rate > 0
        assert window_size is None or isinstance(window_size, int) and window_size >= 1
        if window_size is None:
            window_size = sampling_rate // 2
        assert isinstance(window_stride, int) and window_stride >= 1
        mel_fn = transforms.MelSpectrogram(sample_rate=sampling_rate, f_min=3, f_max=100, n_mels=mels, center=True,
                                           n_fft=x.shape[-1],
                                           win_length=window_size, hop_length=window_stride).to(x.device)
        mel_spectrogram = mel_fn(
            einops.rearrange(x, "s c -> c s" if len(x.shape) == 2 else "b s c -> b c s"))  # (b c m s)
        mel_spectrogram = einops.rearrange(mel_spectrogram, "b c m s -> b s c m")
        return mel_spectrogram

    @staticmethod
    def plot_mel_spectrogram(spectrogram: torch.Tensor, scale: int = 2):
        assert len(spectrogram.shape) == 3  # s c m
        import matplotlib.pyplot as plt
        from matplotlib import cm
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
        # axs.set_ylabel(ylabel)
        # axs.set_xlabel("frame")
        #     im = axs.flat[i_channel].imshow(einops.rearrange(spectrogram[:, i_channel, :], "s m -> m s"), origin="lower", aspect="auto")
        # fig.colorbar(im)
        # fig.colorbar(im, ax=axs)
        plt.show(block=False)

    @staticmethod
    def wavelet_decompose(x, scales):
        assert isinstance(x, torch.Tensor) and len(x.shape) in {2, 3}
        assert any([isinstance(scales, t) for t in {np.ndarray, torch.Tensor, list}])
        assert all([width > 0 for width in scales])
        # loss of gradients
        x = x.detach().cpu()

        if len(x.shape) == 2:
            x_decomposed = torch.stack([
                torch.as_tensor(signal.cwt(x[:, i_channel], signal.ricker, scales))
                for i_channel in range(x.shape[-1])], dim=-1)  # (w, s, e)
        else:
            x_decomposed = torch.stack([
                torch.stack([
                    torch.as_tensor(signal.cwt(x[i_batch, :, i_channel], signal.ricker, scales))
                    for i_channel in range(x.shape[-1])], dim=-1)
                for i_batch in range(x.shape[0])], dim=0)  # (b, w, s, e)
        return x_decomposed.float()

    def training_step(self, batch, batch_idx):
        eeg, labels = [e for e in batch]  # (b s c), (b l)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)


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
            nn.Dropout1d(),

            nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels,
                      kernel_size=3, stride=2 if self.reduce_output else 1, padding=1),
            nn.BatchNorm1d(num_features=self.in_channels),
            nn.ReLU(),
            nn.Dropout1d(),

            nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1),
            nn.BatchNorm1d(num_features=self.out_channels),
            nn.Dropout1d(),
        )
        self.projection_stream = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels,
                      kernel_size=1, stride=2 if self.reduce_output else 1),
            nn.BatchNorm1d(num_features=self.out_channels),
        )
        self.normalization_stream = nn.Sequential(
            nn.ReLU(),
            nn.Dropout1d(),
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
    x = torch.randn(64, 128, 32) * 1e-6
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            out = model(x)
    print(prof.key_averages().table(sort_by="cpu_time", row_limit=5))
