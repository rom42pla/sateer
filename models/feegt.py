import gc
import logging
import math
from collections import OrderedDict
from pprint import pformat
from typing import Union, List, Optional, Dict

import functorch
import torch
import torchvision
from einops.layers.torch import Rearrange
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
import einops
import torch.autograd.profiler as profiler
from torch.profiler import profile, ProfilerActivity
from torchaudio import transforms

from models.fourinet import FouriEncoder, FouriEncoderBlock, FouriDecoder


class FouriEEGTransformer(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 sampling_rate: int,
                 labels: Union[int, List[str]],

                 window_embedding_dim: int = 512,
                 num_encoders: int = 1,
                 num_decoders: int = 1,
                 dropout_p: Union[int, float] = 0.2,
                 noise_strength: Union[int, float] = 0,

                 learning_rate: float = 1e-4,

                 use_masking: bool = True,
                 mask_perc_min: float = 0.05,
                 mask_perc_max: float = 0.3,

                 mels: int = 8,
                 mel_window_size: Union[int, float] = 1,
                 mel_window_stride: Union[int, float] = 0.05,
                 mix_fourier_with_tokens: bool = True,

                 device: Optional[str] = None):
        super().__init__()

        # metas
        assert isinstance(in_channels, int) and in_channels >= 1
        self.in_channels = in_channels
        assert isinstance(sampling_rate, int) and sampling_rate >= 1
        self.sampling_rate = sampling_rate
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
        assert mel_window_size > 0
        assert mel_window_stride > 0
        self.mel_window_size = mel_window_size
        self.mel_window_stride = mel_window_stride
        assert isinstance(mix_fourier_with_tokens, bool)
        self.mix_fourier_with_tokens = mix_fourier_with_tokens

        # regularization
        assert isinstance(use_masking, bool)
        self.use_masking = use_masking
        if self.use_masking is True:
            assert isinstance(mask_perc_min, float) and 0 <= mask_perc_min < 1
            assert isinstance(mask_perc_max, float) and 0 <= mask_perc_max < 1 and mask_perc_max >= mask_perc_min
            self.mask_perc_max, self.mask_perc_min = mask_perc_max, mask_perc_min
        else:
            self.mask_perc_max, self.mask_perc_min = None, None
        assert 0 <= dropout_p < 1
        self.dropout_p = dropout_p
        assert noise_strength >= 0
        self.noise_strength = noise_strength

        # model architecture
        assert isinstance(num_encoders, int) and num_encoders >= 1
        self.num_encoders: int = num_encoders
        assert isinstance(num_decoders, int) and num_decoders >= 1
        self.num_decoders = num_decoders
        assert isinstance(window_embedding_dim, int) and window_embedding_dim >= 1
        self.window_embedding_dim = window_embedding_dim

        # optimization
        assert isinstance(learning_rate, float) and learning_rate > 0
        self.learning_rate = learning_rate

        self.special_tokens_vocab = {
            k: i
            for i, k in enumerate(["start", "end", "mask"])
        }
        self.tokens_embedder = nn.Embedding(len(self.special_tokens_vocab), self.window_embedding_dim)
        self.labels_embedder = nn.Embedding(len(self.labels), self.window_embedding_dim)
        self.add_noise = nn.Sequential(
            AddGaussianNoise(strength=self.noise_strength)
        )

        self.get_spectrogram = MelSpectrogram(sampling_rate=self.sampling_rate,
                                              min_freq=0, max_freq=50, mels=self.mels,
                                              window_size=self.mel_window_size,
                                              window_stride=self.mel_window_stride)
        # self.merge_mels = nn.Sequential(
        #     Rearrange("b s c m -> b c s m"),
        #     nn.Conv2d(in_channels=self.in_channels, out_channels=128,
        #               kernel_size=7, stride=2, padding=3),
        #     nn.SELU(),
        #
        #     nn.Conv2d(in_channels=128, out_channels=256,
        #               kernel_size=5, stride=2, padding=2),
        #     nn.SELU(),
        #
        #     nn.Conv2d(in_channels=256, out_channels=self.window_embedding_dim,
        #               kernel_size=3, stride=2, padding=1),
        #     nn.SELU(),
        #     Rearrange("b c s m -> b s c m"),
        #     nn.AdaptiveAvgPool2d(output_size=(self.window_embedding_dim, 1)),
        #     Rearrange("b s c m -> b s (c m)"),
        # )
        self.merge_mels = nn.Sequential(
            nn.Linear(in_features=self.mels, out_features=1),
            Rearrange("b s c m -> b s (c m)"),
            FouriEncoderBlock(in_features=self.in_channels,
                              mid_features=self.window_embedding_dim * 4,
                              out_features=self.window_embedding_dim,
                              ),
        )

        self.encoder = FouriEncoder(embeddings_dim=self.window_embedding_dim,
                                    num_encoders=self.num_encoders,
                                    dropout_p=self.dropout_p,
                                    )
        # self.decoder = FouriDecoder(embeddings_dim=self.window_embedding_dim,
        #                             num_decoders=self.num_decoders,
        #                             dropout_p=self.dropout_p,
        #                             )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                batch_first=True,
                d_model=self.window_embedding_dim,
                dim_feedforward=self.window_embedding_dim * 4,
                dropout=dropout_p,
                activation=F.selu,
                nhead=8,
            ),
            num_layers=num_decoders,
        )
        self.classification = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(in_features=self.window_embedding_dim,
                                      out_features=self.window_embedding_dim * 4)),
                ("activation1", nn.SELU()),
                ("dropout", nn.AlphaDropout(p=self.dropout_p)),
                ("linear2", nn.Linear(in_features=self.window_embedding_dim * 4,
                                      out_features=2)),
                # ("reshape", Rearrange("b (c d) -> b c d", c=len(self.labels))),
            ]))
            for _ in range(len(self.labels))
        ])

        self.float()
        assert device is None or device in {"cuda", "cpu"}
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        self.save_hyperparameters()

    def forward(self,
                eegs: torch.Tensor):
        assert eegs.shape[-1] == self.in_channels
        if eegs.device != self.device:
            eegs = eegs.to(self.device)  # (b s c)
        # cast from microvolts to volts
        eegs *= 1e6
        # eventually adds gaussian noise
        if self.training:
            eegs = self.add_noise(eegs)

        with profiler.record_function("spectrogram"):
            spectrogram = self.get_spectrogram(eegs)  # (b s c m)

        with profiler.record_function("preparation"):
            x = self.merge_mels(spectrogram)  # (b s c)

        with profiler.record_function("encoder"):
            # eventually adds masking
            if self.training and self.use_masking:
                x = self.apply_random_mask(x)
            # adds start and end tokens
            x = self.add_start_end_tokens(x)
            # adds positional embeddings
            x = self.add_positional_embeddings(x)  # (b s d)
            # encoder pass
            x_encoded = self.encoder(x)  # (b s d)

        with profiler.record_function("decoder"):
            # adds the labels for the tokens
            label_tokens = self.labels_embedder(
                torch.as_tensor(list(range(len(self.labels))),
                                device=x_encoded.device)).repeat(x_encoded.shape[0], 1, 1)  # (b l d)
            # # adds start and end tokens
            # label_tokens = self.add_start_end_tokens(label_tokens)  # (b l d)
            # # adds positional embeddings
            # label_tokens = self.add_positional_embeddings(label_tokens)  # (b l d)
            x_decoded = self.decoder(
                label_tokens,
                x_encoded
            )  # (b l d)

        with profiler.record_function("predictions"):
            labels_pred = torch.stack([net(x_decoded[:, i_label, :])
                                       for i_label, net in enumerate(self.classification)],
                                      dim=1)  # (b l d)
            assert labels_pred.shape[1] == len(self.labels)
            assert len(labels_pred.shape) == 3
            if self.training is False:
                labels_pred = F.softmax(labels_pred, dim=-1)  # (b l d)
        return labels_pred

    def add_positional_embeddings(self, x: torch.Tensor):
        sequence_length, embeddings_dim = x.shape[-2], x.shape[-1]
        pe = torch.zeros(sequence_length, embeddings_dim, device=self.device)
        position = torch.arange(0, sequence_length, device=self.device).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, embeddings_dim, 2, dtype=torch.float, device=self.device) *
                              -(math.log(10000.0) / embeddings_dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        x = x + pe
        del pe, position, div_term
        return x

    def add_start_end_tokens(self, x: torch.Tensor):
        # generates start and end tokens
        start_token, end_token = self.tokens_embedder(torch.as_tensor([
            self.special_tokens_vocab["start"],
            self.special_tokens_vocab["end"],
        ], device=x.device))
        # adds start and end token
        x = torch.cat([start_token.repeat(x.shape[0], 1, 1),
                       x,
                       end_token.repeat(x.shape[0], 1, 1)], dim=1)  # (b s d)
        del start_token, end_token
        return x

    def apply_random_mask(self, x: torch.Tensor):
        # generates mask token
        mask_token = self.tokens_embedder(torch.as_tensor([
            self.special_tokens_vocab["mask"],
        ], device=self.device))[0]
        # applies the mask
        mask_rand = torch.rand(*x.shape[:2],
                               dtype=torch.float, device=self.device)
        mask = (mask_rand >= self.mask_perc_min) * (mask_rand <= self.mask_perc_max)
        x[mask] = mask_token
        del mask_token, mask_rand, mask
        return x

    def training_step(self, batch, batch_idx):
        return self.step(batch)

    def validation_step(self, batch, batch_idx):
        return self.step(batch)

    def step(self, batch):
        # name of the current phase
        phase: str = "train" if self.training is True else "val"
        eegs: torch.Tensor = batch["eegs"]
        labels: torch.Tensor = batch["labels"]
        labels_pred = self(eegs=eegs)  # (b l d)
        losses = [F.cross_entropy(labels_pred[:, i_label, :], labels[:, i_label],
                                  label_smoothing=0.1 if phase == "train" else 0.0)
                  for i_label in range(labels.shape[-1])]
        accs: torch.Tensor = torch.as_tensor(
            [torchmetrics.functional.accuracy(F.softmax(labels_pred[:, i_label, :], dim=-1) if phase == "val"
                                              else labels_pred[:, i_label, :],
                                              labels[:, i_label], average="micro")
             for i_label in range(labels.shape[-1])])
        return {
            "loss": sum(losses),
            "accs": accs,
        }

    def training_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        self.log_stats(outputs)
        del outputs
        gc.collect()

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        self.log_stats(outputs)
        del outputs
        gc.collect()

    def log_stats(self, outputs: List[Dict[str, torch.Tensor]]):
        # name of the current phase
        phase: str = "train" if self.training is True else "val"
        # loss
        self.log(f"loss_{phase}", torch.stack([e["loss"] for e in outputs]).mean(),
                 prog_bar=True if phase == "val" else False)
        # classification metrics
        accs = torch.stack([e["accs"] for e in outputs])
        self.log(f"acc_mean_{phase}", accs.mean(),
                 prog_bar=True)
        for i_label, label in enumerate(self.labels):
            self.log(f"acc_{label}_{phase}", accs[:, i_label].mean(),
                     prog_bar=False)
        del accs, outputs

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.learning_rate)
        return optimizer

    def on_fit_end(self) -> None:
        logging.info(f"best epoch:\n{pformat(self.logger.logs.sort_values(by='acc_mean_val').iloc[0].to_dict())}")


class AddGaussianNoise(nn.Module):
    def __init__(self, strength: float = 0.1):
        super().__init__()
        assert strength >= 0
        self.strength = strength

    def forward(self, x: torch.Tensor):
        noise = torch.normal(mean=torch.zeros_like(x, device=x.device, requires_grad=False) + x.mean(),
                             std=torch.zeros_like(x, device=x.device, requires_grad=False) + x.std())
        noise = noise * self.strength
        return x + noise


class MelSpectrogram(nn.Module):
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


if __name__ == "__main__":
    batch_size = 64
    sampling_rate = 128
    seconds = 10
    batch = {
        "eegs": torch.randn(batch_size, seconds * sampling_rate, 32, dtype=torch.float32) * 1e-6,
        "labels": torch.ones(batch_size, 4, dtype=torch.long),
        "sampling_rates": torch.zeros(batch_size, dtype=torch.long) + sampling_rate,
    }
    model = FouriEEGTransformer(in_channels=32, sampling_rate=sampling_rate,
                                labels=4, window_embedding_dim=512,
                                num_encoders=2, use_masking=True,
                                mask_perc_min=0.1, mask_perc_max=0.3,
                                mix_fourier_with_tokens=True)
    model.training = True
    print(model)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        model.training_step(batch, 0)
    print(prof.key_averages(group_by_input_shape=False).table(sort_by="cpu_time", row_limit=8))

    # print(torchvision.models.resnet18())
