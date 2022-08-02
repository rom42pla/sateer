import gc
import logging
import math
import warnings
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

from models.layers import FouriEncoder, FouriEncoderBlock, FouriDecoder, AddGaussianNoise, MelSpectrogram, \
    GetSinusoidalPositionalEmbeddings, GetLearnedPositionalEmbeddings, GetTokenTypeEmbeddings


class FouriEEGTransformer(pl.LightningModule):
    def __init__(
            self,
            in_channels: int,
            sampling_rate: int,
            labels: Union[int, List[str]],

            mels: int = 16,
            mel_window_size: Union[int, float] = 1,
            mel_window_stride: Union[int, float] = 0.05,

            mixing_sublayer_type: str = "fourier",
            hidden_size: int = 768,
            num_encoders: int = 12,
            num_decoders: int = 12,
            num_attention_heads: int = 8,
            positional_embedding_type: str = "sinusoidal",
            max_position_embeddings: int = 512,
            dropout_p: Union[int, float] = 0.1,

            noise_strength: Union[int, float] = 0,
            # use_masking: bool = True,
            # mask_perc_min: float = 0.05,
            # mask_perc_max: float = 0.3,

            learning_rate: float = 1e-4,
            device: Optional[str] = None
    ):
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
        assert isinstance(mixing_sublayer_type, str)
        self.mixing_sublayer_type = mixing_sublayer_type

        # model architecture
        assert isinstance(hidden_size, int) and hidden_size >= 1
        self.hidden_size = hidden_size
        assert isinstance(num_encoders, int) and num_encoders >= 1
        self.num_encoders: int = num_encoders
        assert isinstance(num_decoders, int) and num_decoders >= 1
        self.num_decoders = num_decoders
        assert isinstance(num_attention_heads, int) and num_attention_heads >= 1
        self.num_attention_heads = num_attention_heads
        assert isinstance(positional_embedding_type, str) and positional_embedding_type in {"sinusoidal", "learned"}
        self.positional_embedding_type = positional_embedding_type
        assert isinstance(max_position_embeddings, int) and max_position_embeddings >= 1
        self.max_position_embeddings = max_position_embeddings
        assert 0 <= dropout_p < 1
        self.dropout_p = dropout_p

        # regularization
        # assert isinstance(use_masking, bool)
        # self.use_masking = use_masking
        # if self.use_masking is True:
        #     assert isinstance(mask_perc_min, float) and 0 <= mask_perc_min < 1
        #     assert isinstance(mask_perc_max, float) and 0 <= mask_perc_max < 1 and mask_perc_max >= mask_perc_min
        #     self.mask_perc_max, self.mask_perc_min = mask_perc_max, mask_perc_min
        # else:
        #     self.mask_perc_max, self.mask_perc_min = None, None
        assert noise_strength >= 0
        self.noise_strength = noise_strength

        # optimization
        assert isinstance(learning_rate, float) and learning_rate > 0
        self.learning_rate = learning_rate
        assert device is None or device in {"cuda", "cpu"}
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.special_tokens_vocab = {
            k: i
            for i, k in enumerate(["start", "end"])
            # for i, k in enumerate(["start", "end", "mask"])
        }
        self.tokens_embedder = nn.Embedding(len(self.special_tokens_vocab), self.hidden_size)
        if self.positional_embedding_type == "sinusoidal":
            self.position_embedder = GetSinusoidalPositionalEmbeddings(
                max_position_embeddings=self.max_position_embeddings,
            )
        elif self.positional_embedding_type == "learned":
            self.position_embedder = GetLearnedPositionalEmbeddings(
                max_position_embeddings=self.max_position_embeddings,
                hidden_size=self.hidden_size
            )
        self.labels_embedder = nn.Embedding(len(self.labels), self.hidden_size)
        self.token_type_embedder = GetTokenTypeEmbeddings(
            hidden_size=self.hidden_size,
        )
        self.add_noise = nn.Sequential(
            AddGaussianNoise(strength=self.noise_strength)
        )

        self.get_spectrogram = MelSpectrogram(sampling_rate=self.sampling_rate,
                                              min_freq=0, max_freq=50,
                                              mels=self.mels,
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
            # nn.Linear(in_features=128//2 + 1, out_features=self.window_embedding_dim),
            # nn.SELU(),
            # nn.Linear(in_features=self.window_embedding_dim, out_features=1),
            # nn.AlphaDropout(self.dropout_p),
            # nn.Linear(in_features=self.mels, out_features=1),
            Rearrange("b s c m -> b s (c m)"),
            FouriEncoderBlock(in_features=self.in_channels * self.mels,
                              mid_features=self.hidden_size * 4,
                              out_features=self.hidden_size,
                              mixing_sublayer_type=self.mixing_sublayer_type,
                              ),
        )

        self.encoder = FouriEncoder(hidden_size=self.hidden_size,
                                    num_encoders=self.num_encoders,
                                    dropout_p=self.dropout_p,
                                    mixing_sublayer_type=self.mixing_sublayer_type,
                                    )
        # self.decoder = FouriDecoder(embeddings_dim=self.window_embedding_dim,
        #                             num_decoders=self.num_decoders,
        #                             dropout_p=self.dropout_p,
        #                             )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                batch_first=True,
                d_model=self.hidden_size,
                dim_feedforward=self.hidden_size * 4,
                dropout=dropout_p,
                activation=F.selu,
                nhead=self.num_attention_heads,
            ),
            num_layers=num_decoders,
        )
        # replaces dropouts with alpha-dropout in the decoder
        for _, module in self.decoder.layers.named_children():
            for attr_str in dir(module):
                target_attr = getattr(module, attr_str)
                if type(target_attr) == nn.Dropout:
                    setattr(module, attr_str,
                            nn.AlphaDropout(self.dropout_p))

        self.classification = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(in_features=self.hidden_size,
                                      out_features=self.hidden_size * 4)),
                ("activation1", nn.SELU()),
                ("dropout", nn.AlphaDropout(p=self.dropout_p)),
                ("linear2", nn.Linear(in_features=self.hidden_size * 4,
                                      out_features=2)),
                # ("reshape", Rearrange("b (c d) -> b c d", c=len(self.labels))),
            ]))
            for _ in range(len(self.labels))
        ])

        self.float()
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
            # if self.training and self.use_masking:
            #     x = self.apply_random_mask(x)
            # adds start and end tokens
            x = self.add_start_end_tokens(x)
            # adds positional embeddings and type embeddings
            x = x + \
                self.position_embedder(x) + \
                self.token_type_embedder(x, special_tokens_indices=[0, -1])  # (b s d)
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

    # def apply_random_mask(self, x: torch.Tensor):
    #     # generates mask token
    #     mask_token = self.tokens_embedder(torch.as_tensor([
    #         self.special_tokens_vocab["mask"],
    #     ], device=self.device))[0]
    #     # applies the mask
    #     mask_rand = torch.rand(*x.shape[:2],
    #                            dtype=torch.float, device=self.device)
    #     mask = (mask_rand >= self.mask_perc_min) * (mask_rand <= self.mask_perc_max)
    #     x[mask] = mask_token
    #     del mask_token, mask_rand, mask
    #     return x

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
        best_epoch = self.logger.logs.groupby('epoch').min().sort_values(by='acc_mean_val',
                                                                         ascending=False).iloc[0:1, :][
            ["loss_train", "loss_val", "acc_mean_train", "acc_mean_val"]]
        print(best_epoch)


if __name__ == "__main__":
    batch_size = 64
    sampling_rate = 128
    seconds = 10
    batch = {
        "eegs": torch.randn(batch_size, seconds * sampling_rate, 32, dtype=torch.float32) * 1e-6,
        "labels": torch.ones(batch_size, 4, dtype=torch.long),
        "sampling_rates": torch.zeros(batch_size, dtype=torch.long) + sampling_rate,
    }
    model = FouriEEGTransformer(
        in_channels=32,
        sampling_rate=sampling_rate,
        labels=4,

        mels=8,
        mel_window_size=1,
        mel_window_stride=0.05,

        mixing_sublayer_type="fourier",

        hidden_size=512,
        num_encoders=4,
        num_decoders=4,
        num_attention_heads=8,
        positional_embedding_type="learned",
        max_position_embeddings=512,
        dropout_p=0.25,
    )
    model.training = True
    print(model)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        model.training_step(batch, 0)
    print(prof.key_averages(group_by_input_shape=False).table(sort_by="cpu_time", row_limit=8))

    # print(torchvision.models.resnet18())
