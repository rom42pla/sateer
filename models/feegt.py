import gc
import logging
import math
import os
import time
import warnings
from collections import OrderedDict
from os.path import join
from pprint import pformat
from typing import Union, List, Optional, Dict

import functorch
import numpy as np
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
from torch.utils.data import DataLoader
from torchaudio import transforms

from datasets.deap import DEAPDataset
from datasets.dreamer import DREAMERDataset
from datasets.eeg_emrec import EEGClassificationDataset
from datasets.seed import SEEDDataset
from models.layers import AddGaussianNoise, MelSpectrogram, \
    GetSinusoidalPositionalEmbeddings, GetLearnedPositionalEmbeddings, GetTokenTypeEmbeddings, GetUserEmbeddings


class FouriEEGTransformer(pl.LightningModule):
    def __init__(
            self,
            in_channels: int,
            sampling_rate: int,
            labels: Union[int, List[str]],
            labels_classes: Union[int, List[int]] = 2,

            mels: int = 16,
            mel_window_size: Union[int, float] = 1,
            mel_window_stride: Union[int, float] = 0.05,

            users_embeddings: bool = False,

            encoder_only: bool = False,
            mixing_sublayer_type: str = "attention",
            hidden_size: int = 512,
            num_encoders: int = 4,
            num_decoders: int = 4,
            num_attention_heads: int = 8,
            positional_embedding_type: str = "sinusoidal",
            max_position_embeddings: int = 2048,
            dropout_p: Union[int, float] = 0.2,

            data_augmentation: bool = True,
            cropping: bool = True,
            flipping: bool = False,
            noise_strength: Union[int, float] = 0.01,

            learning_rate: float = 2e-4,
            device: Optional[str] = None,
            **kwargs
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

        assert isinstance(labels_classes, int) or isinstance(labels_classes, list), \
            f"the labels classes must be a list of integers or a positive integer, not {labels_classes}"
        if isinstance(labels_classes, list):
            assert all([isinstance(labels_class, int) for labels_class in labels_classes]), \
                f"if the name of the labels are given ({labels_classes}), they must all be strings"
            assert len(labels_classes) == len(labels)
            self.labels_classes = labels_classes
        else:
            assert labels_classes > 0, \
                f"there must be a positive number of classes, not {labels_classes}"
            self.labels_classes = [labels_classes for _ in self.labels]
        assert isinstance(users_embeddings, bool)
        self.users_embeddings: bool = users_embeddings

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
        assert isinstance(encoder_only, bool)
        self.encoder_only = encoder_only
        assert isinstance(hidden_size, int) and hidden_size >= 1
        self.hidden_size = hidden_size
        assert isinstance(num_encoders, int) and num_encoders >= 1
        self.num_encoders: int = num_encoders
        if self.encoder_only is False:
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
        assert isinstance(data_augmentation, bool)
        self.data_augmentation = data_augmentation
        if self.data_augmentation is True:
            assert isinstance(cropping, bool)
            self.cropping = cropping
            assert isinstance(flipping, bool)
            self.flipping = flipping
            assert noise_strength >= 0
            self.noise_strength = noise_strength

        # optimization
        assert isinstance(learning_rate, float) and learning_rate > 0
        self.learning_rate = learning_rate
        assert device is None or device in {"cuda", "cpu"}
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.special_tokens_vocab: Dict[str, int] = {}
        # self.special_tokens_vocab["start"] = len(self.special_tokens_vocab)
        # self.special_tokens_vocab["end"] = len(self.special_tokens_vocab)
        self.special_tokens_vocab["ues"] = len(self.special_tokens_vocab)
        # self.special_tokens_vocab = {
        #     k: i
        #     for i, k in enumerate(["start", "end"])
        #     # for i, k in enumerate(["start", "end", "mask"])
        # }
        if len(self.special_tokens_vocab) > 0:
            self.special_tokens_embedder = nn.Embedding(len(self.special_tokens_vocab), self.hidden_size)
        if self.users_embeddings:
            self.users_embedder = GetUserEmbeddings(hidden_size=self.hidden_size)
        if self.positional_embedding_type == "sinusoidal":
            self.position_embedder_spectrogram = GetSinusoidalPositionalEmbeddings(
                max_position_embeddings=self.max_position_embeddings,
            )
            self.position_embedder = GetSinusoidalPositionalEmbeddings(
                max_position_embeddings=self.max_position_embeddings,
            )
        elif self.positional_embedding_type == "learned":
            self.position_embedder_spectrogram = GetLearnedPositionalEmbeddings(
                max_position_embeddings=self.max_position_embeddings,
                hidden_size=self.in_channels * self.mels
            )
            self.position_embedder = GetLearnedPositionalEmbeddings(
                max_position_embeddings=self.max_position_embeddings,
                hidden_size=self.hidden_size
            )
        self.token_type_embedder = GetTokenTypeEmbeddings(
            hidden_size=self.hidden_size,
        )
        if self.encoder_only is False:
            self.labels_embedder = nn.Embedding(len(self.labels), self.hidden_size)

        self.get_spectrogram = MelSpectrogram(sampling_rate=self.sampling_rate,
                                              min_freq=0, max_freq=50,
                                              mels=self.mels,
                                              window_size=self.mel_window_size,
                                              window_stride=self.mel_window_stride)
        self.merge_mels = nn.Sequential(
            Rearrange("b s c m -> b s (c m)"),
            nn.Linear(self.in_channels * self.mels, self.hidden_size),
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                batch_first=True,
                d_model=self.hidden_size,
                dim_feedforward=self.hidden_size * 4,
                dropout=self.dropout_p,
                activation=F.selu,
                nhead=self.num_attention_heads,
            ),
            num_layers=num_encoders,
        )

        if self.encoder_only is False:
            self.decoder = nn.TransformerDecoder(
                decoder_layer=nn.TransformerDecoderLayer(
                    batch_first=True,
                    d_model=self.hidden_size,
                    dim_feedforward=self.hidden_size * 4,
                    dropout=self.dropout_p,
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

        self.classification = nn.ModuleList()
        for i_label, label in enumerate(self.labels):
            self.classification.add_module(
                label,
                nn.Sequential(OrderedDict([
                    ("linear1", nn.Linear(in_features=self.hidden_size,
                                          out_features=self.hidden_size * 4)),
                    ("activation", nn.SELU()),
                    ("dropout", nn.AlphaDropout(p=self.dropout_p)),
                    ("linear2", nn.Linear(in_features=self.hidden_size * 4,
                                          out_features=self.labels_classes[i_label])),
                ])))

        self.float()
        # self.to(device)
        self.save_hyperparameters()

    def forward(
            self,
            input_eegs: torch.Tensor,
            ids: Optional[Union[int, str, List[Union[int, str]], np.ndarray]] = None
    ):
        # ensures that the inputs are well-defined
        assert input_eegs.shape[-1] == self.in_channels
        assert len(input_eegs.shape) in {2, 3}
        if ids is not None:
            if isinstance(ids, int) or isinstance(ids, str):
                ids = [ids]
            elif isinstance(ids, list):
                assert all([isinstance(id, int) or isinstance(id, str)
                            for id in ids])
            # if isinstance(ids, list):
            # else:
            #     raise TypeError(f"ids must be a string, an integer or a list of such, not {type(ids)}")
            assert len(input_eegs) == len(ids), f"length between eegs and ids mismatch: {len(input_eegs)} != {len(ids)}"

        # makes a fresh copy of the input to avoid errors
        eegs = input_eegs.clone()  # (b s c) or (s c)

        # # cast from microvolts to volts
        # eegs *= 1e6

        # eventually adds a batch dimension
        is_batched = True if len(eegs.shape) == 3 else False
        if not is_batched:
            eegs = einops.rearrange(eegs, "s c -> () s c")

        # initializes variables and structures
        outputs: Dict[str, torch.Tensor] = {}

        # eventually adds data augmentation
        if self.training is True and self.data_augmentation is True:
            with profiler.record_function("data augmentation"):
                if self.cropping is True:
                    crop_amount = int(torch.rand(1, device=eegs.device) * 0.25 * eegs.shape[1])
                    assert crop_amount < eegs.shape[1]
                    if crop_amount > 0:
                        # from left
                        if torch.rand(1, device=eegs.device) <= 0.5:
                            eegs = eegs[:, crop_amount:]
                        # from right
                        else:
                            eegs = eegs[:, :-crop_amount]
                if self.flipping is True:
                    for i_batch, batch in enumerate(eegs):
                        if torch.rand(1, device=eegs.device) <= 0.25:
                            eegs[i_batch] = torch.flip(eegs[i_batch], dims=[0])
                if self.noise_strength > 0:
                    eegs = AddGaussianNoise(strength=self.noise_strength)(eegs)

        # converts the eegs to a spectrogram
        with profiler.record_function("spectrogram"):
            spectrogram = self.get_spectrogram(eegs)  # (b s c m)

        # prepares the spectrogram for the encoder
        with profiler.record_function("preparation"):
            x = self.merge_mels(spectrogram)  # (b s c)
            assert len(x.shape) == 3, f"invalid number of dimensions ({x.shape} must be long 3)"
            assert x.shape[-1] == self.hidden_size, \
                f"invalid hidden size after merging ({x.shape[-1]} != {self.hidden_size})"

        # pass the spectrogram through the encoder
        with profiler.record_function("encoder"):
            # eventually adds positional embeddings and type embeddings
            if self.users_embeddings:
                with profiler.record_function("user embeddings"):
                    users_embeddings = self.users_embedder(ids).type_as(x)  # (b c)
                    # print(users_embeddings.shape)
                    # print(users_embeddings.unsqueeze(1).shape)
                    # print(self.special_tokens_embedder(
                    #         torch.as_tensor([self.special_tokens_vocab["ues"]],
                    #                         device=self.device)).repeat(x.shape[0], 1, 1).shape)
                    # print(x.shape)
                    x = torch.cat([
                        users_embeddings.unsqueeze(1),
                        self.special_tokens_embedder(
                            torch.as_tensor([self.special_tokens_vocab["ues"]],
                                            device=self.device)).repeat(x.shape[0], 1, 1),
                        x], dim=1)
                    x = x + self.token_type_embedder(x, special_tokens_indices=[1])
            # adds the positional embeddings
            x = x + \
                self.position_embedder(x)
            # encoder pass
            x_encoded = self.encoder(x)  # (b s d)
            assert len(x_encoded.shape) == 3, f"invalid number of dimensions ({x_encoded.shape} must be long 3)"
            assert x_encoded.shape[-1] == self.hidden_size, \
                f"invalid hidden size after encoder ({x_encoded.shape[-1]} != {self.hidden_size})"

        # eventually pass the encoded spectrogram to the decoder
        if self.encoder_only is False:
            with profiler.record_function("decoder"):
                # prepares the labels tensor
                label_tokens = self.labels_embedder(
                    torch.as_tensor(list(range(len(self.labels))),
                                    device=x_encoded.device)).repeat(x_encoded.shape[0], 1, 1)  # (b l d)
                # adds the positional embeddings
                label_tokens = label_tokens + \
                               self.position_embedder(label_tokens)  # (b l d)
                # decoder pass
                x_decoded = self.decoder(
                    label_tokens,
                    x_encoded
                )  # (b l d)
                assert len(x_decoded.shape) == 3, f"invalid number of dimensions ({x_decoded.shape} must be long 3)"
                assert x_decoded.shape[-1] == self.hidden_size, \
                    f"invalid hidden size after merging ({x_decoded.shape[-1]} != {self.hidden_size})"

        # makes the predictions using the encoded or decoded data
        with profiler.record_function("predictions"):
            if self.encoder_only is True:
                labels_pred = torch.stack([net(x_encoded[:, 0, :])
                                           for i_label, net in enumerate(self.classification)],
                                          dim=1)  # (b l d)
            else:
                labels_pred = torch.stack([net(x_decoded[:, i_label if self.encoder_only else 0, :])
                                           for i_label, net in enumerate(self.classification)],
                                          dim=1)  # (b l d)

            assert labels_pred.shape[1] == len(self.labels)
            assert len(labels_pred.shape) == 3
            if self.training is False:
                labels_pred = F.softmax(labels_pred, dim=-1)  # (b l d)
            outputs["labels_pred"] = labels_pred
        return outputs

    def training_step(self, batch, batch_idx):
        return self.step(batch)

    def validation_step(self, batch, batch_idx):
        return self.step(batch)

    def step(self, batch):
        # name of the current phase
        phase: str = "train" if self.training is True else "val"
        eegs: torch.Tensor = batch["eegs"]
        labels: torch.Tensor = batch["labels"]
        ids = batch["subject_id"]
        starting_time = time.time()
        net_outputs = self(eegs, ids=ids)  # (b l d)
        results = {
            "loss": sum(
                [F.cross_entropy(net_outputs["labels_pred"][:, i_label, :], labels[:, i_label],
                                 label_smoothing=0.1 if phase == "train" else 0.0)
                 for i_label in range(labels.shape[-1])]),
            "acc": torch.as_tensor(
                [torchmetrics.functional.accuracy(
                    F.softmax(net_outputs["labels_pred"][:, i_label, :], dim=-1) if phase == "val"
                    else net_outputs["labels_pred"][:, i_label, :],
                    labels[:, i_label], average="micro")
                    for i_label in range(labels.shape[-1])]),
            "f1": torch.as_tensor(
                [torchmetrics.functional.f1_score(
                    F.softmax(net_outputs["labels_pred"][:, i_label, :], dim=-1) if phase == "val"
                    else net_outputs["labels_pred"][:, i_label, :],
                    labels[:, i_label], average="micro")
                    for i_label in range(labels.shape[-1])]),
            "precision": torch.as_tensor(
                [torchmetrics.functional.precision(
                    F.softmax(net_outputs["labels_pred"][:, i_label, :], dim=-1) if phase == "val"
                    else net_outputs["labels_pred"][:, i_label, :],
                    labels[:, i_label], average="micro")
                    for i_label in range(labels.shape[-1])]),
            "recall": torch.as_tensor(
                [torchmetrics.functional.recall(
                    F.softmax(net_outputs["labels_pred"][:, i_label, :], dim=-1) if phase == "val"
                    else net_outputs["labels_pred"][:, i_label, :],
                    labels[:, i_label], average="micro")
                    for i_label in range(labels.shape[-1])]),
            "time": time.time() - starting_time,
        }
        return results

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
        # time
        self.log(f"time_{phase}", sum([e["time"] for e in outputs]) / len(outputs),
                 prog_bar=False, sync_dist=True)
        # losses
        self.log(f"loss_{phase}", torch.stack([e["loss"] for e in outputs]).mean(),
                 prog_bar=True if phase == "val" else False, sync_dist=True)
        # classification metrics
        for metric in ["acc", "f1", "precision", "recall"]:
            metric_data = torch.stack([e[metric] for e in outputs])
            self.log(f"{metric}_mean_{phase}", metric_data.mean(),
                     prog_bar=True if metric == "acc" else False, sync_dist=True)
            for i_label, label in enumerate(self.labels):
                self.log(f"{metric}_{label}_{phase}", metric_data[:, i_label].mean(),
                         prog_bar=False, sync_dist=True)
            del metric_data
        del outputs

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.hparams.learning_rate)

    def on_fit_end(self) -> None:
        if self.logger is not None:
            best_epoch = self.logger.logs.groupby('epoch').min().sort_values(by='acc_mean_val',
                                                                             ascending=False).iloc[0:1, :][
                ["loss_train", "loss_val", "acc_mean_train", "acc_mean_val"]]
            print(best_epoch)


if __name__ == "__main__":
    dataset: EEGClassificationDataset = SEEDDataset(
        path=join("..", "..", "..", "datasets", "eeg_emotion_recognition", "seed"),
        window_size=1,
        window_stride=1,
        drop_last=False,
        discretize_labels=True,
        normalize_eegs=True,
    )
    dataloader = DataLoader(dataset, batch_size=64, num_workers=os.cpu_count() - 2, shuffle=True)
    model = FouriEEGTransformer(
        in_channels=len(dataset.electrodes),
        sampling_rate=dataset.sampling_rate,
        labels=dataset.labels,
        labels_classes=dataset.labels_classes,

        users_embeddings=False,

        mels=8,
        mel_window_size=1,
        mel_window_stride=0.05,

        mixing_sublayer_type="attention",
        hidden_size=512,
        num_encoders=4,
        num_decoders=4,
        encoder_only=False,
        num_attention_heads=8,
        positional_embedding_type="sinusoidal",
        max_position_embeddings=2048,
        dropout_p=0.25,
        data_augmentation=True,
    )
    model.training = True
    print(model)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            max_epochs=1,
            check_val_every_n_epoch=1,
            logger=False,
            log_every_n_steps=1,
            enable_progress_bar=True,
            enable_model_summary=False,
            enable_checkpointing=False,
            gradient_clip_val=1,
            limit_train_batches=1,
        )
        trainer.fit(model,
                    train_dataloaders=dataloader)
    print(prof.key_averages(group_by_input_shape=False).table(sort_by="cpu_time", row_limit=8))
    # print(torchvision.models.resnet18())
