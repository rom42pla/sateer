import gc
import json
import logging
import os
import random
import re
import warnings
from copy import deepcopy
from os.path import join, isdir, exists
from typing import Dict, Any, List, Optional, Union

import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning
from torch.utils.data import Dataset, DataLoader, Subset

from datasets.amigos import AMIGOSDataset
from datasets.deap import DEAPDataset
from datasets.dreamer import DREAMERDataset
from datasets.eeg_emrec import EEGClassificationDataset
from datasets.seed import SEEDDataset
from loggers.logger import FouriEEGTransformerLogger
from models.sateer import SATEER


def parse_dataset_class(name: str):
    if name == "deap":
        dataset_class = DEAPDataset
    elif name == "dreamer":
        dataset_class = DREAMERDataset
    elif name == "amigos":
        dataset_class = AMIGOSDataset
    elif name == "seed":
        dataset_class = SEEDDataset
    else:
        raise NotImplementedError(f"unknown dataset {name}")
    return dataset_class


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed)


def init_logger() -> None:
    pd.set_option('display.max_columns', None)
    warnings.filterwarnings("ignore", category=LightningDeprecationWarning)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.basicConfig(
        format='\x1b[42m\x1b[30m[%(asctime)s, %(levelname)s]\x1b[0m %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')


def merge_logs(experiment_path: str,
               setting: str) -> pd.DataFrame:
    assert setting in {"within_subject", "cross_subject"}
    logs: pd.DataFrame = pd.DataFrame()
    if setting == "within_subject":
        for subject in [f for f in os.listdir(experiment_path)
                        if isdir(join(experiment_path, f))]:
            for fold in [f for f in os.listdir(join(experiment_path, subject))
                         if re.match(r"fold_[0-9]+", f)]:
                i_fold = int(fold.split("_")[-1])
                if not exists(join(experiment_path, subject, fold, "logs.csv")):
                    continue
                logs_fold = pd.read_csv(join(experiment_path, subject, fold, "logs.csv"))
                logs_fold["subject"] = subject
                logs_fold["fold"] = i_fold
                logs = pd.concat([logs, logs_fold], ignore_index=True)
    else:
        for fold in [f for f in os.listdir(experiment_path)
                     if re.match(r"fold_[0-9]+", f)]:
            i_fold = int(fold.split("_")[-1])
            if not exists(join(experiment_path, fold, "logs.csv")):
                continue
            logs_fold = pd.read_csv(join(experiment_path, fold, "logs.csv"))
            logs_fold["fold"] = i_fold
            logs = pd.concat([logs, logs_fold], ignore_index=True)
    return logs


def init_callbacks(
        progress_bar: bool = True,
        swa: bool = False,
        learning_rate: float = 1e-4,
) -> List[Callback]:
    callbacks: List[Callback] = [
        EarlyStopping(monitor="loss_val", mode="min", min_delta=1e-3, patience=5,
                      verbose=False, check_on_train_epoch_end=False, strict=True),
    ]
    if progress_bar:
        callbacks += [
            TQDMProgressBar(refresh_rate=100),
        ]
    if swa:
        callbacks += [
            StochasticWeightAveraging(swa_lrs=learning_rate),
        ]
    return callbacks


def save_to_json(object: Any, path: str) -> None:
    with open(path, 'w') as fp:
        json.dump(object, fp, indent=4)


def read_json(path: str) -> Dict:
    with open(path, 'r') as fp:
        x = json.load(fp)
    return x


def split_dataset(
        dataset: Dataset,
        train_set_perc: float = 0.8,
):
    assert isinstance(train_set_perc, float)
    assert 0 < train_set_perc < 1
    shuffled_indices = np.random.permutation(len(dataset)).tolist()
    train_indices = shuffled_indices[:int(len(dataset) * train_set_perc)]
    val_indices = shuffled_indices[int(len(dataset) * train_set_perc):]
    assert set(train_indices).isdisjoint(set(val_indices))
    assert set(train_indices).union(set(val_indices)) == set(shuffled_indices)
    dataset_train = Subset(dataset, train_indices)
    dataset_val = Subset(dataset, val_indices)
    return dataset_train, dataset_val


def train_k_fold(
        dataset: Union[Dataset, Subset],
        experiment_path: str,
        k_folds: int = 10,
        batch_size: int = 64,
        max_epochs: int = 1000,
        precision: int = 32,
        auto_lr_finder: bool = False,
        learning_rate: float = 5e-5,
        disable_gradient_clipping: bool = False,
        disable_swa: bool = True,
        progress_bar: bool = True,
        model_name: str = "model",
        **kwargs,
) -> pd.DataFrame:
    # initialize the logs
    logs: pd.DataFrame = pd.DataFrame()
    # initializes the model
    base_model: SATEER = SATEER(
        in_channels=len(dataset.electrodes),
        sampling_rate=dataset.sampling_rate,
        labels=dataset.labels,
        labels_classes=dataset.labels_classes,

        mels=kwargs['mels'],
        mel_window_size=kwargs['mel_window_size'],
        mel_window_stride=kwargs['mel_window_stride'],

        users_embeddings=not kwargs['disable_users_embeddings'],
        num_users=len(dataset.subject_ids),

        encoder_only=kwargs['encoder_only'],
        hidden_size=kwargs['hidden_size'],
        num_encoders=kwargs['num_encoders'],
        num_decoders=kwargs['num_decoders'],
        num_attention_heads=kwargs['num_attention_heads'],
        positional_embedding_type=kwargs['positional_embedding_type'],
        max_position_embeddings=kwargs['max_position_embeddings'],
        dropout_p=kwargs['dropout_p'],

        data_augmentation=not kwargs['disable_data_augmentation'],
        shifting=not kwargs['disable_shifting'],
        cropping=not kwargs['disable_cropping'],
        flipping=not kwargs['disable_flipping'],
        noise_strength=kwargs['noise_strength'],
        spectrogram_time_masking_perc=kwargs['spectrogram_time_masking_perc'],
        spectrogram_frequency_masking_perc=kwargs['spectrogram_frequency_masking_perc'],

        learning_rate=learning_rate,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    # sets up the k_fold
    shuffled_indices = torch.randperm(len(dataset)).tolist()
    fold_starting_indices = torch.linspace(start=0, end=len(dataset),
                                           steps=k_folds + 1, dtype=torch.long).tolist()
    folds_indices = [shuffled_indices[i1:i2]
                     for i1, i2 in zip(fold_starting_indices[:-1], fold_starting_indices[1:])]
    assert len([i for f in folds_indices for i in f]) == len(dataset) \
           and {i for f in folds_indices for i in f} == set(range(len(dataset)))
    # loops over folds
    for i_fold in range(k_folds):
        # makes a fresh copy of the model
        model = deepcopy(base_model)
        # retrieves training and validation sets
        train_indices: List[int] = [i
                                    for fold_no, fold in enumerate(folds_indices)
                                    for i in fold
                                    if fold_no != i_fold]
        val_indices: List[int] = [i
                                  for fold_no, fold in enumerate(folds_indices)
                                  for i in fold
                                  if fold_no == i_fold]
        assert set(train_indices).isdisjoint(set(val_indices))
        assert set(train_indices).union(set(val_indices)) == {i
                                                              for f in folds_indices
                                                              for i in f}
        # builds the dataloaders
        num_workers: int = os.cpu_count() - 1
        if torch.cuda.is_available():
            num_workers: int = (os.cpu_count() // torch.cuda.device_count()) - 1
        dataloader_train: DataLoader = DataLoader(Subset(dataset, train_indices),
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers // 2,
                                                  pin_memory=True if torch.cuda.is_available() else False)
        dataloader_val: DataLoader = DataLoader(Subset(dataset, val_indices),
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=num_workers // 2,
                                                pin_memory=True if torch.cuda.is_available() else False)
        del train_indices, val_indices
        logging.info(f"fold {i_fold + 1} of {k_folds}, "
                     f"|dataloader_train| = {len(dataloader_train)}, "
                     f"|dataloader_val| = {len(dataloader_val)}")
        # initializes the trainer
        accelerator: str = "cpu"
        devices: int = 1
        strategy: Optional[str] = None
        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = torch.cuda.device_count()
            if torch.cuda.device_count() >= 2:
                strategy = "ddp"
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
            max_epochs=max_epochs,
            check_val_every_n_epoch=1,
            logger=FouriEEGTransformerLogger(path=join(experiment_path, f"fold_{i_fold}")),
            log_every_n_steps=1,
            enable_progress_bar=progress_bar,
            enable_model_summary=False,
            enable_checkpointing=False,
            gradient_clip_val=1 if disable_gradient_clipping is False else 0,
            auto_lr_find=False if auto_lr_finder is False else "learning_rate",
            callbacks=init_callbacks(
                progress_bar=progress_bar,
                swa=not disable_swa,
                learning_rate=learning_rate,
            ),
        )

        # eventually selects a starting learning rate
        if auto_lr_finder is True:
            deepcopy(trainer).tune(model,
                                   train_dataloaders=deepcopy(dataloader_train),
                                   val_dataloaders=deepcopy(dataloader_val))
            # learning_rate = float(f'{model.learning_rate:+.1g}')
            logging.info(f"optimal learning rate is {model.learning_rate}")

        # trains the model
        trainer.fit(model,
                    train_dataloaders=dataloader_train,
                    val_dataloaders=dataloader_val)
        assert not trainer.logger.logs.empty
        assert base_model.state_dict().__str__() != model.state_dict().__str__(), \
            f"model not updating"
        fold_logs = deepcopy(trainer.logger.logs)
        fold_logs["fold"] = i_fold
        logs = pd.concat([logs, fold_logs], ignore_index=True)
        # frees some memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del trainer, model, dataloader_train, dataloader_val
        gc.collect()
    return logs


def train(
        dataset_train: Dataset,
        dataset_val: Dataset,
        experiment_path: str,
        model: Optional[pl.LightningModule] = None,
        batch_size: int = 64,
        max_epochs: int = 1000,
        precision: int = 16,
        auto_lr_finder: bool = False,
        disable_gradient_clipping: bool = False,
        disable_swa: bool = True,
        progress_bar: bool = True,
        learning_rate: float = 1e-4,
        save_model: bool = False,
        model_name: str = "model",
        **kwargs,
) -> pd.DataFrame:
    if model is None:
        dataset = dataset_train.dataset
        model = SATEER(
            in_channels=len(dataset.electrodes),
            sampling_rate=dataset.sampling_rate,
            labels=dataset.labels,
            labels_classes=dataset.labels_classes,

            mels=kwargs['mels'],
            mel_window_size=kwargs['mel_window_size'],
            mel_window_stride=kwargs['mel_window_stride'],

            users_embeddings=not kwargs['disable_users_embeddings'],
            num_users=len(dataset.subject_ids),

            encoder_only=kwargs['encoder_only'],
            hidden_size=kwargs['hidden_size'],
            num_encoders=kwargs['num_encoders'],
            num_decoders=kwargs['num_decoders'],
            num_attention_heads=kwargs['num_attention_heads'],
            positional_embedding_type=kwargs['positional_embedding_type'],
            max_position_embeddings=kwargs['max_position_embeddings'],
            dropout_p=kwargs['dropout_p'],

            data_augmentation=not kwargs['disable_data_augmentation'],
            shifting=not kwargs['disable_shifting'],
            cropping=not kwargs['disable_cropping'],
            flipping=not kwargs['disable_flipping'],
            noise_strength=kwargs['noise_strength'],
            spectrogram_time_masking_perc=kwargs['spectrogram_time_masking_perc'],
            spectrogram_frequency_masking_perc=kwargs['spectrogram_frequency_masking_perc'],

            learning_rate=learning_rate,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    initial_weights = deepcopy(model.state_dict().__str__())

    dataloader_train: DataLoader = DataLoader(dataset_train,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=os.cpu_count() - 1,
                                              pin_memory=True if torch.cuda.is_available() else False)
    dataloader_val: DataLoader = DataLoader(dataset_val,
                                            batch_size=batch_size, shuffle=False,
                                            num_workers=os.cpu_count() - 1,
                                            pin_memory=True if torch.cuda.is_available() else False)
    # frees some memory
    gc.collect()

    callbacks = init_callbacks(
        progress_bar=progress_bar,
        swa=not disable_swa,
        learning_rate=learning_rate,
    )
    if save_model:
        callbacks += [
            ModelCheckpoint(
                monitor='acc_mean_val',
                mode="max",
                dirpath=experiment_path,
                filename=model_name + "_{acc_mean_val:.3f}",
                save_top_k=1,
            ),
        ]
        save_to_json(dict(model.hparams), join(experiment_path, "hparams.json"))
    # initializes the trainer
    accelerator: str = "cpu"
    devices: int = 1
    strategy: Optional[str] = None
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = torch.cuda.device_count()
        if torch.cuda.device_count() >= 2:
            strategy = "ddp"
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        logger=FouriEEGTransformerLogger(path=experiment_path),
        log_every_n_steps=1,
        enable_progress_bar=progress_bar,
        enable_model_summary=True if save_model else False,
        enable_checkpointing=save_model,
        gradient_clip_val=1 if disable_gradient_clipping is False else 0,
        auto_lr_find=False if auto_lr_finder is False else "learning_rate",
        callbacks=callbacks
    )
    # trains the model
    trainer.fit(model,
                train_dataloaders=dataloader_train,
                val_dataloaders=dataloader_val)
    assert not trainer.logger.logs.empty
    assert initial_weights != model.state_dict().__str__(), \
        f"model not updating"
    logs: pd.DataFrame = deepcopy(trainer.logger.logs)
    # frees some memory
    del trainer, \
        dataloader_train, dataloader_val
    return logs
