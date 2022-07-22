import gc
import json
import logging
import os
import random
import warnings
from copy import deepcopy
from os.path import join
from typing import Dict, Any, List, Union

import numpy as np
import pandas as pd

import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping, RichProgressBar, StochasticWeightAveraging
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl

from datasets.deap import DEAPDataset
from datasets.dreamer import DREAMERDataset
from loggers.logger import FouriEEGTransformerLogger
from models.feegt import FouriEEGTransformer


def parse_dataset_class(name: str):
    if name == "deap":
        dataset_class = DEAPDataset
    elif name == "dreamer":
        dataset_class = DREAMERDataset
    else:
        raise NotImplementedError(f"unknown dataset {name}")
    return dataset_class


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_logger() -> None:
    pd.set_option('display.max_columns', None)
    warnings.filterwarnings("ignore", category=LightningDeprecationWarning)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.basicConfig(
        format='\x1b[42m\x1b[30m[%(asctime)s, %(levelname)s]\x1b[0m %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')


def init_callbacks(swa: bool = False) -> List[Callback]:
    callbacks: List[Callback] = [
        EarlyStopping(monitor="loss_val", mode="min", min_delta=1e-4, patience=10,
                      verbose=False, check_on_train_epoch_end=False, strict=True),
        RichProgressBar(
            theme=RichProgressBarTheme(
                description="green_yellow",
                progress_bar="green1",
                progress_bar_finished="green1",
                progress_bar_pulse="#6206E0",
                batch_progress="green_yellow",
                time="grey82",
                processing_speed="grey82",
                metrics="grey82",
            )
        ),
    ]
    if swa:
        callbacks += [
            StochasticWeightAveraging(),
        ]
    return callbacks


def merge_logs(logs: List[Dict[str, Union[pd.DataFrame, int]]]) -> pd.DataFrame:
    merged_logs: pd.DataFrame = pd.DataFrame()
    for experiment_logs in logs:
        log_df: pd.DataFrame = experiment_logs["logs"]
        log_df["fold"] = experiment_logs["fold"]
        if "subject" in experiment_logs.keys():
            log_df["subject"] = experiment_logs["subject"]
        merged_logs = pd.concat([merged_logs, log_df], ignore_index=True)
    return merged_logs


def save_dict(dictionary: Dict[Any, Any], path: str) -> None:
    with open(path, 'w') as fp:
        json.dump(dictionary, fp, indent=4)


def train_k_fold(
        dataset: Dataset, base_model: pl.LightningModule,
        experiment_path: str, benchmark: bool = False,
        k_folds: int = 10,
        batch_size: int = 64,
        max_epochs: int = 1000,
        precision: int = 32,
        auto_lr_finder: bool = False,
        gradient_clipping: bool = False,
        stochastic_weight_average: bool = False,
        **kwargs,
) -> List[Dict[str, Union[int, pd.DataFrame]]]:
    # initialize the logs
    logs: List[Dict[str, Union[int, pd.DataFrame]]] = []
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
        # frees some memory
        gc.collect()
        # makes a fresh copy of the base model
        model = deepcopy(base_model)
        # retrieves training and validation sets
        train_indices: List[int] = [i
                                    for fold_no, fold in enumerate(folds_indices)
                                    for i in fold
                                    if fold_no != i_fold]
        test_indices: List[int] = [i
                                   for fold_no, fold in enumerate(folds_indices)
                                   for i in fold
                                   if fold_no == i_fold]
        assert set(train_indices).isdisjoint(set(test_indices))
        assert set(train_indices).union(set(test_indices)) == {i
                                                               for f in folds_indices
                                                               for i in f}
        # builds the dataloaders
        dataloader_train: DataLoader = DataLoader(Subset(dataset, train_indices),
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=os.cpu_count() - 2,
                                                  pin_memory=True if torch.cuda.is_available() else False)
        dataloader_val: DataLoader = DataLoader(Subset(dataset, test_indices),
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=os.cpu_count() - 2,
                                                pin_memory=True if torch.cuda.is_available() else False)
        logging.info(f"fold {i_fold + 1} of {k_folds}, "
                     f"|dataloader_train| = {len(dataloader_train)}, "
                     f"|dataloader_val| = {len(dataloader_val)}")
        # initializes the trainer
        logger = FouriEEGTransformerLogger(path=join(experiment_path, f"fold_{i_fold}"),
                                           plot=benchmark)
        trainer = pl.Trainer(
            gpus=1 if torch.cuda.is_available() else 0,
            precision=precision,
            # min_epochs=20,
            max_epochs=max_epochs,
            check_val_every_n_epoch=1,
            logger=logger,
            log_every_n_steps=1,
            enable_progress_bar=True,
            enable_model_summary=False,
            enable_checkpointing=False,
            gradient_clip_val=1 if gradient_clipping else 0,
            auto_lr_find=auto_lr_finder,
            callbacks=init_callbacks(swa=stochastic_weight_average),
        )
        if benchmark:
            print(model)
        # eventually selects a starting learning rate
        if auto_lr_finder is True:
            trainer.tune(model,
                         train_dataloaders=dataloader_train,
                         val_dataloaders=dataloader_val)
            logging.info(f"learning rate has been set to {model.learning_rate}")
        # trains the model
        trainer.fit(model,
                    train_dataloaders=dataloader_train,
                    val_dataloaders=dataloader_val)
        assert base_model.state_dict().__str__() != model.state_dict().__str__(), \
            f"model not updating"
        logs += [{
            "logs": logger.logs,
            "fold": i_fold,
        }]
        # frees some memory
        del trainer, model, logger, \
            dataloader_train, dataloader_val
        if benchmark:
            break
    return logs
