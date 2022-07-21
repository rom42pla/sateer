import json
import logging
import random
import warnings
from typing import Dict, Any, List, Union

import numpy as np
import pandas as pd

import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping, RichProgressBar, StochasticWeightAveraging
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning

from datasets.deap import DEAPDataset
from datasets.dreamer import DREAMERDataset


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
        EarlyStopping(monitor="acc_mean_val", mode="max", min_delta=1e-4, patience=10,
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
