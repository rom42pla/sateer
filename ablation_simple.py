import gc
import logging
import re
from datetime import datetime
from os import makedirs
from os.path import join
from pprint import pformat, pprint
from typing import Union, Dict, Iterable

import numpy as np
import optuna
import pandas as pd
import torch
from optuna import Trial
from optuna.samplers import TPESampler

from torch.utils.data import Subset
import pytorch_lightning as pl

from arg_parsers.ablation import get_args
from plots import plot_metrics, plot_ablation
from utils import parse_dataset_class, set_global_seed, save_to_json, init_logger, train
from datasets.eeg_emrec import EEGClassificationDataset
from models.eegst import EEGSpectralTransformer

# sets up the loggers
init_logger()

# retrieves line arguments
args: Dict[str, Union[bool, str, int, float]] = get_args()
logging.info(f"line args:\n{pformat(args)}")

# sets the random seed
set_global_seed(seed=args['seed'])

# sets the logging folder
datetime_str: str = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name: str = f"{datetime_str}"
experiment_path: str = join(args['checkpoints_path'], "ablation", args['dataset_type'], experiment_name)
makedirs(experiment_path)

# saves the line arguments
save_to_json(args, path=join(experiment_path, "line_args.json"))

# sets up the dataset
dataset_class = parse_dataset_class(name=args["dataset_type"])
dataset: EEGClassificationDataset = dataset_class(
    path=args['dataset_path'],
    window_size=args['windows_size'],
    window_stride=args['windows_stride'],
    drop_last=True,
    discretize_labels=True,
    normalize_eegs=True,
)
shuffled_indices = torch.randperm(len(dataset))
dataset_train = Subset(dataset, shuffled_indices[:int(len(dataset) * args['train_set_size'])])
dataset_val = Subset(dataset, shuffled_indices[int(len(dataset) * args['train_set_size']):])

defaults = {}
for i_parameter, (parameter, search_space) in enumerate([
    ("mels", [8, 16, 32]),
    ("mel_window_size", [0.1, 0.2, 0.5, 1]),
    ("mel_window_stride", [0.05, 0.1, 0.25, 0.5]),

    ("users_embeddings", [True, False]),

    ("encoder_only", [True, False]),
    ("hidden_size", [256, 512, 768]),
    ("num_layers", [2, 4, 6]),
    ("positional_embedding_type", ["sinusoidal", "learned"]),
    ("dropout_p", [0.01, 0.1, 0.2]),

    ("shifting", [True, False]),
    ("cropping", [True, False]),
    ("flipping", [True, False]),
    ("noise_strength", [0, 0.001, 0.01]),
    ("spectrogram_time_masking_perc", [0, 0.05, 0.1, 0.15]),
    ("spectrogram_frequency_masking_perc", [0, 0.05, 0.1, 0.15]),
]):
    logging.info(f"started trial with parameter {parameter}")
    for i_value, value in enumerate(search_space):
        model: EEGSpectralTransformer = EEGSpectralTransformer(
            in_channels=len(dataset.electrodes),
            sampling_rate=dataset.sampling_rate,
            labels=dataset.labels,
            num_users=len(dataset.subject_ids),

            **{parameter: value}
        )
        try:
            logs = train(
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                model=model,
                experiment_path=join(experiment_path, parameter, str(i_value)),
                **args
            )
        except Exception as e:
            logging.info(f"skipped trial because of exception\n{e}")
            continue
        logs.to_csv(join(experiment_path, f"{parameter}_{i_value}_logs.csv"))
        save_to_json({
            "parameter": parameter,
            "value": value,
        }, path=join(experiment_path, f"{parameter}_{i_value}_desc.json"))

plot_ablation(experiment_path)
