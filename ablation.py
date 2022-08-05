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
from models.feegt import FouriEEGTransformer

# sets up the loggers
init_logger()

# retrieves line arguments
args: Dict[str, Union[bool, str, int, float]] = get_args()
logging.info(f"line args:\n{pformat(args)}")

# gets testing attributes
tested_parameters = [
    "_".join(k.split("_")[1:]) for k, v in args.items()
    if re.fullmatch(r"test_.*", k) and v is True
]
logging.info(f"tested parameters:\n{pformat(tested_parameters)}")
assert len(tested_parameters) > 0

# sets the random seed
set_global_seed(seed=args['seed'])

# sets the logging folder
datetime_str: str = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name: str = f"{datetime_str}_{args['dataset_type']}_{args['study_name']}"
experiment_path: str = join(args['checkpoints_path'], "ablation", experiment_name)
makedirs(experiment_path)

# saves the line arguments
save_to_json(args, path=join(experiment_path, "line_args.json"))
save_to_json(tested_parameters, path=join(experiment_path, "tested_args.json"))

# sets up the dataset
dataset_class = parse_dataset_class(name=args["dataset_type"])
dataset: EEGClassificationDataset = dataset_class(
    path=args['dataset_path'],
    split_in_windows=True,
    window_size=args['windows_size'],
    window_stride=args['windows_stride'],
    drop_last=True,
    discretize_labels=not args['dont_discretize_labels'],
    normalize_eegs=not args['dont_normalize_eegs'],
)
shuffled_indices = torch.randperm(len(dataset))
dataset_train = Subset(dataset, shuffled_indices[:int(len(dataset) * args['train_set_size'])])
dataset_val = Subset(dataset, shuffled_indices[int(len(dataset) * args['train_set_size']):])

defaults = {}
for parameter, default, search_space in [
    ("users_embeddings", False, [True, False]),

    ("mels", 16, [8, 16, 32, 48, 64]),
    ("mel_window_size", 1, [0.1, 0.2, 0.5, 1]),
    ("mel_window_stride", 0.1, [0.05, 0.1, 0.25, 0.5]),

    ("mixing_sublayer_type", "attention", ["fourier", "identity", "attention"]),

    ("encoder_only", False, [True, False]),
    ("hidden_size", 512, [256, 512, 768]),
    ("num_layers", 4, [2, 4, 6, 8]),
    ("positional_embedding_type", "sinusoidal", ["sinusoidal", "learned"]),
    ("dropout_p", 0.2, [0, 0.1, 0.2, 0.3]),
    ("dreamer_data_augmentation", True, [True, False]),
    ("flipping", False, [True, False]),
    ("cropping", True, [True, False]),
    ("noise_strength", 0.01, [0, 0.01, 0.05]),
]:
    defaults[parameter] = {
        "search_space": search_space,
        "default": default,
    }


def objective(trial: Trial):
    gc.collect()

    trial_args = {}
    for parameter in defaults.keys():
        if parameter in tested_parameters:
            parameter_type = type(defaults[parameter]["default"])
            if args["grid_search"] is False \
                    and parameter_type in [int, float] \
                    and parameter not in {"hidden_size", "mels"}:
                if parameter_type is int:
                    suggested_value = trial.suggest_int(
                        parameter,
                        min(defaults[parameter]["search_space"]),
                        max(defaults[parameter]["search_space"]),
                    )
                elif parameter_type is float:
                    suggested_value = trial.suggest_float(
                        parameter,
                        min(defaults[parameter]["search_space"]),
                        max(defaults[parameter]["search_space"]),
                    )
            else:
                suggested_value = trial.suggest_categorical(parameter,
                                                            defaults[parameter]["search_space"])
            trial_args[parameter] = suggested_value
        else:
            trial_args[parameter] = defaults[parameter]["default"]
    logging.info(f"started trial {trial.number} with parameters:\n{pformat(trial_args)}")

    model: pl.LightningModule = FouriEEGTransformer(
        in_channels=len(dataset.electrodes),
        sampling_rate=dataset.sampling_rate,
        labels=dataset.labels,
        learning_rate=args['learning_rate'],

        num_attention_heads=4,
        num_encoders=trial_args["num_layers"],
        num_decoders=trial_args["num_layers"],
        **trial_args
    )
    logs = train(
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        model=model,
        experiment_path=join(experiment_path, f"trial_{trial.number}"),
        precision=32,
        **args
    )
    for k, v in trial_args.items():
        logs[k] = v
    logs.to_csv(join(experiment_path, f"trial_{trial.number}", "logs.csv"))
    best_accuracy = logs.max()["acc_mean_val"]
    del logs
    return best_accuracy


if args['grid_search'] is True:
    search_space = {
        parameter: values["search_space"] if parameter in tested_parameters else [values["default"]]
        for parameter, values in defaults.items()
    }
    study = optuna.create_study(
        study_name=args['study_name'],
        sampler=optuna.samplers.GridSampler(search_space),
        direction="maximize"
    )
    study.optimize(objective)
else:
    study = optuna.create_study(
        study_name=args['study_name'],
        sampler=TPESampler(),
        direction="maximize",
    )
    study.optimize(objective, n_trials=50)

# frees some memory
del dataset
gc.collect()

# saves the results of the ablation
# logs = pd.DataFrame([
#     {**trial.params, "acc_mean_val": trial.value} for trial in study.trials
# ]).sort_values(by="acc_mean_val", ascending=False)
# logs.to_csv(join(experiment_path, "results.csv"), index=False)
# plot_ablation(logs=logs,
#               experiment_path=experiment_path)
plot_ablation(experiment_path)
