import gc
import logging
from datetime import datetime
from os import makedirs
from os.path import join
from pprint import pformat
from typing import Union, Dict

import numpy as np
import optuna
import pandas as pd
import torch
from optuna import Trial

from torch.utils.data import Subset
import pytorch_lightning as pl

from arg_parsers.ablation import get_args
from plots import plot_metrics
from utils import parse_dataset_class, set_global_seed, save_dict, init_logger, train
from datasets.eeg_emrec import EEGClassificationDataset
from models.feegt import FouriEEGTransformer

# sets up the loggers
init_logger()

# retrieves line arguments
args: Dict[str, Union[bool, str, int, float]] = get_args()
logging.info(f"line args:\n{pformat(args)}")

# sets the random seed
set_global_seed(seed=args['seed'])

# sets the logging folder
datetime_str: str = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name: str = f"{datetime_str}_{args['dataset_type']}"
experiment_path: str = join(args['checkpoints_path'], "ablation", experiment_name)
makedirs(experiment_path)

# saves the line arguments
save_dict(dictionary=args, path=join(experiment_path, "line_args.json"))

# sets up the dataset
dataset_class = parse_dataset_class(name=args["dataset_type"])
dataset: EEGClassificationDataset = dataset_class(
    path=args['dataset_path'],
    split_in_windows=True if args['windows_size'] is not None else False,
    window_size=args['windows_size'], drop_last=True,
    discretize_labels=args['discretize_labels'],
    normalize_eegs=args['normalize_eegs'],
)
shuffled_indices = torch.randperm(len(dataset))
dataset_train = Subset(dataset, shuffled_indices[:int(len(dataset) * args['train_set_size'])])
dataset_val = Subset(dataset, shuffled_indices[int(len(dataset) * args['train_set_size']):])

# sets up the structures needed for the experiment
logs: pd.DataFrame = pd.DataFrame()


def objective(trial: Trial):
    trial_args = {
        "num_encoders": trial.suggest_int("num_encoders", 1, 8),
        "window_embedding_dim": trial.suggest_int("window_embedding_dim", -2048, 2048),
        "dropout_p": trial.suggest_float("dropout_p", 0, 0.99),
        "noise_strength": trial.suggest_float("noise_strength", 0, 0.99),
        "use_masking": trial.suggest_categorical("use_masking", [True, False]),
    }
    model: pl.LightningModule = FouriEEGTransformer(
        in_channels=len(dataset.electrodes),
        sampling_rate=dataset.sampling_rate,
        labels=dataset.labels,

        num_encoders=trial_args['num_encoders'],
        window_embedding_dim=trial_args['window_embedding_dim'],
        use_masking=trial_args['use_masking'],
        dropout_p=trial_args['dropout_p'],
        noise_strength=trial_args['noise_strength'],

        learning_rate=args['learning_rate'],
        mels=args['mels'],
        mel_window_size=args['mel_window_size'],
        mel_window_stride=args['mel_window_stride']
    )
    logging.info(f"training")
    logs = train(
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        model=model,
        experiment_path=join(experiment_path, f"trial_{trial.number}"),
        **args
    )
    for k, v in trial_args.items():
        logs[k] = v
    logs.to_csv(join(experiment_path, f"trial_{trial.number}", "logs.csv"))
    best_accuracy = logs.min()["acc_mean_val"]
    return best_accuracy


search_space = {
    "num_encoders": list(range(1, 6 + 1)) if not args['num_encoders'] else args['num_encoders'],
    "window_embedding_dim": [64, 128, 256, 512, 1024] if not args['window_embedding_dim'] else args[
        'window_embedding_dim'],
    "use_masking": [True, False] if args["masking"] is None else args["masking"],
    "dropout_p": np.linspace(0, 0.9, 9) if not args['dropout_p'] else args['dropout_p'],
    "noise_strength": np.linspace(0, 0.9, 9) if not args['noise_strength'] else args['noise_strength']
}
study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space),
                            direction="maximize")
study.optimize(objective)

# frees some memory
del dataset
gc.collect()

# parses the logs into a single dataframe
# logs = merge_logs(experiment_path, setting=args['setting'])
