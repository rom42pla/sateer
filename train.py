import gc
import logging
import os
from datetime import datetime
from os import makedirs
from os.path import join, exists
from pprint import pformat
from typing import Union, Dict

import pandas as pd

import pytorch_lightning as pl
from torch.utils.data import Subset

from arg_parsers import get_training_args
from utils import parse_dataset_class, set_global_seed, save_dict, init_logger, train_k_fold
from datasets.eeg_emrec import EEGClassificationDataset
from models.feegt import FouriEEGTransformer

# sets up the loggers
init_logger()

# retrieves line arguments
args: Dict[str, Union[bool, str, int, float]] = get_training_args()
logging.info(f"line args:\n{pformat(args)}")

# sets the random seed
set_global_seed(seed=args['seed'])

# sets the logging folder
datetime_str: str = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name: str = f"{datetime_str}_{args['setting']}_{args['validation']}_{args['dataset_type']}"
experiment_path: str = join(args['checkpoints_path'], experiment_name)
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
# dataset.plot_subjects_distribution()

# sets up the model
if args['model'] == "feegt":
    model: pl.LightningModule = FouriEEGTransformer(
        in_channels=len(dataset.electrodes),
        sampling_rate=dataset.sampling_rate,
        labels=dataset.labels,
        num_encoders=args['num_encoders'],
        window_embedding_dim=args['window_embedding_dim'],
        use_masking=not args['disable_masking'],
        learning_rate=args['learning_rate'],
        dropout_p=args['dropout_p'],
        noise_strength=args['noise_strength'],
        mels=args['mels'],
        mel_window_size=args['mel_window_size'],
        mel_window_stride=args['mel_window_stride']
    )
else:
    raise NotImplementedError

# sets up the structures needed for the experiment
logs: pd.DataFrame = pd.DataFrame()

if args['setting'] == "cross_subject":
    if args['validation'] == "k_fold":
        # starts the kfold training
        logging.info(f"training on {args['dataset_type']} dataset "
                     f"({len(dataset)} samples)")
        logs_k_fold = train_k_fold(dataset=dataset, base_model=model,
                                   experiment_path=experiment_path,
                                   **args)
        # saves the logs
        logs = pd.concat([logs, logs_k_fold], ignore_index=True)
        logs.to_csv(join(experiment_path, "logs_tmp.csv"), index=False)
    elif args['validation'] == "loso":
        raise NotImplementedError

elif args['setting'] == "within_subject":
    if args['validation'] == "k_fold":
        for i_subject, subject_id in enumerate(dataset.subject_ids):
            # frees some memory
            gc.collect()
            # retrieves the samples for a single subject
            dataset_single_subject = Subset(dataset, [i for i, s in enumerate(dataset)
                                                      if dataset.subject_ids[s["subject_id"]] == subject_id])
            assert all([dataset.subject_ids[s["subject_id"]] == subject_id
                        for s in dataset_single_subject])
            # starts the kfold training
            logging.info(f"training on {args['dataset_type']}, subject {subject_id} "
                         f"({i_subject + 1}/{len(dataset.subject_ids)}, {len(dataset_single_subject)} samples)")
            logs_k_fold = train_k_fold(dataset=dataset_single_subject, base_model=model,
                                       experiment_path=join(experiment_path, subject_id),
                                       **args)
            # saves the logs
            logs_k_fold["subject"] = subject_id
            logs = pd.concat([logs, logs_k_fold], ignore_index=True)
            logs.to_csv(join(experiment_path, "logs_tmp.csv"), index=False)
            # frees some memory
            del dataset_single_subject
    elif args['validation'] == "loso":
        raise NotImplementedError
# frees some memory
del dataset
gc.collect()

# merges all the logs into a single dataframe and saves it
logs.to_csv(join(experiment_path, "logs.csv"), index=False)
logging.info(f"logs saved to {join(experiment_path, 'logs.csv')}")
# deletes the temporary log file
if exists(join(experiment_path, "logs_tmp.csv")):
    os.remove(join(experiment_path, "logs_tmp.csv"))
