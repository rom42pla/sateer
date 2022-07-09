import gc
import json
import logging
import random
from datetime import datetime
from os import listdir, makedirs
from os.path import join, isdir
from pprint import pprint

import numpy as np
import pandas as pd
import torch.cuda
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning
from rich.progress import track
from tqdm.autonotebook import tqdm

from arg_parsers import get_training_args
from datasets.deap import DEAPDataset
from models.eegt import EEGT
import pytorch_lightning as pl

import warnings

# suppresses some warnings
warnings.filterwarnings("ignore", category=LightningDeprecationWarning)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# retrieves line arguments
args = get_training_args()

# sets the random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# sets the logging folder
datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"{datetime_str}_{args.setting}_{args.validation}"
makedirs(join(args.checkpoints_path, experiment_name))

# saves the line arguments
with open(join(args.checkpoints_path, experiment_name, "line_args.json"), 'w') as fp:
    json.dump(vars(args), fp, indent=4)
print(f"Line args")
pprint(vars(args))

# sets up the dataset to use
if args.dataset_type == "deap":
    dataset_class = DEAPDataset

print(f"starting {args.setting} training with {args.validation} validation")
if args.setting == "cross_subject":
    print(f"loading dataset {args.dataset_type} from {args.dataset_path}")
    if args.dataset_type == "deap":
        dataset = dataset_class(path=args.dataset_path,
                                split_in_windows=False,
                                windows_size=args.windows_size, drop_last=True,
                                discretize_labels=args.discretize_labels, normalize_eegs=True,
                                validation=args.validation, k_folds=args.k_folds,
                                labels_to_use=["valence", "arousal", "dominance"],
                                batch_size=args.batch_size)
    if args.validation == "k_fold":
        for i_fold in tqdm(range(dataset.k_folds), desc="fold"):
            gc.collect()
            dataset.set_k_fold(i_fold)

            model = EEGT(in_channels=32,
                         labels=dataset.labels_to_use,
                         sampling_rate=dataset.sampling_rate, windows_length=1,
                         num_encoders=args.num_encoders, num_decoders=args.num_decoders,
                         window_embedding_dim=args.window_embedding_dim,
                         learning_rate=args.learning_rate,
                         mask_perc_min=0.05, mask_perc_max=0.2) \
                .to("cuda" if torch.cuda.is_available() else "cpu")
            trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, precision=32,
                                 max_epochs=args.max_epochs, check_val_every_n_epoch=1,
                                 num_sanity_val_steps=args.batch_size,
                                 logger=CSVLogger(args.checkpoints_path, name=experiment_name,
                                                  version=f"fold_{i_fold}"),
                                 enable_progress_bar=True,
                                 enable_model_summary=True,
                                 limit_train_batches=args.limit_train_batches,
                                 limit_val_batches=args.limit_train_batches,
                                 log_every_n_steps=1,
                                 enable_checkpointing=False,
                                 callbacks=[
                                     # ModelCheckpoint(
                                     #     dirpath=join(args.checkpoints_path, experiment_name, f"fold_{i_fold}"),
                                     #     save_top_k=1,
                                     #     monitor="loss_val", mode="min",
                                     #     filename=args.dataset_type + "_{loss_val:.3f}_{epoch:02d}"),
                                     EarlyStopping(monitor="acc_val",
                                                   min_delta=0, patience=10,
                                                   verbose=False, mode="max", check_on_train_epoch_end=False),
                                 ] if args.checkpoints_path is not None else [])
            trainer.fit(model, datamodule=dataset)
            del trainer, model
    elif args.validation == "loso":
        raise NotImplementedError
elif args.setting == "within_subject":
    if args.validation == "k_fold":
        subject_ids = dataset_class.get_subject_ids_static(args.dataset_path)
        for i_subject, subject_id in tqdm(enumerate(subject_ids),
                                          desc="looping through subjects", total=len(subject_ids)):
            dataset = dataset_class(path=args.dataset_path,
                                    subject_ids=subject_id,
                                    split_in_windows=False,
                                    windows_size=args.windows_size, drop_last=True,
                                    discretize_labels=args.discretize_labels, normalize_eegs=True,
                                    validation=args.validation, k_folds=args.k_folds,
                                    labels_to_use=["valence", "arousal", "dominance"],
                                    batch_size=args.batch_size)
            dataset.setup("fit")
            for i_fold in tqdm(range(dataset.k_folds), desc="fold"):
                gc.collect()
                dataset.set_k_fold(i_fold)

                model = EEGT(in_channels=32,
                             labels=dataset.labels_to_use,
                             sampling_rate=dataset.sampling_rate, windows_length=1,
                             num_encoders=args.num_encoders, num_decoders=args.num_decoders,
                             window_embedding_dim=args.window_embedding_dim,
                             learning_rate=args.learning_rate,
                             mask_perc_min=0.05, mask_perc_max=0.2) \
                    .to("cuda" if torch.cuda.is_available() else "cpu")
                trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, precision=32,
                                     max_epochs=args.max_epochs, check_val_every_n_epoch=1,
                                     num_sanity_val_steps=args.batch_size,
                                     logger=CSVLogger(args.checkpoints_path, name=experiment_name,
                                                      version=join(subject_id, f"fold_{i_fold}")),
                                     enable_progress_bar=False,
                                     enable_model_summary=True if (i_subject == 0 and i_fold == 0) else False,
                                     limit_train_batches=args.limit_train_batches,
                                     limit_val_batches=args.limit_train_batches,
                                     log_every_n_steps=1,
                                     enable_checkpointing=False,
                                     callbacks=[
                                         # ModelCheckpoint(
                                         #     dirpath=join(args.checkpoints_path, experiment_name, f"fold_{i_fold}"),
                                         #     save_top_k=1,
                                         #     monitor="loss_val", mode="min",
                                         #     filename=args.dataset_type + "_{loss_val:.3f}_{epoch:02d}"),
                                         EarlyStopping(monitor="acc_val",
                                                       min_delta=0, patience=10,
                                                       verbose=False, mode="max", check_on_train_epoch_end=False),
                                     ] if args.checkpoints_path is not None else [])
                trainer.fit(model, datamodule=dataset)
                del trainer, model
            subject_metrics_dfs = []
            for fold_dir in [f for f in listdir(join(args.checkpoints_path, experiment_name, subject_id))
                             if isdir(join(args.checkpoints_path, experiment_name, subject_id, f))
                                and f.startswith("fold_")]:
                subject_df = pd.read_csv(
                    join(args.checkpoints_path, experiment_name, subject_id, fold_dir, "metrics.csv"))
                subject_df["subject_id"] = subject_id
                subject_df["fold"] = fold_dir
                subject_metrics_dfs += [subject_df]
            metrics_df = pd.concat(subject_metrics_dfs)
            mean_performances_df = metrics_df[
                ["valence_acc_val", "arousal_acc_val", "dominance_acc_val",
                 "acc_val", "subject_id"]].groupby("subject_id").max().mean()
            print(f"Stats for subject {subject_id}:")
            pprint(mean_performances_df.to_dict())

        # logs metrics
        metrics_df = []
        for subject_id in [f for f in listdir(join(args.checkpoints_path, experiment_name))
                           if isdir(join(args.checkpoints_path, experiment_name, f))]:
            for fold_dir in [f for f in listdir(join(args.checkpoints_path, experiment_name, subject_id))
                             if isdir(join(args.checkpoints_path, experiment_name, subject_id, f))
                                and f.startswith("fold_")]:
                subject_df = pd.read_csv(
                    join(args.checkpoints_path, experiment_name, subject_id, fold_dir, "metrics.csv"))
                subject_df["subject_id"] = subject_id
                metrics_df += [subject_df]
        metrics_df = pd.concat(metrics_df)
        metrics_df.to_csv(join(args.checkpoints_path, experiment_name, "metrics.csv"), index=False)
        mean_performances_df = metrics_df[["valence_acc_val", "arousal_acc_val", "dominance_acc_val", "liking_acc_val",
                                           "acc_val", "subject_id"]].groupby("subject_id").max().mean()
        # saves the mean values
        with open(join(args.checkpoints_path, experiment_name, "mean_performances.json"), 'w') as fp:
            json.dump(mean_performances_df.to_dict(), fp, indent=4)
        print(f"Mean performances from all users")
        pprint(mean_performances_df.to_dict())
