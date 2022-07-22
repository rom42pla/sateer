import gc
import logging
import os
from datetime import datetime
from os import makedirs
from os.path import join
from pprint import pformat
from typing import Union, Dict, List

import pandas as pd

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Subset, DataLoader

from arg_parsers import get_training_args
from utils import parse_dataset_class, set_global_seed, save_dict, init_logger, init_callbacks, merge_logs
from datasets.eeg_emrec import EEGClassificationDataset
from loggers.logger import FouriEEGTransformerLogger
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
experiment_name: str = f"{datetime_str}_{args['setting']}_{args['validation']}"
experiment_path: str = join(args['checkpoints_path'], experiment_name)
makedirs(experiment_path)

# saves the line arguments
save_dict(dictionary=args, path=join(experiment_path, "line_args.json"))

# sets up the dataset
dataset_class = parse_dataset_class(name=args["dataset_type"])
dataset: EEGClassificationDataset = dataset_class(
    path=args['dataset_path'],
    # subject_ids_to_use=subject_id,
    split_in_windows=True if args['windows_size'] is not None else False,
    window_size=args['windows_size'], drop_last=True,
    discretize_labels=args['discretize_labels'],
    normalize_eegs=args['normalize_eegs'],
    # validation=args['validation'], k_folds=args['k_folds'],
    # labels_to_use=["valence", "arousal", "dominance"],
    # batch_size=args['batch_size']
)
# dataset.plot_subjects_distribution()

# sets up the structures needed for the experiment
logs: List[Dict[str, Union[int, pd.DataFrame]]] = []
callbacks: List[Callback] = init_callbacks(swa=args['stochastic_weight_average'])

if args['setting'] == "cross_subject":
    # loads the dataset
    logging.info(f"loading dataset {args['dataset_type']} from {args['dataset_path']}")
    dataset: EEGClassificationDataset = dataset_class(
        path=args['dataset_path'],
        split_in_windows=True if args[
                                     'windows_size'] is not None else False,
        window_size=args['windows_size'], drop_last=True,
        discretize_labels=args['discretize_labels'],
        normalize_eegs=args['normalize_eegs'],
        validation=args['validation'], k_folds=args['k_folds'],
        labels_to_use=["valence", "arousal", "dominance"],
        batch_size=args['batch_size']
    )
    if args['validation'] == "k_fold":
        for i_fold in range(args['k_folds']):
            gc.collect()
            dataset.set_k_fold(i_fold)
            if args['model'] == "feegt":
                model: pl.LightningModule = FouriEEGTransformer(
                    in_channels=len(dataset.electrodes),
                    sampling_rate=dataset.sampling_rate,
                    labels=dataset.labels_to_use,
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
            logger = FouriEEGTransformerLogger(path=join(experiment_path,
                                                         f"fold_{i_fold}"),
                                               plot=False)
            trainer = pl.Trainer(
                gpus=1 if torch.cuda.is_available() else 0,
                precision=args['precision'],
                max_epochs=args['max_epochs'],
                check_val_every_n_epoch=1,
                logger=logger,
                log_every_n_steps=1,
                enable_progress_bar=True,
                enable_model_summary=True if i_fold == 0 else False,
                enable_checkpointing=True,
                gradient_clip_val=1 if args['gradient_clipping'] else 0,
                auto_lr_find=args['auto_lr_finder'],
                callbacks=callbacks + [
                    ModelCheckpoint(
                        dirpath=join(experiment_path, f"fold_{i_fold}"),
                        save_top_k=1,
                        monitor="loss_val", mode="min",
                        filename=args['dataset_type'] + "_{loss_val:.3f}_{epoch:02d}")
                ])
            if args['benchmark']:
                print(model)
            if args['auto_lr_finder'] is True:
                trainer.tune(model, datamodule=dataset)
                logging.info(f"learning rate has been set to {model.lr}")
            trainer.fit(model, datamodule=dataset)
            logs += [{
                "logs": logger.logs,
                "fold": i_fold
            }]
            del trainer, model
            if args['benchmark']:
                break
    elif args['validation'] == "loso":
        raise NotImplementedError

elif args['setting'] == "within_subject":
    if args['validation'] == "k_fold":
        for subject_id in dataset.subject_ids:
            dataset_single_subject = Subset(dataset, [i for i, s in enumerate(dataset)
                                                      if dataset.subject_ids[s["subject_id"]] == subject_id])
            assert all([dataset.subject_ids[s["subject_id"]] == subject_id
                        for s in dataset_single_subject])
            # sets up the k_fold
            shuffled_indices = torch.randperm(len(dataset_single_subject)).tolist()
            fold_starting_indices = torch.linspace(start=0, end=len(dataset_single_subject),
                                                   steps=args['k_folds'] + 1, dtype=torch.long).tolist()
            folds_indices = [shuffled_indices[i1:i2]
                             for i1, i2 in zip(fold_starting_indices[:-1], fold_starting_indices[1:])]
            assert len([i for f in folds_indices for i in f]) == len(dataset_single_subject) \
                   and {i for f in folds_indices for i in f} == set(range(len(dataset_single_subject)))
            # loads the dataset
            logging.info(f"subject {subject_id}, "
                         f"|dataset| = {len(dataset_single_subject)}")
            i_subject = dataset.subject_ids.index(subject_id)
            for i_fold in range(args['k_folds']):
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
                dataloader_train: DataLoader = DataLoader(Subset(dataset_single_subject, train_indices),
                                                          batch_size=args['batch_size'], shuffle=True,
                                                          num_workers=os.cpu_count() - 2,
                                                          pin_memory=True if torch.cuda.is_available() else False)
                dataloader_val: DataLoader = DataLoader(Subset(dataset_single_subject, test_indices),
                                                        batch_size=args['batch_size'], shuffle=False,
                                                        num_workers=os.cpu_count() - 2,
                                                        pin_memory=True if torch.cuda.is_available() else False)
                logging.info(f"fold {i_fold + 1} of {args['k_folds']}, "
                             f"|dataloader_train| = {len(dataloader_train)}, "
                             f"|dataloader_val| = {len(dataloader_val)}")
                gc.collect()

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
                logger = FouriEEGTransformerLogger(path=join(experiment_path,
                                                             subject_id, f"fold_{i_fold}"),
                                                   plot=args['benchmark'])
                trainer = pl.Trainer(
                    gpus=1 if torch.cuda.is_available() else 0,
                    precision=args['precision'],
                    # min_epochs=20,
                    max_epochs=args['max_epochs'],
                    check_val_every_n_epoch=1,
                    logger=logger,
                    log_every_n_steps=1,
                    enable_progress_bar=True,
                    enable_model_summary=False,
                    enable_checkpointing=False,
                    gradient_clip_val=1 if args['gradient_clipping'] else 0,
                    auto_lr_find=args['auto_lr_finder'],
                    callbacks=callbacks
                )
                if args['benchmark']:
                    print(model)
                if args['auto_lr_finder'] is True:
                    trainer.tune(model,
                                 train_dataloaders=dataloader_train,
                                 val_dataloaders=dataloader_val)
                    logging.info(f"learning rate has been set to {model.learning_rate}")
                trainer.fit(model,
                            train_dataloaders=dataloader_train,
                            val_dataloaders=dataloader_val)
                logs += [{
                    "logs": logger.logs,
                    "fold": i_fold,
                    "subject": subject_id,
                }]
                del trainer, model, dataloader_train, dataloader_val
                if args['benchmark']:
                    break
            del dataset_single_subject
            if args['benchmark']:
                break
del dataset
# merges all the logs into a single dataframe and saves it
merged_logs: pd.DataFrame = merge_logs(logs=logs)
merged_logs.to_csv(join(experiment_path, "logs.csv"))
