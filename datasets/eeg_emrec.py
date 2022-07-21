import time
from abc import abstractmethod, ABC
from copy import deepcopy
from multiprocessing import Pool
from os.path import isdir, join, splitext, basename
from pprint import pprint
from typing import Dict, Optional, Union, List, Set, Tuple

import mne
import torch
from torch.utils.data import Dataset, DataLoader, Subset

import os

import scipy.io as sio
import numpy as np
import pickle
import einops
import pytorch_lightning as pl


class EEGClassificationDataset(pl.LightningDataModule, ABC):

    def __init__(self, path: str,
                 sampling_rate: int,
                 electrodes: Union[int, List[str]],
                 labels: List[str],
                 subject_ids: List[str],

                 split_in_windows: bool = False,
                 window_size: Optional[Union[float, int]] = 1,
                 drop_last: Optional[bool] = False,

                 labels_to_use: Optional[Union[str, List[str]]] = None,
                 subject_ids_to_use: Optional[Union[str, List[str]]] = None,

                 discretize_labels: bool = False,
                 normalize_eegs: bool = False,

                 validation: Optional[str] = None,
                 k_folds: Optional[int] = 10,
                 batch_size: int = 32):
        super().__init__()

        assert isdir(path)
        self.path: str = path

        assert isinstance(sampling_rate, int)
        self.sampling_rate: int = sampling_rate

        assert electrodes is None or isinstance(electrodes, list) or isinstance(electrodes, int)
        if isinstance(electrodes, list):
            assert all((isinstance(x, str) for x in electrodes))
        elif isinstance(electrodes, int):
            self.electrodes = [f"electrode_{x}" for x in range(electrodes)]
        self.electrodes: List[str] = electrodes

        assert isinstance(labels, list)
        assert all((isinstance(x, str) for x in labels))
        self.labels: Dict[str, int] = {label: i
                                       for i, label in enumerate(labels)}

        assert isinstance(subject_ids, list)
        assert all((isinstance(x, str) for x in subject_ids))
        self.subject_ids: List[str] = subject_ids
        self.subject_ids.sort()

        assert isinstance(split_in_windows, bool)
        self.split_in_windows: bool = split_in_windows

        if self.split_in_windows is True:
            assert isinstance(window_size, float) or isinstance(window_size, int) and window_size > 0
            self.window_size: float = float(window_size)  # s
            self.samples_per_window = int(np.floor(self.sampling_rate * self.window_size))

            assert isinstance(drop_last, bool)
            self.drop_last = drop_last

        assert labels_to_use is None or isinstance(labels_to_use, str) or isinstance(labels_to_use, list)
        if labels_to_use is None:
            labels_to_use = list(self.labels.keys())
        elif isinstance(labels_to_use, str):
            labels_to_use = [labels_to_use]
        assert set(labels_to_use).issubset(set(self.labels.keys())), \
            f"one or more labels are not allowed"
        self.labels_to_use: List[str] = labels_to_use

        assert subject_ids_to_use is None or isinstance(subject_ids_to_use, str) or isinstance(subject_ids_to_use, list)
        if subject_ids_to_use is None:
            subject_ids_to_use = deepcopy(self.subject_ids)
        elif isinstance(subject_ids_to_use, str):
            subject_ids_to_use = [subject_ids_to_use]
        assert set(subject_ids_to_use).issubset(set(self.subject_ids)), \
            f"one or more subject ids are not in dataset"
        self.subject_ids_to_use: List[str] = subject_ids_to_use
        self.subject_ids_to_use.sort()

        assert isinstance(discretize_labels, bool)
        self.discretize_labels: bool = discretize_labels
        assert isinstance(normalize_eegs, bool)
        self.normalize_eegs: bool = normalize_eegs

        self.eegs_data, self.labels_data, self.subject_ids_data = self.load_data()
        assert len(self.eegs_data) == len(self.labels_data) == len(self.subject_ids_data)
        assert all([e.shape[-1] == len(self.electrodes) for e in self.eegs_data])
        self.setup_data()

        # sets up k-fold
        assert validation in {None, "k_fold", "loso"}
        self.validation = validation
        if self.validation == "k_fold":
            assert isinstance(k_folds, int) and k_folds >= 1
            self.k_folds, self.current_k_fold_index = k_folds, 0
            shuffled_indices = np.random.permutation(len(self))
            fold_starting_indices = np.linspace(start=0, stop=len(self), num=self.k_folds + 1,
                                                endpoint=True, dtype=int)
            self.folds_indices = [shuffled_indices[i1:i2]
                                  for i1, i2 in zip(fold_starting_indices[:-1], fold_starting_indices[1:])]
            self.set_k_fold(self.current_k_fold_index)
        elif self.validation == "loso":
            self.current_loso_index = 0
            self.subjects_ids_indices = {i_subject: subject_id
                                         for i_subject, subject_id in
                                         enumerate(self.subject_ids_to_use)}

        assert isinstance(batch_size, int) and batch_size >= 1
        self.batch_size: int = batch_size

    def __len__(self) -> int:
        return len(self.eegs_data)

    def __getitem__(self, i: int) -> Dict[str, Union[int, torch.Tensor]]:
        return {
            "sampling_rates": self.sampling_rate,
            # "subject_id": self.subject_ids_data[i],
            "eegs": self.eegs_data[i],
            "labels": self.labels_data[i,
                                       [v for k, v in self.labels.items()
                                        if k in self.labels_to_use]],
        }

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.set_k_fold(self.current_k_fold_index)
            train_indices: List[int] = [i
                                        for i_fold, f in enumerate(self.folds_indices)
                                        for i in f
                                        if i_fold != self.current_k_fold_index]
            test_indices: List[int] = [i
                                       for i_fold, f in enumerate(self.folds_indices)
                                       for i in f
                                       if i_fold == self.current_k_fold_index]
            assert set(train_indices).isdisjoint(set(test_indices))
            assert set(train_indices).union(set(test_indices)) == {i
                                                                   for f in self.folds_indices
                                                                   for i in f}
            self.train_split, self.val_split = Subset(self, train_indices), \
                                               Subset(self, test_indices)
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            pass

        if stage == "predict" or stage is None:
            pass

    @staticmethod
    @abstractmethod
    def get_subject_ids_static(path: str):
        pass

    @abstractmethod
    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        pass

    def setup_data(self) -> None:
        # eventually normalize the eegs
        if self.normalize_eegs:
            # loops through the experiments
            for i_experiment, experiment in enumerate(self.eegs_data):
                # scales the experiment to zero mean and unit variance
                scaler = mne.decoding.Scaler(info=mne.create_info(ch_names=self.electrodes, sfreq=self.sampling_rate,
                                                                  verbose=False, ch_types="eeg"),
                                             scalings="mean")
                scaler.fit(einops.rearrange(experiment, "s c -> () c s"))
                experiment_scaled = einops.rearrange(
                    scaler.transform(einops.rearrange(experiment,
                                                      "s c -> () c s")),
                    "b c s -> s (b c)")
                # scales to microvolts
                experiment_scaled *= 1e-6
                self.eegs_data[i_experiment] = experiment_scaled
        # eventually split the experiments in windows
        if self.split_in_windows:
            eegs_data_windowed: List[np.ndarray] = []
            labels_data_windowed: List[np.ndarray] = []
            subject_ids_data_windowed: List[str] = []
            for i_experiment in range(len(self.eegs_data)):
                for i_window_start in range(0,
                                            len(self.eegs_data[i_experiment]),
                                            self.samples_per_window):
                    window = self.eegs_data[i_experiment][i_window_start:i_window_start + self.samples_per_window, :]
                    if len(window) == self.samples_per_window or self.drop_last is False:
                        eegs_data_windowed += [window]
                        labels_data_windowed += [self.labels_data[i_experiment]]
                        subject_ids_data_windowed += [self.subject_ids_data[i_experiment]]
            self.eegs_data = eegs_data_windowed
            self.labels_data = labels_data_windowed
            self.subject_ids_data = subject_ids_data_windowed
            assert len(self.eegs_data) == len(self.labels_data) == len(self.subject_ids_data)
        # handle labels
        self.labels_data = [[1 if label > 3 else 0 for label in w] if self.discretize_labels else w / 5
                            for w in self.labels_data]
        # eventually pads uneven windows
        windows_size = max([len(w) for w in self.eegs_data])
        self.eegs_data = [w if w.shape[0] == windows_size
                          else np.vstack((w, np.zeros((windows_size - w.shape[0], w.shape[1]))))
                          for w in self.eegs_data]
        assert all([len(w) == windows_size for w in self.eegs_data])
        # converts to tensor
        self.eegs_data: torch.Tensor = torch.stack([torch.from_numpy(w) for w in self.eegs_data]).float()
        self.labels_data: torch.Tensor = torch.stack([torch.as_tensor(w) for w in self.labels_data]).long()

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True,
                          num_workers=os.cpu_count() - 2, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_split, batch_size=self.batch_size, shuffle=False,
                          num_workers=os.cpu_count() - 2, pin_memory=True)

    def set_k_fold(self, i: int) -> None:
        assert isinstance(i, int) and 0 <= i < self.k_folds
        self.current_k_fold_index = i

    def set_loso_index(self, i: int) -> None:
        assert isinstance(i, int) and i in self.subjects_ids_indices.keys()
        self.current_loso_index = i

    def plot_samples(self) -> None:
        raw_mne_array = mne.io.RawArray(einops.rearrange(self[0][0], "s c -> c s"),
                                        info=mne.create_info(ch_names=self.electrodes, sfreq=self.sampling_rate,
                                                             verbose=False, ch_types="eeg"), verbose=False)
        raw_mne_array.plot()
