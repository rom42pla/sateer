from copy import deepcopy
from multiprocessing import Pool
from os.path import isdir, join, splitext, basename
from pprint import pprint
from typing import Dict, Optional, Union, List, Set

import torch
from torch.utils.data import Dataset, DataLoader, Subset

import os

import numpy as np
import pickle
import einops
from tqdm import tqdm
import pytorch_lightning as pl


class DEAPDataset(pl.LightningDataModule):
    def __init__(self, path: str, windows_size: Union[float, int] = 1, drop_last: bool = True,
                 subject_ids: Optional[Union[str, List[str]]] = None,
                 labels_to_use: Optional[List[str]] = None,
                 discretize_labels: bool = False, normalize_eegs: bool = False,
                 validation: Optional[str] = None, k_folds: Optional[int] = 10,
                 batch_size: int = 32):
        super().__init__()
        assert isdir(path)
        self.path: str = path

        assert isinstance(discretize_labels, bool)
        self.discretize_labels = discretize_labels
        assert isinstance(normalize_eegs, bool)
        self.normalize_eegs = normalize_eegs

        assert isinstance(windows_size, float) or isinstance(windows_size, int) and windows_size > 0
        self.windows_size: float = float(windows_size)  # s
        self.sampling_rate: int = 128  # Hz
        self.in_channels: int = 32
        self.labels: Dict[str, int] = {"valence": 0,
                                       "arousal": 1,
                                       "dominance": 2,
                                       "liking": 3}  # valence, arousal, dominance, liking
        self.samples_per_window = int(np.floor(self.sampling_rate * self.windows_size))

        assert isinstance(drop_last, bool)
        self.drop_last = drop_last

        assert isinstance(batch_size, int)
        self.batch_size = batch_size

        assert subject_ids is None or isinstance(subject_ids, str) or isinstance(subject_ids, list)
        if isinstance(subject_ids, str):
            subject_ids = [subject_ids]
        if subject_ids is not None:
            assert set(subject_ids).issubset(
                DEAPDataset.get_subject_ids_static(path)), f"one or more subject ids are not in dataset"
            self.subject_ids: List[str] = subject_ids
        else:
            self.subject_ids: List[str] = self.get_subject_ids()

        if labels_to_use is not None:
            assert isinstance(labels_to_use, list)
            for label in labels_to_use:
                assert label in self.labels
            self.labels_to_use = labels_to_use

        self.prepare_data()

        # sets up k-fold
        assert validation in {None, "k_fold", "loso"}
        self.validation = validation
        if self.validation == "k_fold":
            assert k_folds is not None
            assert isinstance(k_folds, int) and k_folds >= 1
            self.k_folds, self.current_k_fold_index = k_folds, 0
            shuffled_indices = np.random.permutation(len(self))
            fold_starting_indices = np.linspace(start=0, stop=len(self), num=self.k_folds + 1,
                                                endpoint=True, dtype=int)
            self.folds_indices = [shuffled_indices[i1:i2]
                                  for i1, i2 in zip(fold_starting_indices[:-1], fold_starting_indices[1:])]
        elif self.validation == "loso":
            self.current_loso_index = 0
            self.subjects_ids_indices = {i_subject: subject_id
                                         for i_subject, subject_id in
                                         enumerate(DEAPDataset.get_subject_ids_static(path=self.path))}

    def __len__(self):
        return len(self.eeg_windows)

    def __getitem__(self, idx):
        return self.eeg_windows[idx], self.label_windows[idx]

    def prepare_data(self) -> None:
        global parse_eegs

        def parse_eegs(path) -> Optional[Dict[str, List[np.ndarray]]]:
            subject_id = basename(splitext(path)[0])
            data = {"eegs": [], "labels": [], "subject_id": []}
            subject_data_dict: Dict[str, np.ndarray] = pickle.load(open(path, "rb"), encoding='latin1')
            subject_eegs: np.ndarray = einops.rearrange(subject_data_dict["data"],
                                                        "v e s -> v s e")[:, :self.sampling_rate * 60, :32]
            subject_labels: np.ndarray = subject_data_dict["labels"]
            for i_experiment in range(subject_eegs.shape[-1]):
                # split the eegs in windows
                windows = np.split(subject_eegs[i_experiment],
                                   np.arange(self.samples_per_window, subject_eegs[i_experiment].shape[0],
                                             self.samples_per_window),
                                   0)
                # adjusts the labels
                labels = subject_labels[i_experiment]
                labels = labels[[self.labels[l] for l in self.labels_to_use]]
                labels = np.tile(labels, reps=(len(windows), 1))
                assert np.isclose(labels, labels[0]).all()
                assert len(windows) == len(labels)
                # appends the windows
                data["eegs"] += windows
                data["labels"] += [l for l in labels]
                data["subject_id"] += [subject_id for _ in windows]
            return data

        eegs_path = [join(self.path, "data_preprocessed_python", s)
                     for s in os.listdir(join(self.path, "data_preprocessed_python"))
                     if basename(splitext(s)[0]) in self.subject_ids]
        with Pool(processes=len(self.subject_ids)) as pool:
            data_pool = pool.map(parse_eegs, eegs_path)
            data_pool = [d for d in data_pool if d is not None]
            self.eeg_windows: List[np.ndarray] = [eeg for data in data_pool for eeg in data["eegs"]]
            self.label_windows: List[np.ndarray] = [labels for data in data_pool for labels in data["labels"]]
            self.subject_ids_windows: List[np.ndarray] = [labels for data in data_pool for labels in data["subject_id"]]
        assert len(self.eeg_windows) == len(self.label_windows)
        # eventually drops uneven windows
        if self.drop_last is True:
            i_windows = [i for i, w in enumerate(self.eeg_windows)
                         if w.shape[0] == self.samples_per_window]
            self.eeg_windows, self.label_windows, self.subject_ids_windows = [self.eeg_windows[i] for i in i_windows], \
                                                                             [self.label_windows[i] for i in
                                                                              i_windows], \
                                                                             [self.subject_ids_windows[i] for i in
                                                                              i_windows]
        # eventually pads uneven windows
        else:
            self.eeg_windows = [w if w.shape[0] == self.samples_per_window
                                else np.vstack((w, np.zeros((self.samples_per_window - w.shape[0], w.shape[1]))))
                                for w in self.eeg_windows]
        # handle labels
        for i_window, window_labels in enumerate(self.label_windows):
            if self.discretize_labels:
                self.label_windows[i_window] = [1 if label >= 5 else 0 for label in window_labels]
            else:
                self.label_windows[i_window] /= 9
        # converts data to tensors
        self.eeg_windows = torch.stack([torch.as_tensor(w).float() for w in self.eeg_windows])
        if self.normalize_eegs:
            # self.eeg_windows -= self.eeg_windows.amin(dim=[0, 1]).repeat(*self.eeg_windows.shape[:2], 1)
            # self.eeg_windows /= self.eeg_windows.amax(dim=[0, 1]).repeat(*self.eeg_windows.shape[:2], 1)
            self.eeg_windows = self.eeg_windows / self.eeg_windows.amax()
            # self.eeg_windows = (self.eeg_windows - self.eeg_windows.mean()) / self.eeg_windows.std()
        self.label_windows = torch.stack([torch.as_tensor(w).long() for w in self.label_windows])

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            if self.validation == "k_fold":
                train_indices = [i
                                 for i_fold, f in enumerate(self.folds_indices)
                                 for i in f
                                 if i_fold != self.current_k_fold_index]
                test_indices = [i
                                for i_fold, f in enumerate(self.folds_indices)
                                for i in f
                                if i_fold == self.current_k_fold_index]
            elif self.validation == "loso":
                train_indices = [
                    i for i in range(len(self))
                    if self.subject_ids_windows[i] != self.subjects_ids_indices[self.current_loso_index]
                ]
                test_indices = [
                    i for i in range(len(self))
                    if self.subject_ids_windows[i] == self.subjects_ids_indices[self.current_loso_index]
                ]
            self.train_split, self.val_split = Subset(self, train_indices), \
                                               Subset(self, test_indices)

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True,
                          num_workers=os.cpu_count() - 2)

    def val_dataloader(self):
        return DataLoader(self.val_split, batch_size=self.batch_size, shuffle=False,
                          num_workers=os.cpu_count() - 2)

    def set_k_fold(self, i: int) -> None:
        assert isinstance(i, int) and 0 <= i < self.k_folds
        self.current_k_fold_index = i

    def set_loso_index(self, i: int) -> None:
        assert isinstance(i, int) and i in self.subjects_ids_indices.keys()
        self.current_loso_index = i

    def get_subject_ids(self) -> List[str]:
        subject_ids = [basename(splitext(s)[0])
                       for s in os.listdir(join(self.path, "data_preprocessed_python"))]
        subject_ids.sort()
        return subject_ids

    @staticmethod
    def get_subject_ids_static(path: str) -> List[str]:
        assert isdir(path)
        subject_ids = [basename(splitext(s)[0])
                       for s in os.listdir(join(path, "data_preprocessed_python"))]
        subject_ids.sort()
        return subject_ids
