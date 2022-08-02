import gc
from abc import abstractmethod, ABC
from copy import deepcopy
from math import ceil
from os.path import isdir, join, exists
from pprint import pprint
from typing import Dict, Optional, Union, List, Tuple, Any

import mne
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

import numpy as np
import einops
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class EEGClassificationDatasetCached(Dataset, ABC):

    def __init__(self, path: str,
                 sampling_rate: int,
                 electrodes: Union[int, List[str]],
                 labels: List[str],
                 subject_ids: List[str],

                 split_in_windows: bool = False,
                 window_size: Optional[Union[float, int]] = 1,
                 window_stride: Optional[Union[float, int]] = None,
                 drop_last: Optional[bool] = True,

                 # labels_to_use: Optional[Union[str, List[str]]] = None,
                 # subject_ids_to_use: Optional[Union[str, List[str]]] = None,

                 discretize_labels: bool = False,
                 normalize_eegs: bool = True,

                 # validation: Optional[str] = None,
                 # k_folds: Optional[int] = 10,
                 # batch_size: int = 32
                 ):
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
        self.labels: List[str] = labels

        assert isinstance(subject_ids, list)
        assert all((isinstance(x, str) for x in subject_ids))
        self.subject_ids: List[str] = subject_ids
        self.subject_ids.sort()

        assert isinstance(split_in_windows, bool)
        self.split_in_windows: bool = split_in_windows

        if self.split_in_windows is True:
            assert window_size > 0
            self.window_size: float = float(window_size)  # s
            self.samples_per_window: int = int(np.floor(self.sampling_rate * self.window_size))
            assert window_stride is None or window_stride > 0
            if window_stride is None:
                window_stride = deepcopy(self.window_size)
            self.window_stride: float = float(window_stride)  # s
            self.samples_per_stride: int = int(np.floor(self.sampling_rate * self.window_stride))

            assert isinstance(drop_last, bool)
            self.drop_last: bool = drop_last

        # assert labels_to_use is None or isinstance(labels_to_use, str) or isinstance(labels_to_use, list)
        # if labels_to_use is None:
        #     labels_to_use = list(self.labels.keys())
        # elif isinstance(labels_to_use, str):
        #     labels_to_use = [labels_to_use]
        # assert set(labels_to_use).issubset(set(self.labels.keys())), \
        #     f"one or more labels are not allowed"
        # self.labels_to_use: List[str] = labels_to_use

        # assert subject_ids_to_use is None or isinstance(subject_ids_to_use, str) or isinstance(subject_ids_to_use, list)
        # if subject_ids_to_use is None:
        #     subject_ids_to_use = deepcopy(self.subject_ids)
        # elif isinstance(subject_ids_to_use, str):
        #     subject_ids_to_use = [subject_ids_to_use]
        # assert set(subject_ids_to_use).issubset(set(self.subject_ids)), \
        #     f"one or more subject ids are not in dataset"
        # self.subject_ids_to_use: List[str] = subject_ids_to_use
        # self.subject_ids_to_use.sort()

        assert isinstance(discretize_labels, bool)
        self.discretize_labels: bool = discretize_labels
        assert isinstance(normalize_eegs, bool)
        self.normalize_eegs: bool = normalize_eegs

        self.lookup: List[Dict[str, Any]] = self.preprocess_files()
        if self.normalize_eegs:
            self.means, self.stds = self.get_scaling_parameters()
        self.windows: List[Dict[str, Any]] = self.get_windows()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> Dict[str, Union[int, torch.Tensor]]:
        window = self.windows[i]
        eegs = np.load(join(self.path, "_cached_files", window["subject_id"],
                            window["filename"]))[window["window_start"]:window["window_end"]]
        if self.normalize_eegs:
            eegs = self.get_normalized_eegs(eegs)
        labels = window["labels"]
        labels = self.get_processed_labels(labels)
        return {
            "sampling_rates": self.sampling_rate,
            "subject_id": window["subject_id"],
            "labels": labels,
            "eegs": eegs
        }

    def prepare_data(self) -> None:
        pass

    @staticmethod
    @abstractmethod
    def get_subject_ids_static(path: str):
        pass

    @abstractmethod
    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        pass

    @abstractmethod
    def preprocess_files(self):
        pass

    @abstractmethod
    def get_processed_labels(self, labels: List[int]) -> List[int]:
        pass

    def get_windows(self) -> List[Dict[str, Any]]:
        windows = []
        for i_experiment in range(len(self.lookup)):
            for i_window_start in range(0,
                                        self.lookup[i_experiment]["duration"],
                                        self.samples_per_stride):
                # eventually skips uneven windows
                if (i_window_start + self.samples_per_window > self.lookup[i_experiment]["duration"]) \
                        and self.drop_last is True:
                    continue
                windows += [{
                    **self.lookup[i_experiment],
                    "window_start": i_window_start,
                    "window_end": i_window_start + self.samples_per_window,
                }]
        return windows
        # window = self.eegs_data[i_experiment][i_window_start:i_window_start + self.samples_per_window, :]
        # if len(window) == self.samples_per_window or self.drop_last is False:
        #     eegs_data_windowed += [window]
        #     labels_data_windowed += [self.labels_data[i_experiment]]
        #     subject_ids_data_windowed += [self.subject_ids_data[i_experiment]]

    def get_scaling_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        means_path: str = join(self.path, "_cached_files", "means.npy")
        stds_path: str = join(self.path, "_cached_files", "stds.npy")
        # eventually recomputes the means
        if not (exists(means_path) and exists(stds_path)):
            means_tmp, stds_tmp = [], []
            # loops through electrodes
            for i_electrode, electrode in tqdm(enumerate(self.electrodes),
                                               total=len(self.electrodes),
                                               desc="obtaining means and stds for normalization"):
                x = np.concatenate(
                    [np.load(join(self.path, "_cached_files", record["subject_id"], record["filename"]))[:, i_electrode]
                     for record in self.lookup])
                means_tmp += [x.mean()]
                stds_tmp += [x.std()]
                del x
            # saves the parameters for each electrode
            np.save(join(self.path, "_cached_files", "means.npy"), np.asarray(means_tmp))
            np.save(join(self.path, "_cached_files", "stds.npy"), np.asarray(stds_tmp))
        # loads means and stds
        means: np.ndarray = np.load(means_path)
        stds: np.ndarray = np.load(stds_path)
        return means, stds

    def get_normalized_eegs(self, eegs: np.ndarray) -> np.ndarray:
        # scales to zero mean and unit variance
        eegs = (eegs - self.means) / self.stds
        # casts to microvolts
        eegs *= 1e-6
        return eegs

    def plot_amplitudes_distribution(
            self,
            title: str = "distribution of amplitudes",
            scale: Union[int, float] = 4,
    ):
        cols: int = 8
        rows: int = ceil(len(self.electrodes) / cols)
        fig, axs = plt.subplots(nrows=rows, ncols=cols,
                                figsize=(scale * cols, scale * rows),
                                tight_layout=True)
        fig.suptitle(title)
        for i_electrode, ax in enumerate(axs.flat):
            if i_electrode >= len(self.electrodes):
                ax.set_visible(False)
                continue
            ax.hist(
                np.concatenate([x["eegs"][:, i_electrode] for x in self]),
                bins=32
            )
            ax.set_title(self.electrodes[i_electrode])
            ax.set_xlabel("mV")
            ax.set_ylabel("count")
            ax.set_yscale("log")
        # adjusts the ylim
        max_ylim = max([ax.get_ylim()[-1] for ax in axs.flat])
        for ax in axs.flat:
            ax.set_ylim([None, max_ylim])
        plt.show()
        fig.clf()
