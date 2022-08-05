import gc
from abc import abstractmethod, ABC
from copy import deepcopy
from math import ceil
from os.path import isdir
from pprint import pprint
from typing import Dict, Optional, Union, List, Tuple

import mne
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

import numpy as np
import einops
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class EEGClassificationDataset(Dataset, ABC):

    def __init__(self, path: str,
                 sampling_rate: int,
                 electrodes: Union[int, List[str]],
                 labels: List[str],
                 subject_ids: List[str],

                 split_in_windows: bool = False,
                 window_size: Optional[Union[float, int]] = 1,
                 window_stride: Optional[Union[float, int]] = None,
                 drop_last: Optional[bool] = False,

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

        self.eegs_data, self.labels_data, self.subject_ids_data = self.load_data()
        assert len(self.eegs_data) == len(self.labels_data) == len(self.subject_ids_data)
        assert all([e.shape[-1] == len(self.electrodes) for e in self.eegs_data])
        self.setup_data()

    def __len__(self) -> int:
        return len(self.eegs_data)

    def __getitem__(self, i: int) -> Dict[str, Union[int, str, torch.Tensor]]:
        return {
            "sampling_rates": self.sampling_rate,
            # "subject_id": self.subject_ids.index(self.subject_ids_data[i]),
            "subject_id": self.subject_ids_data[i],
            "eegs": self.eegs_data[i],
            "labels": self.labels_data[i],
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
                                            self.samples_per_stride):
                    window = self.eegs_data[i_experiment][i_window_start:i_window_start + self.samples_per_window, :]
                    if len(window) == self.samples_per_window or self.drop_last is False:
                        eegs_data_windowed += [window]
                        labels_data_windowed += [self.labels_data[i_experiment]]
                        subject_ids_data_windowed += [self.subject_ids_data[i_experiment]]
            self.eegs_data = eegs_data_windowed
            self.labels_data = labels_data_windowed
            self.subject_ids_data = subject_ids_data_windowed
            assert len(self.eegs_data) == len(self.labels_data) == len(self.subject_ids_data)
        # eventually pads uneven windows
        windows_size = max([len(w) for w in self.eegs_data])
        self.eegs_data = [w if w.shape[0] == windows_size
                          else np.vstack((w, np.zeros((windows_size - w.shape[0], w.shape[1]))))
                          for w in self.eegs_data]
        assert all([len(w) == windows_size for w in self.eegs_data])
        # converts to tensor
        self.eegs_data: np.ndarray = np.stack(self.eegs_data).astype(np.float32)
        self.labels_data: np.ndarray = np.stack(self.labels_data).astype(np.long)

    def plot_samples(self) -> None:
        raw_mne_array = mne.io.RawArray(einops.rearrange(self[0][0], "s c -> c s"),
                                        info=mne.create_info(ch_names=self.electrodes, sfreq=self.sampling_rate,
                                                             verbose=False, ch_types="eeg"), verbose=False)
        raw_mne_array.plot()

    def plot_subjects_distribution(self) -> None:
        subject_indices_samples = sorted([s["subject_id"]
                                          for s in self])
        subject_ids_samples = [self.subject_ids[i]
                               for i in subject_indices_samples]
        fig, ax = plt.subplots(1, 1,
                               figsize=(15, 5),
                               tight_layout=True)
        sns.countplot(x=subject_ids_samples,
                      palette="rocket", ax=ax)
        plt.show()
        fig.clf()

    def plot_labels_distribution(
            self,
            title: str = "distribution of labels",
            scale: Union[int, float] = 4,
    ) -> None:
        if self.discretize_labels:
            cols = min(8, len(self.labels))
            rows = 1 if (len(self.labels) <= 8) else ceil(len(self.labels) / 8)
            fig, axs = plt.subplots(nrows=rows, ncols=cols,
                                    figsize=(scale * cols, scale * rows),
                                    tight_layout=True)
            fig.suptitle(title)
            labels_data = np.stack([x["labels"] for x in self])
            for i_ax, ax in enumerate(axs.flat):
                if i_ax >= len(self.labels):
                    ax.set_visible(False)
                    continue
                unique_labels = np.unique(labels_data[:, i_ax])
                sizes = [np.count_nonzero(labels_data[:, i_ax] == unique_label)
                         for unique_label in unique_labels]
                axs.flat[i_ax].pie(sizes, labels=unique_labels, autopct='%1.1f%%',
                                   shadow=False)
                axs.flat[i_ax].axis('equal')
                axs.flat[i_ax].set_title(self.labels[i_ax])
        else:
            # builds the dataframe
            df = pd.DataFrame()
            for x in self:
                for i_label, label in enumerate(self.labels):
                    df = pd.concat([df, pd.DataFrame([{
                        "label": label,
                        "value": x["labels"][i_label]
                    }])], ignore_index=True).sort_values(by="value", ascending=True)
            fig, axs = plt.subplots(nrows=1, ncols=len(self.labels),
                                    figsize=(scale * len(self.labels), scale),
                                    tight_layout=True)
            fig.suptitle(title)
            # plots
            for i_label, label in enumerate(self.labels):
                ax = axs[i_label]
                sns.histplot(
                    data=df[df["label"] == label],
                    bins=16,
                    palette="rocket",
                    ax=ax
                )
                ax.set_xlabel(label)
                ax.set_ylabel("count")
                ax.get_legend().remove()
            # adjusts the ylim
            max_ylim = max([ax.get_ylim()[-1] for ax in axs])
            for ax in axs:
                ax.set_ylim([0, max_ylim])
        plt.show()
        fig.clf()

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
