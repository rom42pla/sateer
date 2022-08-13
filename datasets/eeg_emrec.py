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
                 labels_classes: Union[int, List[int]] = 2,

                 window_size: Optional[Union[float, int]] = 1,
                 window_stride: Optional[Union[float, int]] = None,
                 drop_last: Optional[bool] = False,

                 discretize_labels: bool = False,
                 normalize_eegs: bool = True,
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
        assert isinstance(labels_classes, int) or isinstance(labels_classes, list), \
            f"the labels classes must be a list of integers or a positive integer, not {labels_classes}"
        if isinstance(labels_classes, list):
            assert all([isinstance(labels_class, int) for labels_class in labels_classes]), \
                f"if the name of the labels are given ({labels_classes}), they must all be strings"
            assert len(labels_classes) == len(labels)
            self.labels_classes = labels_classes
        else:
            assert labels_classes > 0, \
                f"there must be a positive number of classes, not {labels_classes}"
            self.labels_classes = [labels_classes for _ in self.labels]

        assert isinstance(subject_ids, list)
        assert all((isinstance(x, str) for x in subject_ids))
        self.subject_ids: List[str] = subject_ids
        self.subject_ids.sort()

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

        assert isinstance(discretize_labels, bool)
        self.discretize_labels: bool = discretize_labels
        assert isinstance(normalize_eegs, bool)
        self.normalize_eegs: bool = normalize_eegs

        self.eegs_data, self.labels_data, self.subject_ids_data = self.load_data()
        if self.normalize_eegs:
            self.eegs_data = self.normalize(self.eegs_data)
        self.windows = self.get_windows()
        assert len(self.eegs_data) == len(self.labels_data) == len(self.subject_ids_data)
        assert all([e.shape[-1] == len(self.electrodes) for e in self.eegs_data])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> Dict[str, Union[int, str, np.ndarray]]:
        window = self.windows[i]
        eegs = self.eegs_data[window["experiment"]][window["start"]:window["end"]]
        # eventually pad the eegs
        if eegs.shape[0] != self.samples_per_window:
            eegs = np.concatenate([eegs,
                                   np.zeros([self.samples_per_window - eegs.shape[0], eegs.shape[1]])],
                                  axis=0)
        assert eegs.shape[0] == self.samples_per_window
        return {
            "sampling_rates": self.sampling_rate,
            "subject_id": window["subject_id"],
            "eegs": eegs.astype(np.float32),
            "labels": window["labels"]
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

    def normalize(self, eegs: List[np.ndarray]):
        # loops through the experiments
        for i_experiment, experiment in enumerate(eegs):
            # scales the experiment to zero mean and unit variance
            experiment_scaled = (experiment - experiment.mean(axis=0)) / experiment.std(axis=0)
            experiment_scaled = np.nan_to_num(experiment_scaled)
            # scaler = mne.decoding.Scaler(info=mne.create_info(ch_names=self.electrodes, sfreq=self.sampling_rate,
            #                                                   verbose=False, ch_types="eeg"),
            #                              scalings="median")
            # scaler.fit(einops.rearrange(experiment, "s c -> () c s"))
            # experiment_scaled = einops.rearrange(
            #     scaler.transform(einops.rearrange(experiment,
            #                                       "s c -> () c s")),
            #     "b c s -> s (b c)")
            # scales to microvolts
            experiment_scaled *= 1e-6
            eegs[i_experiment] = experiment_scaled
        return eegs

    def get_windows(self) -> List[Dict[str, Union[int, str]]]:
        windows: List[Dict[str, Union[int, str]]] = []
        for i_experiment in range(len(self.eegs_data)):
            for i_window_start in range(0,
                                        len(self.eegs_data[i_experiment]),
                                        self.samples_per_stride):
                window = {
                    "experiment": i_experiment,
                    "start": i_window_start,
                    "end": i_window_start + self.samples_per_window,
                    "subject_id": self.subject_ids_data[i_experiment],
                    "labels": np.asarray(self.labels_data[i_experiment]),
                }
                if self.drop_last is True and (window["end"] - window["start"]) != self.samples_per_window:
                    continue
                windows += [window]
        return windows

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
