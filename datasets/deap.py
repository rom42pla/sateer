from abc import ABC
from copy import deepcopy
from multiprocessing import Pool
from os.path import isdir, join, splitext, basename
from pprint import pprint
from typing import Dict, Optional, Union, List, Set, Any, Tuple

import mne.io
import torch
from torch.utils.data import Dataset, DataLoader, Subset

import os

import scipy.io as sio
import numpy as np
import pickle
import einops
import pytorch_lightning as pl

from datasets.eeg_emrec import EEGClassificationDataset


class DEAPDataset(EEGClassificationDataset):
    def __init__(self, path: str, **kwargs):
        super(DEAPDataset, self).__init__(path=path,
                                          sampling_rate=128,
                                          electrodes=["Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5",
                                                      "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz", "Fp2", "AF4",
                                                      "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6",
                                                      "CP2", "P4", "P8", "PO4", "O2"],
                                          labels=["valence", "arousal", "dominance", "liking"],
                                          subject_ids=DEAPDataset.get_subject_ids_static(path=path),
                                          **kwargs)

    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        global parse_eegs

        def parse_eegs(subject_id: str) -> Tuple[List[np.ndarray], List[np.ndarray], str]:
            with open(join(self.path, "data_preprocessed_python", f"{subject_id}.dat"), "rb") as fp:
                subject_data: Dict[str, np.ndarray] = pickle.load(fp, encoding='latin1')
            subject_data["data"] = einops.rearrange(subject_data["data"],
                                                    "b c s -> b s c")[:, :self.sampling_rate * 60, :32]
            eegs: List[np.ndarray] = []
            labels: List[np.ndarray] = []
            experiments_no = len(subject_data["data"])
            assert experiments_no \
                   == len(subject_data["data"]) == len(subject_data["labels"])
            for i_experiment in range(experiments_no):
                # loads the eeg for the experiment
                eegs += [subject_data["data"][i_experiment]]  # (s c)
                # loads the labels for the experiment
                labels += [subject_data["labels"][i_experiment]]  # (l)
            # eventually discretizes the labels
            labels = [[1 if label > 5 else 0 for label in w] if self.discretize_labels else (w - 1) / 8
                      for w in labels]
            return eegs, labels, subject_id

        with Pool(processes=len(self.subject_ids)) as pool:
            data_pool = pool.map(parse_eegs, [s_id for s_id in self.subject_ids])
            data_pool = [d for d in data_pool if d is not None]
            eegs: List[np.ndarray] = [e for eeg_lists, _, _ in data_pool
                                      for e in eeg_lists]
            labels: List[np.ndarray] = [l for _, labels_lists, _ in data_pool
                                        for l in labels_lists]
            subject_ids: List[str] = [s_id for eegs_lists, _, subject_id in data_pool
                                      for s_id in [subject_id] * len(eegs_lists)]
        assert len(eegs) == len(labels) == len(subject_ids)
        return eegs, labels, subject_ids

    @staticmethod
    def get_subject_ids_static(path: str) -> List[str]:
        assert isdir(path)
        subject_ids = [basename(splitext(s)[0])
                       for s in os.listdir(join(path, "data_preprocessed_python"))]
        subject_ids.sort()
        return subject_ids


if __name__ == "__main__":
    dataset = DEAPDataset(path=join("..", "..", "..", "datasets", "eeg_emotion_recognition", "deap"),
                          discretize_labels=True)
    dataset.plot_labels_distribution()
