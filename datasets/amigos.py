from multiprocessing import Pool
from os.path import isdir, join, splitext, basename
from typing import Dict, List, Tuple

import os

import numpy as np
import scipy.io as sio
import pickle
import einops

from datasets.eeg_emrec import EEGClassificationDataset


class AMIGOSDataset(EEGClassificationDataset):
    def __init__(self, path: str, **kwargs):
        super(AMIGOSDataset, self).__init__(path=path,
                                            sampling_rate=128,
                                            electrodes=["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
                                                        "P8", "T8", "FC6", "F4", "F8", "AF4"],
                                            labels=["arousal", "valence", "dominance", "liking",
                                                    "familiarity", "neutral", "disgust", "happiness",
                                                    "surprise", "anger", "fear", "sadness"],
                                            subject_ids=AMIGOSDataset.get_subject_ids_static(path=path),
                                            **kwargs)

    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        global parse_eegs

        def parse_eegs(subject_id: str) -> Tuple[List[np.ndarray], List[np.ndarray], str]:
            subject_data = sio.loadmat(join(self.path, "data_preprocessed", f"Data_Preprocessed_{subject_id}.mat"),
                                       simplify_cells=True)
            # some data is corrupted
            valid_indices = {i
                             for i, (eegs, labels) in enumerate(zip(subject_data["joined_data"],
                                                                    subject_data["labels_selfassessment"]))
                             if eegs.any() and labels.any()}
            eegs: List[np.ndarray] = [e[:, :14].astype(np.float32) for i, e in enumerate(subject_data["joined_data"])
                                      if i in valid_indices]
            labels: List[np.ndarray] = [e.astype(int) for i, e in enumerate(subject_data["labels_selfassessment"])
                                        if i in valid_indices]
            assert len(eegs) == len(labels)
            for i_trial, labels_trial in enumerate(labels):
                if self.discretize_labels:
                    labels[i_trial][:5] = np.asarray([1 if label > 5 else 0 for label in labels_trial[:5]])
                else:
                    labels_trial[labels_trial > 9] = 9
                    labels[i_trial][:5] = (labels_trial[:5] - 1) / 8
                    assert labels[i_trial][:5].min() >= 1 and labels[i_trial][:5].max() <= 9
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
        subject_ids = [basename(splitext(s)[0]).split("_")[-1]
                       for s in os.listdir(join(path, "data_preprocessed"))]
        subject_ids.sort()
        return subject_ids


if __name__ == "__main__":
    dataset = AMIGOSDataset(path=join("..", "..", "..", "datasets", "eeg_emotion_recognition", "amigos"),
                            discretize_labels=True, normalize_eegs=True, split_in_windows=True)
    print("loaded", len(dataset))
    dataset.plot_labels_distribution()
    dataset.plot_amplitudes_distribution()
