import os
import re
from os.path import isdir, join, basename, splitext
from typing import List, Tuple

import einops
import scipy.io as sio
import numpy as np
from tqdm import tqdm

from datasets.eeg_emrec import EEGClassificationDataset


class SEEDDataset(EEGClassificationDataset):
    def __init__(self, path: str, **kwargs):
        super(SEEDDataset, self).__init__(
            name="SEED",
            path=path,
            sampling_rate=200,
            electrodes=['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3',
                        'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
                        'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7',
                        'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
                        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
                        'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4',
                        'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6',
                        'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'],
            labels=["emotion"],
            labels_classes=3,
            subject_ids=SEEDDataset.get_subject_ids_static(path=path),
            **kwargs
        )

    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        labels_common: np.ndarray = sio.loadmat(join(self.path, "Preprocessed_EEG", "label.mat"),
                                                simplify_cells=True)["label"]
        eegs: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        subject_ids: List[str] = []
        # loops through subjects
        for subject_id in tqdm(self.subject_ids, desc=f"preprocessing SEED dataset files"):
            subject_files = {s
                             for s in os.listdir(join(self.path, "Preprocessed_EEG"))
                             if re.fullmatch(f"{subject_id}_.*", s)}
            for subject_file in subject_files:
                subject_data = sio.loadmat(join(self.path, "Preprocessed_EEG", subject_file),
                                           simplify_cells=True)
                for key in {k for k in subject_data.keys() if re.fullmatch(r".+_eeg[0-9]+", k)}:
                    trial_number = int(re.search(r"[0-9]+", key.split("_")[-1])[0])
                    eegs += [einops.rearrange(subject_data[key], "c s -> s c")]
                    labels += [np.asarray([labels_common[trial_number - 1] + 1])]
                    subject_ids += [subject_id]
        return eegs, labels, subject_ids

    @staticmethod
    def get_subject_ids_static(path: str) -> List[str]:
        assert isdir(path)
        subject_ids = list({basename(splitext(s)[0]).split("_")[0]
                            for s in os.listdir(join(path, "Preprocessed_EEG"))
                            if re.fullmatch(r"[0-9]+_.*", s)})
        subject_ids.sort()
        return subject_ids


if __name__ == "__main__":
    dataset = SEEDDataset(path=join("..", "..", "..", "datasets", "eeg_emotion_recognition", "seed"),
                          discretize_labels=True, normalize_eegs=True)
