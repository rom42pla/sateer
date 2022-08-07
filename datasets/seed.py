import gc
import json
import re
import uuid
from multiprocessing import Pool
from os.path import isdir, join, splitext, basename, exists
from pprint import pprint
from typing import Dict, List, Tuple, Any, Union

import os

import numpy as np
import scipy.io as sio
import pickle
import einops
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.eeg_emrec_cached import EEGClassificationDatasetCached


class SEEDDataset(EEGClassificationDatasetCached):
    def __init__(self, path: str, **kwargs):
        super(SEEDDataset, self).__init__(path=path,
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
                                          **kwargs)

    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        pass

    def preprocess_files(self) -> List[Dict[str, Union[bool, int, str, List[Union[int, float]]]]]:
        lookup: List[Dict[str, Union[bool, int, str, List[Union[int, float]]]]] = []
        labels_common: np.ndarray = sio.loadmat(join(self.path, "Preprocessed_EEG", "label.mat"),
                                                simplify_cells=True)["label"]
        # loops through subjects
        for subject_id in tqdm(self.subject_ids, desc=f"preprocessing SEED dataset files"):
            # eventually creates the directory
            if not isdir(join(self.path, "_cached_files", subject_id)):
                os.makedirs(join(self.path, "_cached_files", subject_id))
            subject_files = {s
                             for s in os.listdir(join(self.path, "Preprocessed_EEG"))
                             if re.fullmatch(f"{subject_id}_.*", s)}
            for subject_file in subject_files:
                # checks if the lookup json exists and it's valide
                if exists(join(self.path, "_cached_files", subject_id, "lookup.json")):
                    # reads the lookup
                    with open(join(self.path, "_cached_files", subject_id, "lookup.json"), "r") as fp:
                        lookup_subject = json.load(fp)
                    # checks that all files exists
                    if all([exists(join(self.path, "_cached_files", subject_id, record["filename"]))
                            for record in lookup_subject]):
                        lookup += lookup_subject
                        continue
                    # removes the previous, corrupted directory
                    os.removedirs(join(self.path, "_cached_files", subject_id))
                    os.makedirs(join(self.path, "_cached_files", subject_id))
                lookup_subject: List[Dict[str, str]] = []
                subject_data = sio.loadmat(join(self.path, "Preprocessed_EEG", subject_file),
                                           simplify_cells=True)
                for key in {k for k in subject_data.keys() if re.fullmatch(r".+_eeg[0-9]+", k)}:
                    trial_number = int(re.search(r"[0-9]+", key.split("_")[-1])[0])
                    filename = f"{uuid.uuid4().hex[:16]}.npy"
                    eegs = einops.rearrange(subject_data[key], "c s -> s c")
                    label = labels_common[trial_number - 1] + 1
                    # if self.discretize_labels \
                    # else ((labels_common[trial_number - 1] + 1) / 2)
                    np.save(join(self.path, "_cached_files", subject_id, filename), eegs)
                    lookup_subject += [{
                        "filename": filename,
                        "labels": [int(label)],
                        # "discretized_labels": self.discretize_labels,
                        "subject_id": subject_id,
                        "duration": len(eegs)
                    }]
                with open(join(self.path, "_cached_files", subject_id, "lookup.json"), "w") as fp:
                    json.dump(lookup_subject, fp, indent=4)
                lookup += lookup_subject
        return lookup

    def get_processed_labels(self, labels: List[int]) -> List[int]:
        if self.discretize_labels:
            labels = [l + 1 for l in labels]
        else:
            labels = [(l + 1) / 2 for l in labels]
        return labels

    @staticmethod
    def get_subject_ids_static(path: str) -> List[str]:
        assert isdir(path)
        subject_ids = list({basename(splitext(s)[0]).split("_")[0]
                            for s in os.listdir(join(path, "Preprocessed_EEG"))
                            if re.fullmatch(r"[0-9]+_.*", s)})
        subject_ids.sort()
        return subject_ids


if __name__ == "__main__":
    dataset_path = join("..", "..", "..", "datasets", "eeg_emotion_recognition", "seed")
    dataset = SEEDDataset(path=dataset_path,
                          discretize_labels=True, normalize_eegs=True, split_in_windows=True,
                          window_size=10, window_stride=0.5)
    print("loaded", len(dataset))
    dl = DataLoader(dataset, batch_size=1024, num_workers=2)
    for b in tqdm(dl, total=len(dl)):
        pass
    # dataset.plot_labels_distribution()
    # dataset.plot_amplitudes_distribution()
