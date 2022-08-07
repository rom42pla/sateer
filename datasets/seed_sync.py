import os
import re
import uuid
from multiprocessing import Pool
from os.path import isdir, join, basename, splitext
from typing import Dict, List, Any, Tuple

import einops
import psutil
import scipy.io as sio
import numpy as np
from tqdm import tqdm

from datasets.eeg_emrec import EEGClassificationDataset


class SEEDDataset(EEGClassificationDataset):
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
        # loads DREAMER.mat
        # data_raw = sio.loadmat(join(self.path, "DREAMER.mat"), simplify_cells=True)["DREAMER"]["Data"]
        #
        # global parse_eegs
        #
        # def parse_eegs(subject_no) -> Tuple[List[np.ndarray], List[np.ndarray], str]:
        #     subject_id: str = self.subject_ids[subject_no]
        #     assert subject_id in self.subject_ids
        #     subject_data = data_raw[subject_no]
        #     eegs: List[np.ndarray] = []
        #     labels: List[np.ndarray] = []
        #     experiments_no = len(subject_data["EEG"]["stimuli"])
        #     assert experiments_no \
        #            == len(subject_data["EEG"]["stimuli"]) == len(subject_data["ScoreArousal"]) \
        #            == len(subject_data["ScoreValence"]) == len(subject_data["ScoreDominance"])
        #     for i_experiment in range(experiments_no):
        #         # loads the eeg for the experiment
        #         eegs += [subject_data["EEG"]["stimuli"][i_experiment]]  # (s c)
        #         # loads the labels for the experiment
        #         labels += [np.asarray([subject_data[k][i_experiment]
        #                                for k in ["ScoreArousal", "ScoreValence",
        #                                          "ScoreDominance"]])]  # (l)
        #     # eventually discretizes the labels
        #     labels = [[1 if label > 3 else 0 for label in w] if self.discretize_labels else (w - 1) / 4
        #               for w in labels]
        #     return eegs, labels, subject_id
        #
        # with Pool(processes=len(self.subject_ids)) as pool:
        #     data_pool = pool.map(parse_eegs, [i for i in range(len(self.subject_ids))])
        #     data_pool = [d for d in data_pool if d is not None]
        #     eegs: List[np.ndarray] = [e for eeg_lists, _, _ in data_pool
        #                               for e in eeg_lists]
        #     labels: List[np.ndarray] = [l for _, labels_lists, _ in data_pool
        #                                 for l in labels_lists]
        #     subject_ids: List[str] = [s_id for eegs_lists, _, subject_id in data_pool
        #                               for s_id in [subject_id] * len(eegs_lists)]
        # assert len(eegs) == len(labels) == len(subject_ids)
        # return eegs, labels, subject_ids

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
    # dataset.plot_labels_distribution()
    # dataset.plot_amplitudes_distribution()
