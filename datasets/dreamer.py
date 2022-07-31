from multiprocessing import Pool
from os.path import isdir, join
from typing import Dict, List, Any, Tuple

import scipy.io as sio
import numpy as np

from datasets.eeg_emrec import EEGClassificationDataset


class DREAMERDataset(EEGClassificationDataset):
    def __init__(self, path: str, **kwargs):
        super(DREAMERDataset, self).__init__(path=path,
                                             sampling_rate=128,
                                             electrodes=['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
                                                         'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'],
                                             labels=["valence", "arousal", "dominance"],
                                             subject_ids=DREAMERDataset.get_subject_ids_static(path=path),
                                             **kwargs)

    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        # loads DREAMER.mat
        data_raw = sio.loadmat(join(self.path, "DREAMER.mat"), simplify_cells=True)["DREAMER"]["Data"]

        global parse_eegs

        def parse_eegs(subject_no) -> Tuple[List[np.ndarray], List[np.ndarray], str]:
            subject_id: str = self.subject_ids[subject_no]
            assert subject_id in self.subject_ids
            subject_data = data_raw[subject_no]
            eegs: List[np.ndarray] = []
            labels: List[np.ndarray] = []
            experiments_no = len(subject_data["EEG"]["stimuli"])
            assert experiments_no \
                   == len(subject_data["EEG"]["stimuli"]) == len(subject_data["ScoreArousal"]) \
                   == len(subject_data["ScoreValence"]) == len(subject_data["ScoreDominance"])
            for i_experiment in range(experiments_no):
                # loads the eeg for the experiment
                eegs += [subject_data["EEG"]["stimuli"][i_experiment]]  # (s c)
                # loads the labels for the experiment
                labels += [np.asarray([subject_data[k][i_experiment]
                                       for k in ["ScoreArousal", "ScoreValence",
                                                 "ScoreDominance"]])]  # (l)
            # eventually discretizes the labels
            labels = [[1 if label > 3 else 0 for label in w] if self.discretize_labels else (w - 1) / 4
                      for w in labels]
            return eegs, labels, subject_id

        with Pool(processes=len(self.subject_ids)) as pool:
            data_pool = pool.map(parse_eegs, [i for i in range(len(self.subject_ids))])
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
        data: Dict[str, Any] = sio.loadmat(join(path, "DREAMER.mat"), simplify_cells=True)["DREAMER"]["Data"]
        subject_ids: List[str] = [f"s{i}" for i in range(len(data))]
        subject_ids.sort()
        return subject_ids


if __name__ == "__main__":
    dataset = DREAMERDataset(path=join("..", "..", "..", "datasets", "eeg_emotion_recognition", "dreamer"),
                             discretize_labels=True, normalize_eegs=True, split_in_windows=True)
    dataset.plot_labels_distribution()
    dataset.plot_amplitudes_distribution()
