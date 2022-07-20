from os import makedirs
from os.path import isdir
from typing import Optional

from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.distributed import rank_zero_only
import pandas as pd


class MyLogger(LightningLoggerBase):
    def __init__(self, path: Optional[str] = None):
        super().__init__()
        if path is not None:
            if not isdir(path):
                makedirs(path)
        self.logs: pd.DataFrame = pd.DataFrame()

    @property
    def name(self):
        return "MyLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        self.logs = self.logs.append(metrics, ignore_index=True)

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        print(self.logs)
