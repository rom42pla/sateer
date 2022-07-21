from os import makedirs
from os.path import isdir, join, dirname
from typing import Optional, Union, List

from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.distributed import rank_zero_only

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class FouriEEGTransformerLogger(LightningLoggerBase):
    def __init__(self, path: Optional[str] = None, plot: bool = True):
        super().__init__()
        assert path is None or isinstance(path, str)
        self.path = path
        if self.path is not None:
            if not isdir(self.path):
                makedirs(self.path)
        self.logs: pd.DataFrame = pd.DataFrame()
        assert isinstance(plot, bool)
        self.plot = plot

    @property
    def name(self):
        return "MyLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    def experiment(self):
        pass

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        self.logs = pd.concat([self.logs, pd.DataFrame([metrics])],
                              ignore_index=True)

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status):
        print(self.logs)
        self.make_plot(key=f"loss", legend_name=f"loss",
                       y_lims=[0, None],
                       plot=self.plot, path=join("plots"))
        self.make_plot(key=f"acc_mean", legend_name=f"accuracy (mean)",
                       y_lims=[0.4, 1],
                       plot=self.plot, path=join("plots"))
        for label in ["valence", "arousal", "dominance"]:
            self.make_plot(key=f"acc_{label}", legend_name=f"accuracy ({label})",
                           y_lims=[0.4, 1],
                           plot=self.plot, path=join("plots"))

    @rank_zero_only
    def make_plot(self, key: str,
                  legend_name: Optional[str] = None,
                  title: Optional[str] = None,
                  x_label: Optional[str] = None,
                  y_label: Optional[str] = None,
                  x_lims: Optional[List[Union[int, float]]] = None,
                  y_lims: Optional[List[Union[int, float]]] = None,
                  plot: bool = True,
                  path: Optional[str] = None):
        assert isinstance(plot, bool)
        assert path is None or isinstance(path, str)
        assert plot is True or path is not None, \
            f"the plot is not being shown or saved"
        assert isinstance(key, str)
        assert legend_name is None or isinstance(legend_name, str)
        if legend_name is None:
            legend_name = key
        assert title is None or isinstance(title, str)
        for lims in [x_lims, y_lims]:
            assert lims is None or isinstance(lims, list) \
                   and any([v is None or isinstance(v, t)
                            for v in lims for t in (int, float)]), \
                f"invalid limits {lims} ({type(lims)})"
        for label in [x_label, y_label]:
            assert label is None or isinstance(label, str), \
                f"invalid label {label}"
        size = ((21 / 2) / 2.54) * 1.5
        fig, ax = plt.subplots(1, 1,
                               figsize=(size, size), tight_layout=True)
        for phase_key, phase_name in [("train", "training"),
                                      ("val", "validation")]:
            sns.lineplot(data=self.logs, x="epoch", y=f"{key}_{phase_key}",
                         ax=ax)
        ax.legend([f"{phase_name} {legend_name}"
                   for phase_name in ["training", "validation"]])
        # title
        if title is not None:
            fig.suptitle(title)
        # limits
        if x_lims is not None:
            ax.set_xlim(*x_lims)
        if y_lims is not None:
            ax.set_ylim(*y_lims)
        # labels
        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)
        # eventually saves the plot
        if path is not None:
            path = join(self.path, path)
            if not isdir(path):
                makedirs(path)
            plt.savefig(join(path, f"{key}.svg"))
            plt.savefig(join(path, f"{key}.png"))
        # eventually plots
        if plot is True:
            plt.show()
