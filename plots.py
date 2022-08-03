import pandas as pd
import json
import re
from os import listdir, makedirs
from os.path import join, exists, isdir
from typing import Union, Dict, Any, Optional, List

import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from utils import read_json


def plot_metrics(logs: pd.DataFrame,
                 metrics: List[str],
                 labels: Optional[List[str]] = None,
                 y_label: Optional[str] = None,
                 mode: str = "max",
                 experiment_path: Optional[str] = None):
    assert mode in {"min", "max"}
    if labels is None:
        labels = metrics
    folds = len(logs["fold"].unique())
    cols = 1
    for i in range(1, folds // 2 + 1):
        if folds % i == 0:
            cols = i
    fig, axs = plt.subplots(nrows=folds // cols, ncols=cols,
                            figsize=(cols * 5, (folds // cols) * 5),
                            tight_layout=True)
    for i_fold, ax in enumerate(axs.flat if cols > 1 else [axs]):
        for metric, label in zip(metrics, labels):
            if mode == "max":
                if "subject" in logs.columns:
                    best = logs[logs['fold'] == i_fold].groupby('subject').max().mean()[metric]
                else:
                    best = logs[logs['fold'] == i_fold].max()[metric]
            else:
                if "subject" in logs.columns:
                    best = logs[logs['fold'] == i_fold].groupby('subject').min().mean()[metric]
                else:
                    best = logs[logs['fold'] == i_fold].min()[metric]
            sns.lineplot(data=logs[logs["fold"] == i_fold], x="epoch", y=metric,
                         label=f"{label} ($\mu = {best:.3f})$",
                         ax=ax, palette="rocket")
        ax.set_xlabel("epoch")
        ax.set_ylabel(y_label)
        ax.set_xlim(0, logs["epoch"].max() + 5)
        ax.set_ylim(logs[metrics].min().min() - 0.05, logs[metrics].max().max() + 0.05)
        ax.set_title(f"Fold {i_fold + 1}")
        ax.legend(loc="lower left")
        ax.grid()
    if experiment_path is not None:
        if not isdir(join(experiment_path, "plots")):
            makedirs(join(experiment_path, "plots"))
        plt.savefig(join(experiment_path, "plots", f"{y_label}.png"))
    plt.show()


# def plot_ablation(logs: pd.DataFrame,
#                   experiment_path: Optional[str] = None):
#     tested_parameters = [c for c in logs.columns
#                          if c != "acc_mean_val"
#                          and len(logs[c].unique()) > 1]
#     if experiment_path is not None:
#         if not isdir(join(experiment_path, "plots")):
#             makedirs(join(experiment_path, "plots"))
#     for parameter in tested_parameters:
#         fig, ax = plt.subplots(nrows=1, ncols=1,
#                                figsize=(5, 5),
#                                tight_layout=True)
#         sns.barplot(data=logs, x=parameter, y="acc_mean_val",
#                     ax=ax, palette="rocket")
#         ax.set_xlabel(parameter)
#         ax.set_ylabel("accuracy")
#         ax.set_ylim(0.4, 1)
#         ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#         ax.grid(axis="y")
#         if experiment_path is not None:
#             plt.savefig(join(experiment_path, "plots", f"{parameter}.png"))
#         plt.show()

def plot_ablation(
        path: str,
        scale: int = 5
):
    # eventually creates the plots folder
    plots_path = join(path, "plots")
    if not exists(plots_path):
        makedirs(plots_path)
    # parses ablation's logs in the different folders
    logs = pd.DataFrame()
    for trial_folder in [f for f in listdir(path) if re.fullmatch(r"trial_[0-9]+", f)]:
        if not "logs.csv" in listdir(join(path, trial_folder)):
            continue
        trial_logs = pd.read_csv(join(path, trial_folder, "logs.csv"), index_col=False)
        trial_logs = trial_logs.groupby("epoch").max().reset_index()
        trial_logs["trial"] = int(trial_folder.split("_")[-1])
        logs = pd.concat([logs, trial_logs], ignore_index=True)
    logs = logs.sort_values(by=["trial", "epoch"]).drop("Unnamed: 0", axis=1)
    # parses tested parameters
    tested_parameters = read_json(path=join(path, "tested_args.json"))
    for parameter in tested_parameters:
        fig, (ax_loss, ax_acc) = plt.subplots(nrows=1, ncols=2,
                                              figsize=(2 * scale, scale),
                                              tight_layout=True)
        for column in ["loss_train", "loss_val"]:
            sns.lineplot(
                data=logs,
                x=parameter,
                y=column,
                palette="rocket",
                ax=ax_loss,
            )
        for column in ["acc_mean_train", "acc_mean_val"]:
            sns.lineplot(
                data=logs,
                x=parameter,
                y=column,
                palette="rocket",
                ax=ax_acc,
            )
        # sns.lineplot(
        #     data=logs,
        #     x=parameter,
        #     y="acc_mean_val",
        #     color="g",
        #     ax=ax_acc,
        # )
        for ax in [ax_loss, ax_acc]:
            ax.grid()
            ax.set_xlabel(parameter.replace("_", " ").capitalize())
        ax_loss.set_ylabel("loss")
        ax_acc.set_ylabel("accuracy (mean)")
        ax_loss.legend(labels=[
            f"training ($\mu={logs.groupby('trial').min()['loss_train'].mean():.3f}$)", "_",
            f"validation ($\mu={logs.groupby('trial').min()['loss_val'].mean():.3f}$)", "_"
        ])
        ax_acc.legend(labels=[
            f"training ($\mu={logs.groupby('trial').max()['acc_mean_train'].mean():.3f}$)", "_",
            f"validation ($\mu={logs.groupby('trial').max()['acc_mean_val'].mean():.3f}$)", "_"
        ])
        plt.savefig(join(plots_path, f"{parameter}.png"))
        plt.show()


if __name__ == "__main__":
    plot_ablation(path=join("checkpoints", "ablation", "20220803_090711_dreamer_embeddings"))
