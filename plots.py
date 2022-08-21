import math
import warnings
from pprint import pprint

import einops
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from datasets.amigos import AMIGOSDataset
from datasets.deap import DEAPDataset
from datasets.dreamer import DREAMERDataset
from datasets.eeg_emrec import EEGClassificationDataset
from models.layers import MelSpectrogram

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 16)
import json
import re
from os import listdir, makedirs
from os.path import join, exists, isdir, dirname
from typing import Union, Dict, Any, Optional, List

import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from utils import read_json


def plot_eegs(
        eegs: Union[np.ndarray, torch.Tensor],
        scale: Union[int, float] = 5,
        save_path: Optional[str] = None,
        hide_x_ticks: bool = True,
        hide_y_ticks: bool = True,
):
    assert len(eegs.shape) in {1, 2}
    if len(eegs.shape) == 1:
        eegs = einops.rearrange(eegs, "s -> s ()")
    assert scale > 0
    nrows = min(eegs.shape[-1], 8)
    fig, axs = plt.subplots(nrows=nrows, ncols=1,
                            figsize=(scale, scale * (nrows / 3)),
                            tight_layout=True)
    ylim = [eegs.min(), eegs.max()]
    for i_ax, ax in enumerate(axs if eegs.shape[-1] > 1 else [axs]):
        ax.plot(np.arange(eegs.shape[0]) / dataset.sampling_rate, eegs[:, i_ax])
        ax.set_ylim(ylim)
        ax.set_ylabel("amplitude")
        ax.set_title(f"electrode {dataset.electrodes[i_ax]}")
        if hide_x_ticks:
            ax.set_xticks([])
        if hide_y_ticks:
            ax.set_yticks([])
        ax.set_xlabel("time")
    if save_path is not None:
        if not isdir(dirname(save_path)):
            makedirs(dirname(save_path))
        plt.savefig(save_path)
    plt.show()


def plot_spectrogram(
        spectrogram: Union[np.ndarray, torch.Tensor],
        scale: Union[int, float] = 5,
        save_path: Optional[str] = None,
):
    assert len(spectrogram.shape) == 3
    assert scale > 0
    fig, axs = plt.subplots(nrows=min(spectrogram.shape[1], 8), ncols=1,
                            figsize=(scale, scale * 2),
                            tight_layout=True)
    ylim = [spectrogram.min(), spectrogram.max()]
    for i_ax, ax in enumerate(axs.flat):
        im = ax.imshow(einops.rearrange(spectrogram[:, i_ax, :], "s m -> m s"),
                       vmin=ylim[0], vmax=ylim[1], aspect="auto", cmap=plt.get_cmap("hot"))
        # colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel("amplitude", rotation=270)
        # axis
        ax.set_title(f"electrode {dataset.electrodes[i_ax]}")
        ax.set_ylabel("mels")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("time")
    if save_path is not None:
        if not isdir(dirname(save_path)):
            makedirs(dirname(save_path))
        plt.savefig(save_path)
    plt.show()


def plot_eeg_to_spectrogram(
        eegs: Union[np.ndarray, torch.Tensor],
        spectrogram: Union[np.ndarray, torch.Tensor],
        scale: Union[int, float] = 5,
        save_path: Optional[str] = None,
):
    assert len(eegs.shape) == 2
    assert len(spectrogram.shape) == 3
    assert eegs.shape[1] == spectrogram.shape[1]
    assert scale > 0
    nrows = eegs.shape[1]
    fig, (axs_eegs, axs_spectrogram) = plt.subplots(nrows=nrows, ncols=nrows,
                                                    figsize=(scale * nrows, scale * nrows),
                                                    tight_layout=True)
    ylim_eegs = [eegs.min(), eegs.max()]
    ylim_spectrogram = [spectrogram.min(), spectrogram.max()]
    # eegs
    for i_ax, ax in enumerate(axs_eegs.flat):
        ax.plot(np.arange(eegs.shape[0]) / dataset.sampling_rate, eegs[:, i_ax])
        ax.set_ylim(ylim_eegs)
        ax.set_ylabel("amplitude")
        ax.set_title(f"electrode {dataset.electrodes[i_ax]} (EEG)")
        ax.set_xlabel("time")
    # spectrogram
    for i_ax, ax in enumerate(axs_spectrogram.flat):
        im = ax.imshow(einops.rearrange(spectrogram[:, i_ax, :], "s m -> m s"),
                       vmin=ylim_spectrogram[0], vmax=ylim_spectrogram[1], aspect="auto", cmap=plt.get_cmap("hot"))
        # colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel("amplitude", rotation=270)
        # axis
        ax.set_title(f"electrode {dataset.electrodes[i_ax]} (Mel-spectrogram)")
        ax.set_ylabel("mels")
        ax.invert_yaxis()
        ax.set_xlabel("time")
    if save_path is not None:
        if not isdir(dirname(save_path)):
            makedirs(dirname(save_path))
        plt.savefig(save_path)
    plt.show()


def plot_paper_images(
        dataset: EEGClassificationDataset,
        save_path: Optional[str] = None,
):
    if save_path is not None:
        if not isdir(save_path):
            makedirs(save_path)
    # eegs
    sample_eegs = dataset[0]["eegs"]
    plot_eegs(eegs=sample_eegs, save_path=join(save_path, "eeg.svg"))
    # spectrogram
    spectrogram = MelSpectrogram(
        sampling_rate=dataset.sampling_rate,
        min_freq=0,
        max_freq=50,
        mels=32,
        window_size=10,
        window_stride=0.05,
    )(sample_eegs)
    plot_spectrogram(spectrogram=spectrogram, save_path=join(save_path, "spectrogram.svg"))
    # eeg to spectrogram
    plot_eeg_to_spectrogram(eegs=sample_eegs[:, :2], spectrogram=spectrogram[:, :2],
                            save_path=join(save_path, "eeg_to_spectrogram.svg"))
    # window
    plot_eegs(eegs=sample_eegs[:, 0], hide_x_ticks=False, hide_y_ticks=False,
              save_path=join(save_path, "eeg_window.svg"))


def plot_cross_subject(
        path: str,
        scale: int = 5
):
    # eventually creates the plots folder
    plots_path: str = join(path, "plots")
    if not exists(plots_path):
        makedirs(plots_path)
    # parses logs in the different folders for the experiment
    logs: pd.DataFrame = pd.DataFrame()
    for fold_folder in [f for f in listdir(path) if re.fullmatch(r"fold_[0-9]+", f)]:
        if not "logs.csv" in listdir(join(path, fold_folder)):
            continue
        fold_logs: pd.DataFrame = pd.read_csv(join(path, fold_folder, "logs.csv"), index_col=False)
        fold_logs = fold_logs.groupby("epoch").max().reset_index()
        fold_logs["fold"] = int(fold_folder.split("_")[-1])
        logs = pd.concat([logs, fold_logs], ignore_index=True)
    logs = logs.sort_values(by=["fold", "epoch"]).drop("Unnamed: 0", axis=1)
    # build our line for the final table
    ours: pd.DataFrame = logs.groupby("fold").max().mean()
    # parses line arguments
    line_args = read_json(path=join(path, "line_args.json"))
    # parses the competitors' table
    competitors: pd.DataFrame = pd.read_csv(
        f"https://docs.google.com/spreadsheets/d/1__LlQOL17InzSdcA5QCq3xAXhI4Kpcpo8l_VojkzeSA/gviz/tq?tqx=out:csv&sheet=0")
    # builds competitors' lines for the final table
    competitors = competitors.loc[competitors["dataset"] == line_args["dataset_type"]]
    # competitors = competitors.loc[competitors["setting"] == "cross-subject"]
    competitors = competitors.loc[competitors["validation"] == "10-fold"]
    competitors = competitors.drop(["url", "setting", "validation"], axis=1).sort_values(
        by=["windows_size", "windows_stride", "acc_valence"])
    # merges our method and the competitors' ones
    competitors = pd.concat([competitors, pd.DataFrame([{
        "paper": "Ours",
        "dataset": line_args["dataset_type"],
        "windows_size": line_args["windows_size"],
        "windows_stride": line_args["windows_stride"],
        **{
            f"acc_{column.split('_')[1]}": ours[column] * 100
            for column in ours.index
            if re.fullmatch(r"acc_.+_val", column)
               and column != "acc_mean_val"
        },
    }])], ignore_index=True)
    # rename the columns
    competitors = competitors.rename({
        "windows_size": "windows size (s)",
        "windows_stride": "windows stride (s)",
        **{
            column: f"accuracy ({column.split('_')[1]})"
            for column in competitors.columns
            if re.fullmatch(r"acc_.+", column)
        }
    }, axis=1)
    # capitalize all the columns
    competitors = competitors.rename({
        column: column.capitalize()
        for column in competitors
    }, axis=1)
    # removes unused labels
    competitors = competitors.dropna(axis=1, how="all")
    # saves the table as .csv and latex table
    competitors.to_csv(join(path, "competitors.csv"), index=False, float_format="%.2f")
    competitors.to_latex(join(path, "competitors.tex"), index=False, float_format="%.2f", bold_rows=True)
    print(competitors)


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
    if len(tested_parameters) > 1:
        return
    parameter = tested_parameters[0]
    keys = sorted(logs[parameter].unique().tolist())
    # loops through parameters

    fig, axs = plt.subplots(nrows=len(keys), ncols=2,
                            figsize=(2 * scale, scale),
                            tight_layout=True)
    xlim = [0, logs["epoch"].max()]
    for i_ax, (metric, y_label) in enumerate([
        ("loss", "loss"),
        ("acc_mean", "accuracy"),
        # ("time", "time ($s$)"),
    ]):
        for i_key, key in enumerate(keys):
            ax = axs[i_key, i_ax]
            ax.plot(
                logs[logs[parameter] == key].sort_values(by="epoch", ascending=True)[f"{metric}_train"])
            ax.plot(logs[logs[parameter] == key].sort_values(by="epoch", ascending=True)[f"{metric}_val"])
            ax.set_xlim(xlim)
            ax.set_xlabel("epoch")
        # metric_train, metric_val = [], []
        # for key in keys:
        #     best_row = logs[logs[parameter] == key].sort_values(by="loss_val", ascending=True).iloc[0]
        #     metric_train += [best_row[f"{metric}_train"]]
        #     metric_val = [best_row[f"{metric}_val"]]
        # x = np.arange(len(keys))
        # width = 0.4
        # rects1 = ax.bar(x - width / 2, metric_train, width, label="training")
        # rects2 = ax.bar(x + width / 2, metric_val, width, label="validation")
        #
        # ax.set_xlabel("parameter value")
        # ax.set_ylabel(y_label)
        # ax.set_xticks(x, keys)
        # ax.legend(loc="lower right")
        # ax.bar_label(rects1, padding=3, fmt='%.3f')
        # ax.bar_label(rects2, padding=3, fmt='%.3f')
        # if metric == "acc_mean":
        #     ax.set_ylim(0.5, 1.1)
        # if metric == "time":
        #     ax.set_ylim(0, None)
        fig.suptitle(parameter.replace("_", " ").capitalize())

    plt.savefig(join(plots_path, f"{parameter}.png"))
    plt.show()


def get_best_parameters_combination(
        checkpoints_path: str,
        dataset_type: str
):
    assert exists(checkpoints_path)
    logs = pd.DataFrame()
    for ablation_folder in listdir(checkpoints_path):
        line_args = read_json(path=join(checkpoints_path, ablation_folder, "line_args.json"))
        if line_args['dataset_type'] == dataset_type:
            for trial_folder in [f for f in listdir(join(checkpoints_path, ablation_folder)) if
                                 re.fullmatch(r"trial_[0-9]+", f)]:
                if not "logs.csv" in listdir(join(checkpoints_path, ablation_folder, trial_folder)):
                    continue
                trial_logs = pd.read_csv(join(checkpoints_path, ablation_folder, trial_folder, "logs.csv"),
                                         index_col=False)
                trial_logs = trial_logs.groupby("epoch").max().reset_index()
                trial_logs["trial"] = int(trial_folder.split("_")[-1])
                logs = pd.concat([logs, trial_logs], ignore_index=True)
    logs = logs.sort_values(by=["acc_mean_val"], ascending=False).drop("Unnamed: 0", axis=1)
    return logs.iloc[0].to_dict()


if __name__ == "__main__":
    # deap_best_parameters = get_best_parameters_combination(checkpoints_path=join("checkpoints", "ablation"),
    #                                                           dataset_type="deap")
    # amigos_best_parameters = get_best_parameters_combination(checkpoints_path=join("checkpoints", "ablation"),
    #                                                          dataset_type="amigos")
    # dreamer_best_parameters = get_best_parameters_combination(checkpoints_path=join("checkpoints", "ablation"),
    #                                                           dataset_type="dreamer")
    # dataset: EEGClassificationDataset = AMIGOSDataset(
    #     path=join("..", "..", "datasets", "eeg_emotion_recognition", "amigos"),
    #     window_size=10,
    #     window_stride=2,
    #     drop_last=True,
    #     discretize_labels=True,
    #     normalize_eegs=True,
    # )
    # plot_paper_images(dataset=dataset, save_path=join("imgs", "paper"))
    # plot_ablation(path=join("saved", "ablation_saved", "dreamer_data_augmentation"))
    for filename in listdir(join("checkpoints", "ablation", "dreamer")):
        filepath = join("checkpoints", "ablation", "dreamer", filename)
        plot_ablation(path=filepath)
