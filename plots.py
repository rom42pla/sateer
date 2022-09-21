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

from utils import read_json, parse_dataset_class


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
        },
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
    # print(competitors)

    metrics = pd.DataFrame([{
        "paper": "Ours",
        "dataset": line_args["dataset_type"],
        "windows_size": line_args["windows_size"],
        "windows_stride": line_args["windows_stride"],
        **{
            metric: ours[f"{metric}_mean_val"] * 100
            for metric in ["acc", "precision", "recall", "f1"]
        }
    }])
    metrics.to_csv(join(path, "metrics.csv"), index=False, float_format="%.2f")
    metrics.to_latex(join(path, "metrics.tex"), index=False, float_format="%.2f", bold_rows=True)
    print(metrics)


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
        scale: int = 10
):
    # eventually creates the plots folder
    plots_path = join(path, "plots")
    if not exists(plots_path):
        makedirs(plots_path)
    # parses ablation's logs in the different folders
    for parameter in [f for f in listdir(path)
                      if isdir(join(path, f)) and f != "plots"]:
        parameter_name = parameter.replace("_", " ").capitalize()
        data = {}
        for i_value in listdir(join(path, parameter)):
            if not exists(join(path, parameter, i_value, "logs.csv")):
                continue
            logs = pd.read_csv(join(path, parameter, i_value, "logs.csv"))
            value = read_json(join(path, f"{parameter}_{i_value}_desc.json"))["value"]
            data[value] = logs
        if len(data) == 0:
            continue

        fig = plt.figure(figsize=(scale * 2, scale), tight_layout=True)

        # fig, axs = plt.subplots(nrows=len(data.keys()), ncols=2,
        #                         figsize=(2 * scale, scale),
        #                         tight_layout=True)
        xlim = [0, max([logs["epoch"].max() for logs in data.values()])]
        for i_ax, (metric, y_label, title) in enumerate([
            ("loss", "loss", "loss"),
            ("acc_mean", "accuracy", "accuracy"),
            ("time", "time ($s$)", "time"),
        ]):
            if metric == "time":
                ax = plt.subplot2grid(shape=(len(data.keys()), 3), loc=(0, i_ax), rowspan=len(data.keys()))
                ax.set_title(f"{title.capitalize()} plot")
                x = np.arange(len(data.keys()))
                width = 0.3
                rects_train = ax.barh(x - width / 2, [data[key][f"{metric}_train"].mean() for key in sorted(list(data.keys()))],
                        height=width, label="training")
                rects_val = ax.barh(x + width / 2, [data[key][f"{metric}_val"].mean() for key in sorted(list(data.keys()))],
                        height=width, label="validation")
                ax.set_yticks(x, sorted(list(data.keys())))
                ax.set_xlim(0, 0.5)
                ax.legend(loc="upper right")
                ax.bar_label(rects_train, padding=3, fmt="%.3f")
                ax.bar_label(rects_val, padding=3, fmt="%.3f")
                ax.set_xlabel(y_label)
                ax.set_ylabel("value")
                ax.invert_yaxis()
                continue
            for i_key, key in enumerate(sorted(data.keys())):
                # ax = axs[i_key, i_ax]
                if metric in {"loss", "acc_mean"}:
                    ax = plt.subplot2grid(shape=(len(data.keys()), 3), loc=(i_key, i_ax))
                ax.set_title(f"{title.capitalize()} plot for value '{key}'")
                plotting_data = data[key].sort_values(by="epoch", ascending=True)[
                    ["epoch", f"{metric}_train", f"{metric}_val"]]
                plotting_data_train, plotting_data_val = plotting_data[["epoch", f"{metric}_train"]].dropna(), \
                                                         plotting_data[["epoch", f"{metric}_val"]].dropna()
                if metric in {"loss", "acc_mean"}:
                    ax.plot(plotting_data_train["epoch"],
                            plotting_data_train[f"{metric}_train"])
                    ax.plot(plotting_data_val["epoch"],
                            plotting_data_val[f"{metric}_val"])
                    ax.set_xlim(xlim)
                    if metric == "acc_mean":
                        ax.set_ylim(0.5, 1)
                    ax.set_xlabel("epoch")
                    ax.legend(["training", "validation"], loc="lower left")

                    # ax.set_xlabel("parameter value")
                    #     # ax.set_ylabel(y_label)
                ax.set_ylabel(y_label)
        fig.suptitle(f"Parameter: {parameter_name.lower()}")
        plt.savefig(join(plots_path, f"{parameter}.svg"))
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

def get_datasets_table(datasets_path: str):
    assert isdir(datasets_path)
    header = [
        "name",
        "subjects (\\#)",
        "trials (\\#)",
        "total length (\\SI{}{\\second})",
        "mean trial length (\\SI{}{\\second})",
        "StD. trial length (\\SI{}{\\second})",
        "sampling rate (\\SI{}{\\hertz})",
        "electrodes (\\#)",
        "labels (\\#)"
    ]
    print(" & ".join([n.capitalize() for n in header]), "\\\\")
    for dataset_name in ["deap", "dreamer", "amigos", "seed"]:
        if not isdir(join(datasets_path, dataset_name)):
            print(f"{dataset_name} not found into {datasets_path}")
        dataset: EEGClassificationDataset = parse_dataset_class(dataset_name)(
            path=join(datasets_path, dataset_name)
        )
        row = [
            dataset_name.upper(),
            len(dataset.subject_ids),
            len(dataset.eegs_data),
            np.sum([len(trial)/dataset.sampling_rate for trial in dataset.eegs_data]),
            np.mean([len(trial)/dataset.sampling_rate for trial in dataset.eegs_data]),
            np.std([len(trial)/dataset.sampling_rate for trial in dataset.eegs_data]),
            dataset.sampling_rate,
            len(dataset.electrodes),
            len(dataset.labels)
        ]
        row = [str(n) for n in row]
        assert len(row) == len(header)
        print(" & ".join(row), "\\\\")
        del dataset


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
    plot_ablation(path=join("checkpoints", "ablation", "dreamer", "20220823_180916"))
    # for filename in listdir(join("checkpoints", "training", "with_embeddings")):
    #     filepath = join("checkpoints", "training", "with_embeddings", filename)
    #     plot_cross_subject(filepath)
    #     print(filepath)
        # plot_ablation(path=filepath)
    # get_datasets_table(".")
    # get_datasets_table(join("..", "..", "datasets", "eeg_emotion_recognition"))