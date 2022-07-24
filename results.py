import json
import re
from os import listdir, makedirs
from os.path import join, exists, isdir
from typing import Union, Dict, Any

import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

from arg_parsers import get_results_args
from plots import plot_metrics
from utils import merge_logs

args: Dict[str, Union[str]] = get_results_args()

# parses line arguments for the training
with open(join(args["experiment_path"], "line_args.json"), "r") as fp:
    line_args: Dict[str, Any] = json.load(fp)

# merges all the logs into a single object
logs = merge_logs(args["experiment_path"])

# plots some metrics
plot_metrics(logs=logs,
             metrics=["acc_mean_train", "acc_mean_val"],
             labels=["training", "validation"],
             y_label="accuracy",
             mode="max",
             experiment_path=args["experiment_path"])

plot_metrics(logs=logs,
             metrics=["loss_train", "loss_val"],
             labels=["training", "validation"],
             y_label="loss",
             mode="min",
             experiment_path=args["experiment_path"])
