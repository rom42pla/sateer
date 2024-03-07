from os.path import join
from typing import Union, Dict, Any
import json

from arg_parsers.results import get_args
from plots import plot_metrics
from utils import merge_logs

args: Dict[str, Union[str]] = get_args()

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
