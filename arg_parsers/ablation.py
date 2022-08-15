import argparse
import math
import random
from os import makedirs
from os.path import isdir
from typing import Dict, Union


def get_args() -> Dict[str, Union[bool, str, int, float]]:
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument("--dataset_type",
                        type=str,
                        choices={"deap", "dreamer", "amigos", "seed"},
                        help="Type of dataset")
    parser.add_argument("--dataset_path",
                        type=str,
                        help="Path to dataset's directory")
    parser.add_argument("--windows_size",
                        type=float,
                        default=1,
                        help="Duration of the windows in seconds")
    parser.add_argument("--windows_stride",
                        type=float,
                        default=1,
                        help="Duration of stride of the windows in seconds")
    parser.add_argument("--dont_discretize_labels",
                        default=False,
                        action="store_true",
                        help="Whether not to discretize labels in {0, 1}")
    parser.add_argument("--dont_normalize_eegs",
                        default=False,
                        action="store_true",
                        help="Whether not to normalize the EEGs with zero mean and unit variance")
    parser.add_argument("--train_set_size",
                        default=0.8,
                        type=float,
                        help="The size of the training set, in percentage")

    # training args
    parser.add_argument("--batch_size",
                        default=64,
                        type=int,
                        help="Number of samples per batch")
    parser.add_argument("--min_epochs",
                        default=1,
                        type=int,
                        help="Minimum number of epochs")
    parser.add_argument("--max_epochs",
                        default=1000,
                        type=int,
                        help="Maximum number of epochs")
    parser.add_argument("--study_name",
                        default="unnamed",
                        type=str,
                        help="The name of the study")
    parser.add_argument("--grid_search",
                        default=False,
                        action="store_true",
                        help="Whether to test all possible combinations or a fixed amount of trials")
    parser.add_argument("--checkpoints_path",
                        type=str,
                        help="Path to where to save the saved")
    parser.add_argument("--seed",
                        type=int,
                        help="The seed for reproducibility")
    parser.add_argument("--learning_rate",
                        default=2e-4,
                        type=float,
                        help="Learning rate of the model")

    # tunable parameters
    for parameter in [
        "mels",
        "mel_window_size",
        "mel_window_stride",

        "users_embeddings",

        "encoder_only",
        "hidden_size",
        "num_layers",
        "positional_embedding_type",

        "dropout_p",

        "shifting",
        "cropping",
        "flipping",
        "noise_strength",
        "spectrogram_time_masking_perc",
        "spectrogram_frequency_masking_perc",

        "learning_rate",
    ]:
        parser.add_argument(f"--test_{parameter}",
                            default=False,
                            action="store_true")

    args = parser.parse_args()

    assert isdir(args.dataset_path)
    assert 0 < args.train_set_size < 1
    args.val_set_size = round(1 - args.train_set_size, 2)
    assert args.train_set_size + args.val_set_size == 1
    assert args.checkpoints_path is None or isinstance(args.checkpoints_path, str)
    if isinstance(args.checkpoints_path, str) and not isdir(args.checkpoints_path):
        makedirs(args.checkpoints_path)
    assert args.windows_size > 0
    assert args.windows_stride > 0

    assert args.batch_size >= 1
    assert args.min_epochs >= 1
    assert args.max_epochs >= 1 and args.max_epochs >= args.min_epochs
    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    assert args.learning_rate > 0

    return vars(args)
