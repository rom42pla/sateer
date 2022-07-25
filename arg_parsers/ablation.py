import argparse
import random
from os import makedirs
from os.path import isdir
from typing import Dict, Union


def get_args() -> Dict[str, Union[bool, str, int, float]]:
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument("dataset_type",
                        type=str,
                        choices={"deap", "dreamer"},
                        help="Type of dataset")
    parser.add_argument("dataset_path",
                        type=str,
                        help="Path to dataset's directory")
    parser.add_argument("--windows_size",
                        default=None,
                        type=float,
                        help="Duration of the windows in seconds")
    parser.add_argument("--discretize_labels",
                        default=False,
                        action="store_true",
                        help="Whether not to discretize labels in {0, 1}")
    parser.add_argument("--normalize_eegs",
                        default=False,
                        action="store_true",
                        help="Whether not to normalize the EEGs with zero mean and unit variance")
    parser.add_argument("--checkpoints_path",
                        type=str,
                        help="Path to where to save the checkpoints")
    parser.add_argument("--seed",
                        type=int,
                        help="The seed for reproducibility")
    parser.add_argument("--train_set_size",
                        default=0.8,
                        type=float,
                        help="The size of the training set, in percentage")

    # training args
    parser.add_argument("--batch_size",
                        default=64,
                        type=int,
                        help="Type of validation algorithm ('kfold' or 'loso')")
    parser.add_argument("--min_epochs",
                        default=1,
                        type=int,
                        help="Minimum number of epochs")
    parser.add_argument("--max_epochs",
                        default=1000,
                        type=int,
                        help="Maximum number of epochs")
    parser.add_argument("--benchmark",
                        default=False,
                        action="store_true",
                        help="Whether to test a single training")

    # model args
    parser.add_argument("--num_encoders",
                        default=None,
                        type=int,
                        help="Number of encoders in FEEGT")
    parser.add_argument("--window_embedding_dim",
                        default=None,
                        type=int,
                        help="Dimension of the internal windows embedding in FEEGT")

    # regularization
    parser.add_argument("--dropout_p",
                        default=None,
                        type=float,
                        help="The amount of dropout to use")
    parser.add_argument("--noise_strength",
                        default=None,
                        type=float,
                        help="The amount of gaussian noise to add to the eegs")
    parser.add_argument("--gradient_clipping",
                        default=False,
                        action="store_true",
                        help="Whether to clip the gradients to 1")
    parser.add_argument("--stochastic_weight_average",
                        default=False,
                        action="store_true",
                        help="Whether to use the SWA algorithm")
    parser.add_argument("--masking",
                        default=None,
                        choices={None, "true", "True", "false", "False"},
                        type=str,
                        help="Whether not to mask a percentage of embeddings during training of FEEGT")
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        help="Learning rate of the model")

    parser.add_argument("--mels",
                        default=8,
                        type=int,
                        help="Number of mel banks")
    parser.add_argument("--mel_window_size",
                        default=1,
                        type=float,
                        help="Size of spectrogram's windows")
    parser.add_argument("--mel_window_stride",
                        default=0.05,
                        type=float,
                        help="Size of spectrogram's windows stride")

    args = parser.parse_args()

    assert isdir(args.dataset_path)
    assert 0 < args.train_set_size < 1
    args.val_set_size = 1 - args.train_set_size
    assert args.train_set_size + args.val_set_size == 1
    assert args.checkpoints_path is None or isinstance(args.checkpoints_path, str)
    if isinstance(args.checkpoints_path, str) and not isdir(args.checkpoints_path):
        makedirs(args.checkpoints_path)
    assert args.windows_size is None or args.windows_size > 0
    assert args.batch_size >= 1
    assert args.min_epochs >= 1
    assert args.max_epochs >= 1 and args.max_epochs >= args.min_epochs
    if args.seed is None:
        args.seed = random.randint(0, 1000000)

    assert not args.num_encoders or args.num_encoders >= 1
    assert not args.window_embedding_dim or args.window_embedding_dim >= 1
    if args.masking is not None:
        args.masking = True if args.masking in {"true", "True"} else False
    assert not args.dropout_p or 0 <= args.dropout_p < 1
    assert not args.noise_strength or args.noise_strength >= 0
    assert args.learning_rate > 0
    assert args.mels > 0

    return vars(args)
