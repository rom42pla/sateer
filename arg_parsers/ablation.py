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
    parser.add_argument("--study_name",
                        default="unnamed",
                        type=str,
                        help="The name of the study")
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
    parser.add_argument("--grid_search",
                        default=False,
                        action="store_true",
                        help="Whether to test all possible combinations or a fixed amount of trials")

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

    # tunable parameters
    parser.add_argument("--test_num_encoders",
                        default=False,
                        action="store_true")
    parser.add_argument("--test_embeddings_dim",
                        default=False,
                        action="store_true")
    parser.add_argument("--test_dropout_p",
                        default=False,
                        action="store_true")
    parser.add_argument("--test_noise",
                        default=False,
                        action="store_true")
    parser.add_argument("--test_masking",
                        default=False,
                        action="store_true")
    parser.add_argument("--test_mix_fourier",
                        default=False,
                        action="store_true")
    parser.add_argument("--test_mels",
                        default=False,
                        action="store_true")
    parser.add_argument("--test_mel_window_size",
                        default=False,
                        action="store_true")
    parser.add_argument("--test_mel_window_stride",
                        default=False,
                        action="store_true")

    parser.add_argument("--gradient_clipping",
                        default=False,
                        action="store_true",
                        help="Whether to clip the gradients to 1")
    parser.add_argument("--stochastic_weight_average",
                        default=False,
                        action="store_true",
                        help="Whether to use the SWA algorithm")
    parser.add_argument("--learning_rate",
                        default=0.0002,
                        type=float,
                        help="Learning rate of the model")

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

    assert args.learning_rate > 0
    assert any([v for v in [args.test_num_encoders, args.test_embeddings_dim,
                            args.test_masking, args.test_noise, args.test_dropout_p,
                            args.test_mix_fourier,
                            args.test_mels, args.test_mel_window_size, args.test_mel_window_stride]]), \
        f"you need to specify at least one parameter to analyze"

    return vars(args)
