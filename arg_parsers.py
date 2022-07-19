import argparse
import random
from os import makedirs
from os.path import isdir
from typing import Optional


def get_training_args():
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
    parser.add_argument("--limit_train_batches",
                        default=1.0,
                        help="Whether to limit the number of train batches")
    parser.add_argument("--checkpoints_path",
                        type=str,
                        help="Path to where to save the checkpoints")
    parser.add_argument("--seed",
                        type=int,
                        help="The seed for reproducibility")

    # training args
    parser.add_argument("--batch_size",
                        default=64,
                        type=int,
                        help="Type of validation algorithm ('kfold' or 'loso')")
    parser.add_argument("--min_epochs",
                        default=5,
                        type=int,
                        help="Minimum number of epochs")
    parser.add_argument("--max_epochs",
                        default=100,
                        type=int,
                        help="Maximum number of epochs")
    parser.add_argument("--validation",
                        default="k_fold",
                        type=str,
                        choices={"k_fold", "loso"},
                        help="Type of validation algorithm")
    parser.add_argument("--k_folds",
                        default=10,
                        type=int,
                        help="Number of folds for the cross validation")
    parser.add_argument("--setting",
                        default="cross_subject",
                        type=str,
                        choices={"cross_subject", "within_subject"},
                        help="The setting of the experiment, whether cross- or within-subject")
    parser.add_argument("--precision",
                        default=32,
                        type=int,
                        choices={16, 32},
                        help="Whether to use 32- ore 16-bit precision")
    parser.add_argument("--benchmark",
                        default=False,
                        action="store_true",
                        help="Whether to test a single training")

    # model args
    parser.add_argument("--model",
                        default="cnn_baseline",
                        type=str,
                        choices={"feegt", "eegt", "cnn_baseline"},
                        help="The model to use")
    parser.add_argument("--num_encoders",
                        default=2,
                        type=int,
                        help="Number of encoders in FEEGT")
    parser.add_argument("--window_embedding_dim",
                        default=512,
                        type=int,
                        help="Dimension of the internal windows embedding in FEEGT")
    parser.add_argument("--dropout_p",
                        default=0.2,
                        type=float,
                        help="The amount of dropout to use")
    parser.add_argument("--disable_masking",
                        default=False,
                        action="store_true",
                        help="Whether not to mask a percentage of embeddings during training of FEEGT")
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        help="Learning rate of the model")
    parser.add_argument("--mels",
                        default=8,
                        type=int,
                        help="Number of mel banks")

    args = parser.parse_args()

    assert isdir(args.dataset_path)
    assert args.checkpoints_path is None or isinstance(args.checkpoints_path, str)
    if isinstance(args.checkpoints_path, str) and not isdir(args.checkpoints_path):
        makedirs(args.checkpoints_path)
    assert args.windows_size is None or args.windows_size > 0
    assert args.batch_size >= 1
    assert args.min_epochs >= 1
    assert args.max_epochs >= 1 and args.max_epochs >= args.min_epochs
    if args.validation == "k_fold":
        assert args.k_folds >= 2
    assert args.limit_train_batches is None or \
           True in [isinstance(args.limit_train_batches, t) for t in [int, float, str]]
    if isinstance(args.limit_train_batches, str):
        if "." in args.limit_train_batches:
            args.limit_train_batches = float(args.limit_train_batches)
        else:
            args.limit_train_batches = int(args.limit_train_batches)
        assert args.limit_train_batches > 0
    if args.seed is None:
        args.seed = random.randint(0, 1000000)

    assert args.num_encoders >= 1
    assert 0 <= args.dropout_p < 1
    assert args.learning_rate > 0
    assert args.mels > 0

    return args
