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
                        choices={"deap", "dreamer", "amigos", "seed"},
                        help="Type of dataset")
    parser.add_argument("dataset_path",
                        type=str,
                        help="Path to dataset's directory")
    parser.add_argument("--windows_size",
                        default=None,
                        type=float,
                        help="Duration of the windows in seconds")
    parser.add_argument("--windows_stride",
                        default=None,
                        type=float,
                        help="Duration of stride of the windows in seconds")
    parser.add_argument("--dont_discretize_labels",
                        default=False,
                        action="store_true",
                        help="Whether not to discretize labels in {0, 1}")
    parser.add_argument("--dont_normalize_eegs",
                        default=False,
                        action="store_true",
                        help="Whether not to normalize the EEGs with zero mean and unit variance")
    parser.add_argument("--checkpoints_path",
                        type=str,
                        help="Path to where to save the saved")
    parser.add_argument("--seed",
                        type=int,
                        help="The seed for reproducibility")

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
                        default=16,
                        choices={16, 32},
                        type=int,
                        help="Number of bits per float")

    # architecture
    parser.add_argument("--encoder_only",
                        default=False,
                        action="store_true",
                        help="Whether not to discard the decoder")
    parser.add_argument("--mixing_sublayer_type",
                        default="attention",
                        choices={"attention", "fourier", "identity", "linear"},
                        type=str,
                        help="Dimension of the internal embeddings")
    parser.add_argument("--num_encoders",
                        default=4,
                        type=int,
                        help="Number of encoders")
    parser.add_argument("--num_decoders",
                        default=4,
                        type=int,
                        help="Number of decoders")
    parser.add_argument("--num_attention_heads",
                        default=8,
                        type=int,
                        help="Number of attention heads")
    parser.add_argument("--hidden_size",
                        default=512,
                        type=int,
                        help="Dimension of the internal embeddings")
    parser.add_argument("--positional_embedding_type",
                        default="sinusoidal",
                        choices={"sinusoidal", "learned"},
                        type=str,
                        help="Dimension of the internal embeddings")
    parser.add_argument("--max_position_embeddings",
                        default=2048,
                        type=int,
                        help="The maximum possible length of the spectrograms")
    parser.add_argument("--dropout_p",
                        default=0.2,
                        type=float,
                        help="The amount of dropout to use")
    parser.add_argument("--disable_users_embeddings",
                        default=False,
                        action="store_true",
                        help="Whether not to use users' embeddings")
    parser.add_argument("--auto_lr_finder",
                        default=False,
                        action="store_true",
                        help="Whether to run an automatic learning range finder algorithm")

    # data augmentation
    parser.add_argument("--disable_data_augmentation",
                        default=False,
                        action="store_true",
                        help="Whether to disable EEGs' data augmentation")
    parser.add_argument("--disable_shifting",
                        default=False,
                        action="store_true",
                        help="Whether to disable EEGs' random shifting during data augmentation")
    parser.add_argument("--disable_cropping",
                        default=False,
                        action="store_true",
                        help="Whether to disable EEGs' random crop during data augmentation")
    parser.add_argument("--disable_flipping",
                        default=False,
                        action="store_true",
                        help="Whether to disable EEGs' flipping during data augmentation")
    parser.add_argument("--spectrogram_time_masking_perc",
                        default=0,
                        help="Amount of time masking in the spectrogram")
    parser.add_argument("--spectrogram_frequency_masking_perc",
                        default=0,
                        help="Amount of frequency masking in the spectrogram")
    parser.add_argument("--noise_strength",
                        default=0.01,
                        type=float,
                        help="The amount of gaussian noise to add to the EEGs during data augmentation")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="Learning rate of the model")
    parser.add_argument("--disable_gradient_clipping",
                        default=False,
                        action="store_true",
                        help="Whether to clip the gradients to 1")
    parser.add_argument("--disable_swa",
                        default=False,
                        action="store_true",
                        help="Whether to use the SWA algorithm")

    parser.add_argument("--mels",
                        default=16,
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
    assert args.checkpoints_path is None or isinstance(args.checkpoints_path, str)
    if isinstance(args.checkpoints_path, str) and not isdir(args.checkpoints_path):
        makedirs(args.checkpoints_path)
    assert args.windows_size is None or args.windows_size > 0
    assert args.windows_stride is None or args.windows_stride > 0
    assert args.batch_size >= 1
    assert args.min_epochs >= 1
    assert args.max_epochs >= 1 and args.max_epochs >= args.min_epochs
    if args.validation == "k_fold":
        assert args.k_folds >= 2
    if args.seed is None:
        args.seed = random.randint(0, 1000000)

    assert args.num_encoders >= 1
    assert args.num_decoders >= 1
    assert 0 <= args.dropout_p < 1
    assert args.noise_strength >= 0
    assert args.learning_rate > 0
    assert args.mels > 0

    return vars(args)
