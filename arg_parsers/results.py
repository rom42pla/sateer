import argparse
from os.path import isdir
from typing import Dict, Union


def get_args() -> Dict[str, Union[bool, str, int, float]]:
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument("experiment_path",
                        type=str,
                        help="The folder of the experiment")
    parser.add_argument("setting",
                        type=str,
                        choices={"cross_subject", "within_subject"},
                        help="The setting of the experiment, whether cross- or within-subject")
    args = parser.parse_args()

    assert isdir(args.experiment_path)

    return vars(args)