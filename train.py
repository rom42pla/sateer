import gc
import json
import logging
import random
from datetime import datetime
from os import listdir, makedirs
from os.path import join, isdir
from pprint import pprint

import numpy as np
import pandas as pd
import torch.cuda
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning

from tqdm.autonotebook import tqdm

from arg_parsers import get_training_args
from datasets.deap_preprocessed import DEAPDataset
from datasets.dreamer import DREAMERDataset
from models.cnn_baseline import CNNBaseline
from models.eegt import EEGT
import pytorch_lightning as pl
import intel_extension_for_pytorch as ipex

import warnings

# suppresses some warnings
warnings.filterwarnings("ignore", category=LightningDeprecationWarning)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# retrieves line arguments
args = get_training_args()

# sets the random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# sets the logging folder
datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"{datetime_str}_{args.setting}_{args.validation}"
makedirs(join(args.checkpoints_path, experiment_name))

# saves the line arguments
with open(join(args.checkpoints_path, experiment_name, "line_args.json"), 'w') as fp:
    json.dump(vars(args), fp, indent=4)
print(f"Line args")
pprint(vars(args))

# sets up the dataset to use
if args.dataset_type == "deap":
    dataset_class = DEAPDataset
elif args.dataset_type == "dreamer":
    dataset_class = DREAMERDataset

print(f"starting {args.setting} training with {args.validation} validation")
if args.setting == "cross_subject":
    print(f"loading dataset {args.dataset_type} from {args.dataset_path}")
    dataset = dataset_class(path=args.dataset_path,
                            split_in_windows=True if args.windows_size is not None else False,
                            window_size=args.windows_size, drop_last=True,
                            discretize_labels=args.discretize_labels, normalize_eegs=args.normalize_eegs,
                            validation=args.validation, k_folds=args.k_folds,
                            labels_to_use=["valence", "arousal", "dominance"],
                            batch_size=args.batch_size)
    # dataset.plot_samples()
    if args.validation == "k_fold":
        raise NotImplementedError
    elif args.validation == "loso":
        raise NotImplementedError
elif args.setting == "within_subject":
    if args.validation == "k_fold":
        subject_ids = dataset_class.get_subject_ids_static(args.dataset_path)
        for i_subject, subject_id in tqdm(enumerate(subject_ids),
                                          desc="looping through subjects", total=len(subject_ids)):
            dataset = dataset_class(path=args.dataset_path,
                                    subject_ids_to_use=subject_id,
                                    split_in_windows=True if args.windows_size is not None else False,
                                    window_size=args.windows_size, drop_last=True,
                                    discretize_labels=args.discretize_labels, normalize_eegs=args.normalize_eegs,
                                    validation=args.validation, k_folds=args.k_folds,
                                    labels_to_use=["valence", "arousal", "dominance"],
                                    batch_size=args.batch_size)
            dataset.setup("fit")
            for i_fold in tqdm(range(dataset.k_folds), desc="fold"):
                gc.collect()
                dataset.set_k_fold(i_fold)

                if args.model == "eegt":
                    model: pl.LightningModule = EEGT(in_channels=len(dataset.electrodes),
                                                     labels=dataset.labels_to_use,
                                                     sampling_rate=dataset.sampling_rate,
                                                     windows_length=dataset.window_size,
                                                     num_encoders=args.num_encoders, num_decoders=args.num_decoders,
                                                     window_embedding_dim=args.window_embedding_dim,
                                                     learning_rate=args.learning_rate,
                                                     mask_perc_min=0.05, mask_perc_max=0.2)
                elif args.model == "cnn_baseline":
                    model: pl.LightningModule = CNNBaseline(in_channels=len(dataset.electrodes),
                                                            labels=dataset.labels_to_use,
                                                            sampling_rate=dataset.sampling_rate,
                                                            window_embedding_dim=args.window_embedding_dim,
                                                            learning_rate=args.learning_rate)
                # model.to("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(memory_format=torch.channels_last)
                optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
                model.train()
                model, optimizer = ipex.optimize(model, optimizer=optimizer)
                print("ok")
                print(model)
