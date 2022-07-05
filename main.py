import gc
from datetime import datetime
from os.path import join

import numpy as np
import torch.cuda
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm

from arg_parsers import get_training_args
from datasets.deap import DEAPDataset
from models.eegt import EEGEmotionRecognitionTransformer
import pytorch_lightning as pl

args = get_training_args()
experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"loading dataset {args.dataset_type} from {args.dataset_path}")
if args.dataset_type == "deap":
    dataset = DEAPDataset(path=args.dataset_path,
                          windows_size=args.windows_size, drop_last=True,
                          discretize_labels=args.discretize_labels, normalize_eegs=True,
                          validation=args.validation, k_folds=args.k_folds,
                          batch_size=args.batch_size)
# model = EEGEmotionRecognitionTransformer(in_channels=32,
#                                          labels=4) \
#     .to("cuda" if torch.cuda.is_available() else "cpu")
# model(torch.randn(64, 128, 32))
# exit()
print(f"starting training with {dataset.validation} validation")
if dataset.validation == "k_fold":
    for i_fold in range(dataset.k_folds):
        print(f"training fold_{i_fold}")
        gc.collect()
        dataset.set_k_fold(i_fold)
        dataset.setup(stage="fit")

        model = EEGEmotionRecognitionTransformer(in_channels=32,
                                                 labels=["valence", "arousal", "dominance", "liking"]) \
            .to("cuda" if torch.cuda.is_available() else "cpu")
        trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, precision=32,
                             max_epochs=args.max_epochs, check_val_every_n_epoch=1,
                             num_sanity_val_steps=args.batch_size,
                             logger=CSVLogger(args.checkpoints_path, name=experiment_name, version=f"fold_{i_fold}"),
                             limit_train_batches=args.limit_train_batches,
                             limit_val_batches=args.limit_train_batches,
                             enable_checkpointing=True if args.checkpoints_path is not None else False,
                             callbacks=[
                                 ModelCheckpoint(dirpath=join(args.checkpoints_path, experiment_name, f"fold_{i_fold}"),
                                                 save_top_k=1,
                                                 monitor="loss_val", mode="min",
                                                 filename=args.dataset_type + "_{loss_val:.3f}_{epoch:02d}"),
                                 EarlyStopping(monitor="acc_val",
                                               min_delta=0, patience=20,
                                               verbose=False, mode="max", check_on_train_epoch_end=False),
                             ] if args.checkpoints_path is not None else [])
        trainer.fit(model, datamodule=dataset)
        del trainer, model
        # eventually stops validation
        if args.single_validation_step:
            break

elif dataset.validation == "loso":
    for i_subject in dataset.subjects_ids_indices.keys():
        gc.collect()
        dataset.set_loso_index(i_subject)
        dataset.setup(stage="fit")

        model = EEGEmotionRecognitionTransformer()
        model = model.double()

        trainer = pl.Trainer(gpus=0, precision=32, max_epochs=100, check_val_every_n_epoch=2,
                             logger=False, enable_checkpointing=False)
        del trainer, model
        # trainer.fit(model, datamodule=dataset)

# deap_dataloader = DEAPDataloader(dataset=deap_dataset, batch_size=256)
# print(deap_dataset[0][0].shape)

# model = MyModel()
# print(model)
