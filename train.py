if __name__ == '__main__':
    import gc
    import logging
    from datetime import datetime
    from os import makedirs
    from os.path import join
    from pprint import pformat
    from typing import Union, Dict

    import torch.cuda

    from torch.utils.data import Subset

    from arg_parsers.train import get_args
    from plots import plot_metrics, plot_cross_subject
    from utils import parse_dataset_class, set_global_seed, save_to_json, init_logger, train_k_fold, merge_logs
    from datasets.eeg_emrec import EEGClassificationDataset
    from models.feegt import FouriEEGTransformer

    import torchaudio

    torchaudio.set_audio_backend("sox_io")

    # sets up the loggers
    init_logger()

    # retrieves line arguments
    args: Dict[str, Union[bool, str, int, float]] = get_args()
    logging.info(f"line args:\n{pformat(args)}")

    # sets the random seed
    set_global_seed(seed=args['seed'])

    # sets the logging folder
    datetime_str: str = datetime.now().strftime("%Y%m%d_%H:%M")
    experiment_name: str = f"{datetime_str}_{args['dataset_type']}_size={args['windows_size']}_stride={args['windows_stride']}"
    experiment_path: str = join(args['checkpoints_path'], experiment_name)
    makedirs(experiment_path)

    # saves the line arguments
    save_to_json(args, path=join(experiment_path, "line_args.json"))

    # sets up the dataset
    dataset_class = parse_dataset_class(name=args["dataset_type"])
    dataset: EEGClassificationDataset = dataset_class(
        path=args['dataset_path'],
        window_size=args['windows_size'],
        window_stride=args['windows_stride'],
        drop_last=True,
        discretize_labels=not args['dont_discretize_labels'],
        normalize_eegs=not args['dont_normalize_eegs'],
    )

    if args['setting'] == "cross_subject":
        if args['validation'] == "k_fold":
            # starts the kfold training
            logging.info(f"training on {args['dataset_type']} dataset "
                         f"({len(dataset)} samples)")
            train_k_fold(dataset=dataset,
                         experiment_path=experiment_path,
                         **args)
            plot_cross_subject(path=experiment_path)
        elif args['validation'] == "loso":
            raise NotImplementedError

    elif args['setting'] == "within_subject":
        if args['validation'] == "k_fold":
            for i_subject, subject_id in enumerate(dataset.subject_ids):
                # frees some memory
                gc.collect()
                # retrieves the samples for a single subject
                dataset_single_subject = Subset(dataset, [i for i, s in enumerate(dataset)
                                                          if dataset.subject_ids[s["subject_id"]] == subject_id])
                assert all([dataset.subject_ids[s["subject_id"]] == subject_id
                            for s in dataset_single_subject])
                # starts the kfold training
                logging.info(f"training on {args['dataset_type']}, subject {subject_id} "
                             f"({i_subject + 1}/{len(dataset.subject_ids)}, {len(dataset_single_subject)} samples)")
                train_k_fold(dataset=dataset_single_subject,
                             experiment_path=join(experiment_path, subject_id),
                             progress_bar=False,
                             **args)
                # frees some memory
                del dataset_single_subject
                if args['benchmark']:
                    break
        elif args['validation'] == "loso":
            raise NotImplementedError
    # frees some memory
    del dataset
    gc.collect()
