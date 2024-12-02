"""This file contains the argument parsing for the main.py file."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse the arguments for the main.py file."""
    parser = argparse.ArgumentParser(
        description='This program can instantiate a model for intent recognition and sepratation. \
                The model can either be trained on your data or you can replicate the results \
                    of the bachelor thesis "Intent Recognition and Separation for E-DRZ" \
                    by using the provided data in the zip file in the data folder.',
    )
    parser.add_argument(
        "--dataset_mode",
        type=str,
        help="The mode in which the dataset should be loaded:\n\
            normal - load the dataset as is,\n\
            modified - this maps certain infrequent labels to (semantically) more \
            general labels, according to the mapping provided in the config.py file",
        choices=["normal", "modified"],
        default="normal",
        required=False,
        nargs=1,
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        help="The mode in which the program should run: \
            basic, ablation, separated",
        choices=["basic", "ablation", "separated"],
        required=True,
        nargs=1,
    )
    parser.add_argument(
        "--folds",
        type=int,
        help="The number of folds for cross validation, standard is 5.\
            Only relevant if cv is chosen as mode",
        default=5,
        required=False,
    )
    parser.add_argument(
        "--no_shuffle",
        help="Do not shuffle the dataset before any splitting, whether for \
            simple holdout or cross validation. Shuffling is done by default.",
        const=True,
        default=False,
        action="store_const",
        required=False,
    )
    parser.add_argument(
        "--test_size",
        help="What percentage of the dataset to use for holdout during training. \
            Default is 0.15",
        type=float,
        default=0.15,
        required=False,
    )
    parser.add_argument(
        "--clean",
        help="Start from scratch, delete all preprocessed data and models",
        const=True,
        default=False,
        action="store_const",
        required=False,
    )
    parser.add_argument(
        "-s",
        help="Generate statistics for the data",
        const=True,
        default=False,
        action="store_const",
        required=False,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="The seed for the random number generator connected to\
            shuffling the dataset and to cross validation to ensure reproducibility",
        default=42,
        required=False,
    )
    parser.add_argument(
        "--output_name",
        help="The name to give the trained model to distinguish\
            it from the ones that are not fine-tuned",
        type=str,
        default="ft",
        required=False,
    )
    parser.add_argument(
        "--preprocessing",
        help="Enable to preprocess the raw data files, \
            combine with --train or --cv to create data splits or folds respectively",
        const=True,
        default=False,
        action="store_const",
        required=False,
    )
    parser.add_argument(
        "--train",
        help="Enable to train the model(s)",
        const=True,
        default=False,
        action="store_const",
        required=False,
    )
    parser.add_argument(
        "--cv",
        help="Enable to run cross validation study on the data with the model(s)",
        const=True,
        default=False,
        action="store_const",
        required=False,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        required=False,
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        required=False,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        required=False,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=30,
        required=False,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        required=False,
    )
    parser.add_argument(
        "--infer",
        help="Enable to infer with the model(s)",
        const=True,
        default=False,
        action="store_const",
        required=False,
    )
    args = parser.parse_args()
    preprocess = 1 if args.preprocessing else 0
    train = 1 if args.train else 0
    cv = 1 if args.cv else 0
    infer = 1 if args.infer else 0
    if train + cv + infer != 1:
        raise ValueError("Exactly one of (train, cv, infer) must be chosen")
    if preprocess + infer > 1:
        raise ValueError("Preprocessing and infer cannot be chosen together")
    return args
