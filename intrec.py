#!bt_env/bin/python3
# intrec.py
"""This is the main file for the project.
It is concerned with handling the user input.
"""
import argparse
import os
from data.datapreprocessing import convert_all_raw_data
from config import data_directory, save_config
from utils.utils import delete_models
from training import train


def main():
    """Main function of this project."""
    parser = argparse.ArgumentParser(
        description='This program can instantiate a model for intent recognition and sepratation. \
            The model can either be trained on your data or you can replicate the results \
                of the bachelor thesis "Intent Recognition and Separation for E-DRZ" \
                by using the provided data in the zip file in the data folder.',
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        help="The mode in which the program should run: \
            train, clean_train, validate, infer, or experiment",
        choices=["train", "clean_train", "validate", "infer", "experiment"],
        required=True,
        nargs=1,
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
        "-e",
        "--extension",
        type=str,
        help="The extension of the raw data files to be processed, standard is tsv",
        default="tsv",
        required=False,
        nargs=1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="The seed for the random number generator connected to\
            shuffling the dataset and to cross validation to ensure reproducibility",
        default=42,
        required=False,
        nargs=1,
    )
    args = parser.parse_args()

    # create config with stat flag
    config = {}
    config["statistics"] = args.s
    config["seed"] = args.seed
    config["models"] = [
        "ikim-uk-essen/geberta-base",
        # "flair/ner-german-large",
        # "aseifert/distilbert-base-german-cased-comma-derstandard",
        "dbmdz/bert-base-german-cased",
    ]
    save_config(config)

    if any("train" in m for m in args.mode):
        if "clean_train" in args.mode:
            # delete all preprocessed data
            if os.path.exists(os.path.join(data_directory, "preprocessed_data.json")):
                os.remove(os.path.join(data_directory, "preprocessed_data.json"))
            # delete all models
            delete_models(*config["models"])
        # preprocess data if not already done
        if not os.path.exists(os.path.join(data_directory, "preprocessed_data.json")):
            convert_all_raw_data(args.extension)
        train(os.path.join(data_directory, "preprocessed_data.json"))


if __name__ == "__main__":
    main()
