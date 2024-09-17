#!bt_env/bin/python3
# intrec.py
"""This is the main file for the project.
It is concerned with handling the user input.
"""
import argparse
import os
from data.datapreprocessing import convert_all_raw_data
from config import data_directory, load_config, save_config


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
    args = parser.parse_args()

    # create config with stat flag
    config = dict()
    config["statistics"] = args.s
    save_config(config)

    if any("train" in m for m in args.mode):
        if "clean_train" in args.mode:
            # delete all preprocessed data
            if os.path.exists(os.path.join(data_directory, "preprocessed_data.json")):
                os.remove(os.path.join(data_directory, "preprocessed_data.json"))
        # preprocess data if not already done
        if not os.path.exists(os.path.join(data_directory, "processed_data.json")):
            convert_all_raw_data(args.extension)
        # if args.s:
        #     stats = StatisticsCollector()
        #     stats.write_to_file()
        # TODO: train model


if __name__ == "__main__":
    main()
