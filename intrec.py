#!bt_env/bin/python3
# intrec.py
"""This is the main file for the project.
It is concerned with handling the user input.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='This program can instantiate a model for intent recognition and sepratation. \
            The model can either be trained on your data or you can replicate the results of the bachelor thesis "Intent Recognition and Separation for E-DRZ" \
                by using the provided data in the zip file in the data folder.',
    )
    parser.add_argument(
        "-m",
        "-- mode",
        type=str,
        help="The mode in which the program should run: train, infer, or experiment",
        choices=["train", "infer", "experiment"],
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
    parser.parse_args()


if __name__ == "__main__":
    main()
