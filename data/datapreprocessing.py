"""This file contains functionality for data preprocessing.
"""

import os
import pandas as pd
import json


def get_file_names(ext: str = "tsv") -> list:
    """This function returns a list of all file names with the given extension in the current directory."""
    return [file for file in os.listdir() if file.endswith(ext)]


def convert_to_json(file: str) -> None:
    """This function converts a tsv file in the WebAnno format to a json file with the labels in the BIO format."""
    with open(file, "r") as f:
        # Read WebAnno format
        for line in f.readlines():
            if line.startswith("#FORMAT"):
                continue
