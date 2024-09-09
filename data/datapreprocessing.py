"""This file contains functionality for data preprocessing.
"""

import os
import json


def _get_file_names(ext: str = "tsv") -> list:
    """This function returns a list of all file names with the given extension in the current directory."""
    return [file for file in os.listdir() if file.endswith(ext)]


def _convert_to_json(file: str) -> dict:
    """This function converts a tsv file in the WebAnno format to json-like objects in memory with the labels in the BIO format."""
    labeled_data = dict()
    with open(file, "r") as f:
        # Read WebAnno format
        for line in f.readlines():
            if line.startswith("#") or line == "\n":
                continue
        elements = line.split("\t")  # line is guaranteed to have 4 elements
        turn_no = elements[0]
        content = elements[2]
        label = (
            elements[3].split("[")[0].replace("_", "other")
        )  # get the lable which is prefix of last element
        labeled_data[" # ".join(("# " + turn_no, content))] = label
    return labeled_data


def _concat_json(json1: dict, json2: dict) -> dict:
    """This function concatenates two json formatted memory objects."""
    return {**json1, **json2}


def convert_all_raw_data(path: str = "data/raw_data") -> dict:
    """This function converts all tsv files in the given directory to a single json file with the labels in the BIO format."""
    labeled_data = dict()
    for file in _get_file_names():
        labeled_data = _concat_json(labeled_data, _convert_to_json(file))
    with open("data/labeled_data.json", "w") as f:
        json.dump(labeled_data, f)
