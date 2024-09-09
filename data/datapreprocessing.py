"""This file contains functionality for data preprocessing.
"""

import os
import json
from config import data_directory, raw_data_directory


def _get_file_names_from_dir(ext: str = "tsv") -> list:
    """This function returns a list of all file names with the given extension in the current directory."""
    return [file for file in os.listdir(raw_data_directory) if file.endswith(ext)]


def _convert_to_json(file: str) -> dict:
    """This function converts a tsv file in the WebAnno format to json-like objects in memory with the labels in the BIO format."""
    file = os.path.join(raw_data_directory, file)
    labeled_data = dict()
    with open(file, "r", encoding="utf8") as f:
        current_turn_no = 0
        current_label = ""
        # Read WebAnno format
        for line in f:
            if line.startswith("#") or line == "\n":
                continue
            elements = line.split("\t")  # line is guaranteed to have 4 elements
            turn_id = elements[0]
            turn_no = int(turn_id.split("-")[0])
            content = elements[2]
            # get the lable which is prefix of last element
            label = elements[3].split("[")[0].replace("\\", "")
            # convert labels to BIO format
            if label == "_":
                current_label = ""
                labeled_data[" ".join(("# " + turn_id, content))] = "O"
                continue
            if current_turn_no != turn_no:
                current_turn_no = turn_no
                current_label = label
                labeled_data[" ".join(("# " + turn_id, content))] = "B-" + label
            elif current_label != label:
                current_label = label
                labeled_data[" ".join(("# " + turn_id, content))] = "B-" + label
            else:
                labeled_data[" ".join(("# " + turn_id, content))] = "I-" + label
    return labeled_data


def _concat_json(json1: dict, json2: dict) -> dict:
    """This function concatenates two json formatted memory objects."""
    return {**json1, **json2}


def convert_all_raw_data(exentension: str) -> None:
    """This function converts all tsv files in the given directory to a single json file with the labels in the BIO format."""
    labeled_data = dict()
    if exentension != "":
        for file in _get_file_names_from_dir(exentension):
            labeled_data = _concat_json(labeled_data, _convert_to_json(file))
    else:
        for file in _get_file_names_from_dir():
            labeled_data = _concat_json(labeled_data, _convert_to_json(file))
    with open(
        os.path.join(data_directory, "processed_data.json"), "w", encoding="utf8"
    ) as f:
        json.dump(labeled_data, f)
