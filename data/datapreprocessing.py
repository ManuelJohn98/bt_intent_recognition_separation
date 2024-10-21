"""This file contains functionality for data preprocessing.
"""

import os
import json
import re
from config import data_directory, raw_data_directory, load_config
from stats.collectstatistics import StatisticsCollector


def _get_file_names_from_dir(ext: str = "tsv") -> list:
    """This function returns a list of all file names
    with the given extension in the current directory.
    """
    return [file for file in os.listdir(raw_data_directory) if file.endswith(ext)]


def _remove_count_from_label(label: str) -> str:
    """This function will return a label without any trailing number."""
    match = re.match(r"([A-Za-z_\/]+)", label)
    return match.group(1)


def _preprocess_raw_data(file: str, ext: str = "tsv") -> dict:
    """This function takes a tsv file in the WebAnno
    format to have the data labeled in the BIO format.
    Returns a csv string.
    """
    STAT = load_config()["statistics"]
    file = os.path.join(raw_data_directory, file)
    labeled_data = {}
    labeled_data["data"] = []
    labels = set()
    with open(file, "r", encoding="utf8") as f:
        current_turn_no = 0
        current_label = ""
        # Read WebAnno format
        for line in f:
            labeled_data_point = dict()
            if line.startswith("#") or line == "\n":
                continue
            sep = ""
            match ext:
                case "tsv":
                    sep = "\t"
                case "csv":
                    sep = ","
                case _:
                    raise ValueError(f"Expected 'tsv' or 'csv', but got {ext}")
            elements = line.split(sep)  # line is guaranteed to have 4 elements
            turn_id = elements[0]
            turn_no = int(turn_id.split("-")[0])
            token = elements[2]
            # get the lable which is prefix of last element
            label_with_count = elements[3].split("[")[0].replace("\\", "")
            # convert labels to BIO format
            if label_with_count == "_":
                current_label = ""
                labeled_data_point["turn_no"] = turn_no
                labeled_data_point["label"] = "O"
                labeled_data_point["token"] = token
                labels.add("O")
                if STAT:
                    stats = StatisticsCollector()
                    stats.count_labeled("data", "other")
                continue
            if current_turn_no != turn_no:
                current_turn_no = turn_no
                current_label = label_with_count
                # sometimes the label has a number at the end
                label = _remove_count_from_label(label_with_count)
                labeled_data_point["turn_no"] = turn_no
                labeled_data_point["label"] = "B-" + label
                labeled_data_point["token"] = token
                labels.add("B-" + label)
                if STAT:
                    stats = StatisticsCollector()
                    stats.count_labeled("data", label)
            elif current_label != label_with_count:
                current_label = label_with_count
                label = _remove_count_from_label(label_with_count)
                labeled_data_point["turn_no"] = turn_no
                labeled_data_point["label"] = "B-" + label
                labeled_data_point["token"] = token
                labels.add("B-" + label)
                if STAT:
                    stats = StatisticsCollector()
                    stats.count_labeled("data", label)
                    stats.count_unlabeled("need_for_separation")
            else:
                # only use I for these intermediary labels without actual label
                labeled_data_point["turn_no"] = turn_no
                labeled_data_point["label"] = "I"
                labeled_data_point["token"] = token
                labels.add("I")
            # add the labaled data point to labeled data
            labeled_data["data"].append(labeled_data_point)
    labeled_data["labels"] = labels
    return labeled_data


def _transform_to_huggingface_format(labeled_data: list, labels: set) -> tuple:
    huggingface_format = {}
    meta_data = {}
    huggingface_format["data"] = []
    labels = sorted(labels)
    meta_data["id2label"] = {}
    meta_data["label2id"] = {}
    for i, label in enumerate(labels):
        meta_data["id2label"][i] = label
        meta_data["label2id"][label] = i
    current_turn_no = 0
    data_id = 0
    for data_point in labeled_data[1:]:
        turn_no = data_point["turn_no"]
        if current_turn_no != turn_no:
            # Prepare for new turn
            current_turn_no = turn_no
            data_id += 1
            huggingface_format["data"].append({})
            huggingface_format["data"][-1]["id"] = data_id
            huggingface_format["data"][-1]["labels"] = []
            huggingface_format["data"][-1]["tokens"] = []
        huggingface_format["data"][-1]["labels"].append(
            labels.index(data_point["label"])
        )
        huggingface_format["data"][-1]["tokens"].append(data_point["token"])
    meta_data["num_rows"] = huggingface_format["data"][-1]["id"]
    if load_config()["statistics"]:
        stats = StatisticsCollector()
        stats.add_statistics("num_rows", meta_data["num_rows"])
    huggingface_format["id2label"] = meta_data["id2label"]
    huggingface_format["label2id"] = meta_data["label2id"]
    huggingface_format["num_rows"] = meta_data["num_rows"]
    return huggingface_format, meta_data


# def _concat_json(json1: dict, json2: dict) -> dict:
#     """This function concatenates two json formatted memory objects."""
#     return {**json1, **json2}


def convert_all_raw_data(exentension: str) -> None:
    """This function converts all tsv files in the given directory
    to a single json file with the labels in the BIO format.
    """
    STAT = load_config()["statistics"]
    labeled_data = []
    labels = set()
    if exentension != "":
        for file in _get_file_names_from_dir(exentension):
            preprocessed_data = _preprocess_raw_data(file, exentension)
            labeled_data = labeled_data + preprocessed_data["data"]
            labels = labels.union(preprocessed_data["labels"])
    else:
        for file in _get_file_names_from_dir():
            preprocessed_data = _preprocess_raw_data(file)
            labeled_data = labeled_data + preprocessed_data["data"]
            labels = labels.union(preprocessed_data["labels"])
    transformed_labeled_data, meta_data = _transform_to_huggingface_format(
        labeled_data, labels
    )
    with open(
        os.path.join(data_directory, "preprocessed_data.json"), "w", encoding="utf8"
    ) as f:
        json.dump(transformed_labeled_data, f, ensure_ascii=False)
    with open(
        os.path.join(data_directory, "meta_data.json"), "w", encoding="utf8"
    ) as f:
        json.dump(meta_data, f, ensure_ascii=False)
    if STAT:
        stats = StatisticsCollector()
        stats.write_to_file()
