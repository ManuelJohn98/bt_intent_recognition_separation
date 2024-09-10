# generatestatistics.py

import os
import re
from pprint import pprint
from config import raw_data_directory, statistics_directory


def _statistics_raw_data_from_file(filename) -> dict:
    filename = os.path.join(raw_data_directory, filename)
    label_dict = {"multi_intents": 0}
    new_paragraph = True
    current_label = ""

    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line == "\n":
                new_paragraph = True
                continue
            if line.startswith("#"):
                continue
            elems = line.split(sep="\t")
            label = re.match(r"[a-z_\\\/]+", elems[3]).group(0)
            label = label.replace("\\", "")
            if label == "_":
                label = "other"
            if new_paragraph:
                if label in label_dict:
                    label_dict[label] += 1
                else:
                    label_dict[label] = 1
                new_paragraph = False
                current_label = label
            if current_label != label:
                if label in label_dict:
                    label_dict[label] += 1
                else:
                    label_dict[label] = 1
                current_label = label
                label_dict["multi_intents"] += 1
    return label_dict


def write_statistics_raw_data():
    label_dict = dict()
    for file in os.listdir(raw_data_directory):
        # add statistics of the current file to the label_dict
        other_label_dict = _statistics_raw_data_from_file(file)
        for label in other_label_dict:
            if label in label_dict:
                label_dict[label] += other_label_dict[label]
            else:
                label_dict[label] = other_label_dict[label]
    # sum all values except multi_intents
    label_dict["total"] = sum(
        [label_dict[label] for label in label_dict if label != "multi_intents"]
    )
    pprint(label_dict)

    filename = os.path.join(statistics_directory, "raw_data_statistics.txt")
    with open(filename, "w", encoding="utf8") as f:
        # f.write(f"# {filename}\n")
        for label in label_dict:
            f.write(f"{label} - {label_dict[label]}\n")
