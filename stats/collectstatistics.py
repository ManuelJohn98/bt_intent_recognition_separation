# generatestatistics.py
"""This module contains the StatisticsCollector class for collecting and storing statistics."""

import os

# import re
# from pprint import pprint
from collections import Counter

# import json
from config import statistics_directory
from utils.utils import SingletonMeta


# def _statistics_raw_data_from_file(filename) -> dict:
#     filename = os.path.join(raw_data_directory, filename)
#     label_dict = {"multi_intents": 0}
#     new_paragraph = True
#     current_label = ""

#     with open(filename, "r", encoding="utf-8") as f:
#         for line in f.readlines():
#             if line == "\n":
#                 new_paragraph = True
#                 continue
#             if line.startswith("#"):
#                 continue
#             elems = line.split(sep="\t")
#             label = re.match(r"[a-z_\\\/]+", elems[3]).group(0)
#             label = label.replace("\\", "")
#             if label == "_":
#                 label = "other"
#             if new_paragraph:
#                 if label in label_dict:
#                     label_dict[label] += 1
#                 else:
#                     label_dict[label] = 1
#                 new_paragraph = False
#                 current_label = label
#             if current_label != label:
#                 if label in label_dict:
#                     label_dict[label] += 1
#                 else:
#                     label_dict[label] = 1
#                 current_label = label
#                 label_dict["multi_intents"] += 1
#     return label_dict


# def _sum_statistics_raw_data() -> dict:
#     label_dict = dict()
#     for file in os.listdir(raw_data_directory):
#         # add statistics of the current file to the label_dict
#         other_label_dict = _statistics_raw_data_from_file(file)
#         for label in other_label_dict:
#             if label in label_dict:
#                 label_dict[label] += other_label_dict[label]
#             else:
#                 label_dict[label] = other_label_dict[label]
#     # sum all values except multi_intents
#     label_dict["TOTAL"] = sum(
#         [label_dict[label] for label in label_dict if label != "multi_intents"]
#     )
#     pprint(label_dict)
#     return label_dict


# def _statistics_processed_data_from_file() -> dict:
#     filename = os.path.join(data_directory, "processed_data.json")
#     processed_data = dict()
#     with open(filename, "r", encoding="utf8") as f:
#         processed_data = json.load(f)
#     # count the number of labels
#     label_dict = dict()
#     for value in processed_data.values():
#         if value in label_dict:
#             label_dict[value] += 1
#         else:
#             label_dict[value] = 1
#     # sum all values
#     label_dict["TOTAL"] = sum(label_dict.values())
#     pprint(label_dict)
#     return label_dict


# def write_statistics(mode: str) -> None:
#     if mode == "raw":
#         label_dict = _sum_statistics_raw_data()
#         filename = os.path.join(statistics_directory, "raw_data_statistics.txt")
#     elif mode == "processed":
#         label_dict = _statistics_processed_data_from_file()
#         filename = os.path.join(statistics_directory, "processed_data_statistics.txt")
#     else:
#         raise ValueError(f"Unknown mode: {mode}")
#     with open(filename, "w", encoding="utf8") as f:
#         # f.write(f"# {filename}\n")
#         for label in label_dict:
#             f.write(f"{label} - {label_dict[label]}\n")


class StatisticsCollector(metaclass=SingletonMeta):
    """Statistics about the data and training can be given to the respective functions of this
    class to store and access them all through a singleton object.
    """

    def __init__(self) -> None:
        pass

    def count_labeled(self, name: str, label: str) -> None:
        """
        Increment the count of a specific label in a Counter attribute.

        This method checks if the attribute with the given name exists and is a Counter.
        If the attribute does not exist, it initializes it as a Counter.
        Then, it increments the count of the specified label in that Counter.

        Args:
            name (str): The name of the attribute to be updated.
            label (str): The label whose count needs to be incremented.

        Returns:
            None

        Raises:
            ValueError: If the name of the attribute is an empty string.
        """
        if name == "":
            raise ValueError("Name of the attribute cannot be empty.")
        try:
            getattr(self, name)
        except AttributeError:
            setattr(self, name, Counter())
        finally:
            getattr(self, name)[label] += 1

    def count_unlabeled(self, name: str) -> None:
        """
        Increments the count of an unlabeled attribute by 1.
        If the attribute does not exist, it initializes it to 0 before incrementing.

        Args:
            name (str): The name of the attribute to be incremented.

        Raises:
            ValueError: If the provided name is an empty string.
        """
        if name == "":
            raise ValueError("Name of the attribute cannot be empty.")
        try:
            getattr(self, name)
        except AttributeError:
            setattr(self, name, 0)
        finally:
            setattr(self, name, getattr(self, name) + 1)

    def reset_attribute(self, name: str) -> None:
        """
        Resets the specified attribute of the instance by deleting it.

        Args:
            name (str): The name of the attribute to reset.

        Raises:
            AttributeError: If the attribute does not exist.
        """
        getattr(self, name)
        delattr(self, name)

    def add_statistics(self, name: str, value: int) -> None:
        """
        Adds a new attribute to the instance with the given name and value.

        Args:
            name (str): The name of the attribute to add.
            value (int): The value to assign to the attribute.

        Raises:
            ValueError: If the name of the attribute is an empty string.
        """
        if name == "":
            raise ValueError("Name of the attribute cannot be empty.")
        setattr(self, name, value)

    def __str__(self) -> str:
        res = ""
        for attr in dir(self):
            if not attr.startswith("__"):
                if isinstance(getattr(self, attr), Counter):
                    res += f"----------{attr}----------\n"
                    for key, value in sorted(
                        getattr(self, attr).items(), key=lambda x: x[1], reverse=True
                    ):
                        res += f"{key}: {value}\n"
                elif isinstance(getattr(self, attr), int):
                    res += "---------------------------\n"
                    res += f"{attr}: {getattr(self, attr)}\n"
        return res

    def write_to_file(self) -> None:
        """
        Writes the string representation of the current instance to a file named "statistics.txt"
        located in the `statistics_directory`.

        The file is opened in write mode with UTF-8 encoding. If the file already exists, its
        contents will be overwritten.
        """
        with open(
            os.path.join(statistics_directory, "statistics.txt"), "w+", encoding="utf8"
        ) as f:
            print(self, file=f, end="")
