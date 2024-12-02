"""This module contains utility functions."""

from typing import Any
import os
import json
import numpy as np
from config import MODELS_DIRECTORY, DATA_DIRECTORY, CROSS_VALIDATION_DIRECTORY


class SingletonMeta(type):
    """Singleton Metaclass"""

    _instances = {}

    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwds)
            cls._instances[cls] = instance
        return cls._instances[cls]


# def delete_models(*model_names: str) -> None:
#     """
#     Deletes specified models and their associated checkpoint folders.

#     Args:
#         *model_names (str): Variable length argument list of model names to be deleted.

#     Returns:
#         None

#     Raises:
#         FileNotFoundError: If any of the specified model directories or files do not exist.
#         PermissionError: If the program does not have permission to delete any of the
#         files or directories.

#     Example:
#         delete_models("model1", "model2")
#     """
#     for model_name in model_names:
#         model_name += "_intent_recognition_separation"
#         model = os.path.join(MODELS_DIRECTORY, model_name)
#         model_log = os.path.join(MODELS_DIRECTORY, f"{model_name}_log")
#         if os.path.exists(model):
#             for checkpoint_folder in os.listdir(model):
#                 checkpoint_folder_path = os.path.join(model, checkpoint_folder)
#                 for file in os.listdir(checkpoint_folder_path):
#                     os.remove(os.path.join(checkpoint_folder_path, file))
#                 os.rmdir(checkpoint_folder_path)
#             os.removedirs(model)
#         if os.path.exists(model_log):
#             for checkpoint_folder in os.listdir(model_log):
#                 checkpoint_folder_path = os.path.join(model_log, checkpoint_folder)
#                 for file in os.listdir(checkpoint_folder_path):
#                     os.remove(os.path.join(checkpoint_folder_path, file))
#                 os.rmdir(checkpoint_folder_path)
#             os.removedirs(model_log)


def create_proxy_data(prefix: str, data: list) -> list:
    """Creates proxy data, since the data is multi class and multi label.
    We are interested in the distribution of single intent turns and multi intent turns.

    This function processes the input data to determine if each turn contains
    two different intents, excluding intents with labels 10 and 11. It returns
    a new data structure with a boolean label for each turn indicating whether
    it contains two different intents.

    Args:
        data (list): A list of dictionaries, where each dictionary represents
                     a turn and contains an "id" and "labels".
        prefix (str): The prefix of the metadata file.

    Returns:
        list: A list of lists, where each inner list contains the turn "id"
              and a boolean label (1 if the turn contains two different intents,
              0 otherwise).
    """
    proxy_data = np.empty((0, 2), int)
    meta_data = {}
    with open(
        os.path.join(DATA_DIRECTORY, f"{prefix}metadata.json"), "r", encoding="utf8"
    ) as f:
        meta_data = json.load(f)
    for line in data:
        line_proxy = []
        # check if the line contains two different intents
        if (
            len(
                {
                    label
                    for label in line["labels"]
                    # Convert label to string representation
                    if meta_data["id2label"][str(label)] not in ["I", "O"]
                }
            )
            > 1
        ):
            line_proxy.append(line["id"])
            line_proxy.append(1)
        else:
            line_proxy.append(line["id"])
            line_proxy.append(0)
        proxy_data = np.append(proxy_data, [line_proxy], axis=0)
    return proxy_data


def get_last_checkpoint_dir(model_name: str, prefix: str, output_name: str) -> str:
    model_name += f"_{prefix}{output_name}_log"
    # Get the folder with the highest checkpoint number
    checkpoints_dir = os.path.join(MODELS_DIRECTORY, model_name)
    checkpoints = os.listdir(checkpoints_dir)
    checkpoints = [
        int(checkpoint.split("-")[1])
        for checkpoint in checkpoints
        if checkpoint.startswith("checkpoint")
    ]
    checkpoints.sort()
    last_checkpoint = checkpoints[-1]
    return os.path.join(checkpoints_dir, f"checkpoint-{last_checkpoint}")


def get_train_eval_stats(last_checkpoint_dir: str) -> tuple:
    # Load log_history from the last checkpoint
    log_history = []
    with open(
        os.path.join(last_checkpoint_dir, "trainer_state.json"),
        "r",
        encoding="utf8",
    ) as f:
        log_history = json.load(f)["log_history"]
    # Extract every two adjacent elements from the log_history
    eval_stats = []
    train_stats = []
    for i, elem in enumerate(log_history):
        if i % 2 == 0:
            train_stats.append(elem)
        else:
            eval_stats.append(elem)
    return train_stats, eval_stats


def delete_all_processed_data() -> None:
    """
    Deletes all processed data files in the data directory.

    This function iterates through all files in the `data_directory` and removes
    any file that ends with the ".json" extension.

    Args:
        None

    Returns:
        None
    """
    for file in os.listdir(DATA_DIRECTORY):
        if file.endswith(".json"):
            os.remove(os.path.join(DATA_DIRECTORY, file))


def delete_all_tracked_stats() -> None:
    """
    Deletes all tracked stats json files ending with 'tracked_stats.json'
    in the cross-validation directory.

    This function iterates over all files in the 'cross_validation_directory' and removes those
    that have filenames ending with 'tracked_stats.json'.

    Args:
        None

    Returns:
        None
    """
    for file in os.listdir(CROSS_VALIDATION_DIRECTORY):
        if file.endswith("tracked_stats.json"):
            os.remove(os.path.join(CROSS_VALIDATION_DIRECTORY, file))


def check_for_splits(prefix: str, splits: int) -> bool:
    """
    Checks if all the expected split files exist in the data directory.

    Args:
        prefix (str): The prefix of the split files.
        splits (int): The number of split files to check for.

    Returns:
        bool: True if all split files exist, False otherwise.
    """
    for i in range(splits):
        if not os.path.exists(
            os.path.join(DATA_DIRECTORY, f"{str(i)}_{prefix}train_data.json")
        ):
            return False
        if not os.path.exists(
            os.path.join(DATA_DIRECTORY, f"{str(i)}_{prefix}test_data.json")
        ):
            return False
    return True


def check_for_train_test(prefix: str) -> bool:
    if not os.path.exists(os.path.join(DATA_DIRECTORY, f"{prefix}train_data.json")):
        return False
    if not os.path.exists(os.path.join(DATA_DIRECTORY, f"{prefix}test_data.json")):
        return False
    return True
