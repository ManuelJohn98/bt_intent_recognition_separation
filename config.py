"""This file sets certain global path and other configuration variables."""

import os
import yaml

# set root directory for this project
ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
# set data directory
DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, "data")
# set raw data directory
RAW_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, "raw_data")
# set statistics directory
STATISTICS_DIRECTORY = os.path.join(ROOT_DIRECTORY, "stats")
#  set model directory
MODELS_DIRECTORY = os.path.join(ROOT_DIRECTORY, "models")
# Set cross validation directory
CROSS_VALIDATION_DIRECTORY = os.path.join(ROOT_DIRECTORY, "cross_validation")

FREQUENCY_MAPPING = {
    "question_yes/no": "question",
    "info_request": "question",
    "answer": "info_provide",
    "disconfirm": "_",
}

MODELS = [
    "ikim-uk-essen/geberta-base",
    # "flair/ner-german-large",
    # "aseifert/distilbert-base-german-cased-comma-derstandard",
    "dbmdz/bert-base-german-cased",
]


def load_config() -> dict:
    """
    Load configuration from a YAML file.

    This function reads the 'config.yaml' file located in the current working directory
    and parses its contents into a dictionary.

    Returns:
        dict: A dictionary containing the configuration parameters.

    Raises:
        FileNotFoundError: If the 'config.yaml' file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    if not os.path.exists("config.yaml"):
        return {}
    with open("config.yaml", "r", encoding="utf8") as f:
        return yaml.safe_load(f)


def save_config(config: dict) -> None:
    """
    Save the given configuration dictionary to a YAML file.

    This function raises a KeyError to prevent deletion of
    configuration settings during runtime. Otherwise, it writes
    the new configuration to a file named 'yaml.config'.

    Args:
        config (dict): The configuration dictionary to be saved.

    Raises:
        KeyError: If any keys from the existing configuration are missing in the new configuration.
    """
    old_config = load_config()
    if any(key not in config for key in old_config.keys()):
        # stop user from deleting config settings
        raise KeyError("Deletion of configs not possible during runtime.")
    with open("config.yaml", "w", encoding="utf8") as f:
        yaml.dump(config, f)
