"""This file sets certain global path and other configuration variables."""

import os
import yaml

# set root directory for this project
root_directory = os.path.dirname(os.path.abspath(__file__))
# set data directory
data_directory = os.path.join(root_directory, "data")
# set raw data directory
raw_data_directory = os.path.join(data_directory, "raw_data")
# set statistics directory
statistics_directory = os.path.join(root_directory, "stats")
#  set model directory
models_directory = os.path.join(root_directory, "models")
# Set cross validation directory
cross_validation_directory = os.path.join(root_directory, "cross_validation")


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
