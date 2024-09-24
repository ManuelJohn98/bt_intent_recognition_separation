"""This module contains utility functions."""

from typing import Any
import os
from config import models_directory


class SingletonMeta(type):
    """Singleton Metaclass"""

    _instances = {}

    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwds)
            cls._instances[cls] = instance
        return cls._instances[cls]


def delete_models(*model_names: str) -> None:
    """
    Deletes specified models and their associated checkpoint folders.

    Args:
        *model_names (str): Variable length argument list of model names to be deleted.

    Returns:
        None

    Raises:
        FileNotFoundError: If any of the specified model directories or files do not exist.
        PermissionError: If the program does not have permission to delete any of the files or directories.

    Example:
        delete_models("model1", "model2")
    """
    for model_name in model_names:
        model_name += "_intent_recognition_separation"
        model = os.path.join(models_directory, model_name)
        if not os.path.exists(model):
            continue
        for checkpoint_folder in os.listdir(model):
            checkpoint_folder_path = os.path.join(model, checkpoint_folder)
            for file in os.listdir(checkpoint_folder_path):
                os.remove(os.path.join(checkpoint_folder_path, file))
            os.rmdir(checkpoint_folder_path)
        os.removedirs(model)
