# generatestatistics.py
"""This module contains the StatisticsCollector class for collecting and storing statistics."""

import os

# import re
# from pprint import pprint
from collections import Counter
import json
import matplotlib.pyplot as plt

# import json
from config import STATISTICS_DIRECTORY, CROSS_VALIDATION_DIRECTORY
from utils.general import SingletonMeta, get_last_checkpoint_dir, get_train_eval_stats


class StatisticsCollector(metaclass=SingletonMeta):
    """Statistics about the data and training can be given to the respective functions of this
    class to store and access them all through a singleton object.
    """

    def __init__(self):
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
                elif isinstance(getattr(self, attr), (int, float)):
                    res += "---------------------------\n"
                    res += f"{attr.replace("_", " ")}: {getattr(self, attr)}\n"
        return res

    def plot_checkpoints(self, model_name: str, prefix: str, output_name: str):
        """
        Plots the training and evaluation statistics of a model against the number of steps.
        Args:
            model_name (str): The name of the model for which to plot the statistics.
        This function retrieves the training and evaluation statistics from the last checkpoint
        directory of the specified model. It then creates a figure with two subplots: one for
        the training statistics and one for the evaluation statistics. The training subplot
        shows the training loss over the steps and the evaluation subplot shows the evaluation
        loss, F1 score, accuracy, precision, and recall over the steps.
        """
        train_stats, eval_stats = get_train_eval_stats(
            get_last_checkpoint_dir(model_name, prefix, output_name)
        )

        # Plot the train and eval stats against the number of epochs
        # in two separate subplots
        fig, axs = plt.subplots(2)
        fig.suptitle(f"Train and Eval Stats\n{model_name}_{prefix}{output_name}")

        # add grid only to the y-axis
        axs[0].grid(axis="y")
        axs[1].grid(axis="y")

        # add vertical lines for each epoch
        # for i in range(150, 3000, 150):
        #     axs[0].axvline(x=i, color="gray", linestyle="--", alpha=0.5)
        #     axs[1].axvline(x=i, color="gray", linestyle="--", alpha=0.5)

        # set space between subplots
        fig.tight_layout(pad=3.0)

        # set size for subplots
        fig.set_size_inches(10, 10)

        axs[0].plot(
            [elem["step"] for elem in train_stats],
            [elem["loss"] for elem in train_stats],
            label="Train Loss",
        )
        axs[0].set_title("Train Stats")
        axs[0].set_xlabel("Steps")
        axs[0].set_ylabel("Loss")

        axs[1].plot(
            [elem["step"] for elem in eval_stats],
            [elem["eval_loss"] for elem in eval_stats],
            label="Eval Loss",
        )
        axs[1].plot(
            [elem["step"] for elem in eval_stats],
            [elem["eval_f1"] for elem in eval_stats],
            label="Eval F1",
        )
        axs[1].plot(
            [elem["step"] for elem in eval_stats],
            [elem["eval_accuracy"] for elem in eval_stats],
            label="Eval Accuracy",
            alpha=0.35,
        )
        axs[1].plot(
            [elem["step"] for elem in eval_stats],
            [elem["eval_precision"] for elem in eval_stats],
            label="Eval Precision",
            alpha=0.35,
        )
        axs[1].plot(
            [elem["step"] for elem in eval_stats],
            [elem["eval_recall"] for elem in eval_stats],
            label="Eval Recall",
            alpha=0.35,
        )
        axs[1].set_title("Eval Stats")
        axs[1].set_xlabel("Steps")
        axs[1].legend()
        plt.show()

    def write_to_file(self, prefix: str) -> None:
        """
        Writes the string representation of the current instance to a file named "statistics.txt"
        located in the `statistics_directory`.

        The file is opened in write mode with UTF-8 encoding. If the file already exists, its
        contents will be overwritten.
        """
        with open(
            os.path.join(STATISTICS_DIRECTORY, f"{prefix}statistics.txt"),
            "w+",
            encoding="utf8",
        ) as f:
            print(self, file=f, end="")

    def plot_cv(self, prefix: str) -> None:
        """
        Plots the cross-validation F1 scores for different models.
        Args:
            prefix (str): The prefix used to locate the prefixed tracked_stats.json file.
        Returns:
            None
        """
        cv = {}
        with open(
            os.path.join(CROSS_VALIDATION_DIRECTORY, f"{prefix}tracked_stats.json"),
            "r",
            encoding="utf8",
        ) as f:
            cv = json.load(f)

        colors = ["blue", "lightblue", "lightgreen", "green", "red", "orange", "purple"]

        for i, model in enumerate(cv, 1):
            stats = []
            for fold in cv[model]:
                max_f1 = 0.0
                for checkpoint in range(len(cv[model][fold]["eval"])):
                    max_f1 = max(max_f1, cv[model][fold]["eval"][checkpoint]["eval_f1"])
                stats.append(max_f1)
            plt.boxplot(
                stats,
                positions=[i],
                showfliers=False,
                widths=0.4,
                patch_artist=True,
                label=model,
                boxprops={"facecolor": colors[i - 1]},
            )
        # if "ablation" in prefix:
        #     # add horizontal line as baseline
        #     plt.axhline(y=0.21, color="green", linestyle="--", label="Baseline")
        # show grid
        plt.grid()
        # plt.yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1])
        # plt.ylim(0.4, 1)
        plt.ylabel("F1 Score")
        plt.xticks(list(range(1, len(cv) + 1)), [""] * len(cv))
        plt.xlim(0, len(cv) + 0.5)
        plt.legend()
        plt.show()
