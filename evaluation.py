"""This file contains the Evaluator class."""

import os
import json
from collections import Counter
from training import ModelTrainer
from inference import IntentRecognition, IntentRecognitionSeparation, IntentSeparation
from config import DATA_DIRECTORY
from utils.general import get_train_eval_stats, get_last_checkpoint_dir


def cross_validate_model(
    prefix: str,
    model_name: str,
    output_name: str,
    splits=5,
    seq=False,
    learning_rate=2e-5,
    train_batch_size=16,
    eval_batch_size=16,
    num_train_epochs=30,
    weight_decay=0.01,
) -> dict:
    """
    Perform cross-validation on a model and return training and evaluation statistics.
    Args:
        prefix (str): Prefix for the metadata file.
        model_name (str): Name of the model to be trained.
        output_name (str): Name of the output directory for the model.
        splits (int, optional): Number of splits for cross-validation. Default is 5.
        seq (bool, optional): Whether to the task is sequence classification or not. Default is False.
        learning_rate (float, optional): Learning rate for training. Default is 2e-5.
        train_batch_size (int, optional): Batch size for training. Default is 16.
        eval_batch_size (int, optional): Batch size for evaluation. Default is 16.
        num_train_epochs (int, optional): Number of training epochs. Default is 30.
        weight_decay (float, optional): Weight decay for the optimizer. Default is 0.01.
    Returns:
        dict: A dictionary containing training and evaluation statistics for each split.
    """
    tracked_stats = {}
    for i in range(splits):
        # Rename metadata file
        os.rename(
            os.path.join(DATA_DIRECTORY, f"{prefix}metadata.json"),
            os.path.join(
                DATA_DIRECTORY,
                f"{str(i)}_{prefix}metadata.json",
            ),
        )

        mt = ModelTrainer(f"{str(i)}_{prefix}", model_name, seq)
        mt.train(
            output_name=output_name,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            weight_decay=weight_decay,
        )

        # Track train and eval stats
        tracked_stats[str(i)] = {}
        tracked_stats[str(i)]["train"], tracked_stats[str(i)]["eval"] = (
            get_train_eval_stats(
                get_last_checkpoint_dir(model_name, f"{str(i)}_{prefix}", output_name)
            )
        )

        # Rename metadata file back
        os.rename(
            os.path.join(
                DATA_DIRECTORY,
                f"{str(i)}_{prefix}metadata.json",
            ),
            os.path.join(DATA_DIRECTORY, f"{prefix}metadata.json"),
        )

    return tracked_stats


class Evaluator:
    """Evaluator class"""

    def __init__(self, classifier):
        """
        This instantiates the Evaluator class. It creates a bert model from a given string
        and calculates the F1 score for each class.

        Parameters:
        classifier (object): An instance of one of the following classes:
                             - IntentRecognition
                             - IntentSeparation
                             - IntentRecognitionSeparation

        Raises:
        ValueError: If the classifier is not an instance of the expected classes.

        Attributes:
        mode (str): The mode of the classifier, which can be "IntentRecognition",
                    "IntentSeparation", or "IntentRecognitionSeparation".
        model (object): The classifier instance.
        prefix (str): A prefix string based on the mode of the classifier.
        """
        self.mode = ""
        if isinstance(classifier, IntentRecognition):
            self.mode = "IntentRecognition"
        elif isinstance(classifier, IntentSeparation):
            self.mode = "IntentSeparation"
        elif isinstance(classifier, IntentRecognitionSeparation):
            self.mode = "IntentRecognitionSeparation"
        else:
            raise ValueError("Invalid classifier")
        self.model = classifier
        self.prefix = ""
        if self.mode == "IntentRecognition":
            self.prefix = "separated_"
        elif self.mode == "IntentSeparation":
            self.prefix = "ablation_"
        elif self.mode == "IntentRecognitionSeparation":
            self.prefix = ""

    def _calculate_f1(self, fp, fn, tp):
        if tp + fp == 0 or tp + fn == 0:
            return "N/A"
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def _get_y_true_y_pred(self):
        with open(
            os.path.join(DATA_DIRECTORY, f"{self.prefix}test_data.json"),
            "r",
            encoding="utf8",
        ) as f:
            data = json.load(f)
        y_true = []
        y_pred = []
        if not self.mode == "IntentRecognition":
            for row in data["data"]:
                y_true.append(
                    {
                        "labels": list(
                            map(
                                lambda x: self.model.model.config.id2label[x],
                                row["labels"],
                            )
                        ),
                        "id": row["id"],
                    }
                )
                y_pred.append(
                    list(
                        map(
                            lambda x: self.model.model.config.id2label[x],
                            self.model.predict(" ".join(row["tokens"])),
                        )
                    )
                )
        else:
            for row in data["data"]:
                y_true.append(
                    {
                        "label": self.model.model.config.id2label[row["label"]],
                        "id": row["id"],
                    }
                )
                y_pred.append(self.model.predict(row))
        return y_true, y_pred

    def get_per_class_f1(self) -> dict:
        """
        Calculate the F1 score for each class.
        This method calculates the F1 score for each class based on the true labels and
        predicted labels. It supports two modes: "IntentRecognition" and another mode
        where multiple labels are considered.
        Returns:
            dict: A dictionary where keys are class labels and values are
            the corresponding F1 scores.
        """
        y_true, y_pred = self._get_y_true_y_pred()
        y_true = list(
            map(
                lambda x: (
                    [x["label"]] if self.mode == "IntentRecognition" else x["labels"]
                ),
                y_true,
            )
        )
        fp_list = []
        fn_list = []
        tp_list = []

        for true, pred in zip(y_true, y_pred):
            if not self.mode == "IntentRecognition":
                for true_label, pred_label in zip(true, pred):
                    if true_label == pred_label:
                        tp_list.append(true_label)
                    else:
                        fp_list.append(pred_label)
                        fn_list.append(true_label)
            else:
                if true == pred:
                    tp_list.append(true)
                else:
                    fp_list.append(pred)
                    fn_list.append(true)
        results = {}
        fp_count = Counter(fp_list)
        fn_count = Counter(fn_list)
        tp_count = Counter(tp_list)
        for label in set(fp_list + fn_list + tp_list):
            results[label] = self._calculate_f1(
                fp_count.get(label, 0), fn_count.get(label, 0), tp_count.get(label, 0)
            )
        return results

    def correct_labels(self) -> list:
        y_true, y_pred = self._get_y_true_y_pred()
        correct = []
        for i, row in enumerate(y_true):
            if not self.mode == "IntentRecognition":
                if all(
                    row["labels"][j] == y_pred[i][j] for j in range(len(row["labels"]))
                ):
                    correct.append(y_pred[i])
            else:
                if row["label"] == y_pred[i]:
                    correct.append(y_pred[i])
        return correct

    def correct_labeled_indeces(self):
        y_true, y_pred = self._get_y_true_y_pred()
        correct = []
        for i, row in enumerate(y_true):
            if not self.mode == "IntentRecognition":
                if all(
                    row["labels"][j] == y_pred[i][j] for j in range(len(row["labels"]))
                ):
                    correct.append(row["id"])
            else:
                if row["label"] == y_pred[i]:
                    correct.append(row["id"])
        return correct
