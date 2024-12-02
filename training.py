"""This module is concerned with training the models."""

import os
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import evaluate
import numpy as np
from config import DATA_DIRECTORY, MODELS_DIRECTORY


class ModelTrainer:
    """This class is responsible for training the models."""

    def __init__(self, prefix: str, model_name: str, seq: bool) -> None:
        """
        Initializes the training class with the given parameters.

        Args:
            prefix (str): The prefix for the dataset files.
            model_name (str): The name of the pre-trained model to use.
            sep (bool): A flag indicating whether to use sequence classification or token classification.

        Raises:
            ValueError: If the train and test data are not found.

        Attributes:
            prefix (str): The prefix for the dataset files.
            model_name (str): The name of the pre-trained model to use.
            dataset (Dataset): The loaded dataset.
            metadata (dict): The metadata loaded from the metadata file.
            tokenizer (AutoTokenizer): The tokenizer for the model.
            labels_list (list): The list of labels from the metadata.
            sep (bool): A flag indicating whether to use sequence classification or token classification.
            data_collator (DataCollator): The data collator for the model.
            eval_metrics (Metric): The evaluation metrics for the model.
            model (PreTrainedModel): The pre-trained model for token or sequence classification.
        """
        self.prefix = prefix
        self.model_name = model_name
        self.dataset = load_dataset(
            "json",
            data_files={
                "train": os.path.join(DATA_DIRECTORY, f"{prefix}train_data.json"),
                "test": os.path.join(DATA_DIRECTORY, f"{prefix}test_data.json"),
            },
            field="data",
        )
        self.metadata = {}
        with open(
            os.path.join(DATA_DIRECTORY, f"{prefix}metadata.json"), "r", encoding="utf8"
        ) as f:
            self.metadata = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.labels_list = list(self.metadata["label2id"].keys())
        self.seq = seq
        if not seq:
            self.data_collator = DataCollatorForTokenClassification(self.tokenizer)
            self.eval_metrics = evaluate.load("seqeval")
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=len(self.labels_list),
                id2label=self.metadata["id2label"],
                label2id=self.metadata["label2id"],
            )
        else:
            self.data_collator = DataCollatorWithPadding(self.tokenizer)
            self.eval_metrics = evaluate.load("f1")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.labels_list),
                id2label=self.metadata["id2label"],
                label2id=self.metadata["label2id"],
            )

    def tokenize_and_align_labels(self, data_row):
        """
        Tokenizes input data and aligns labels with the tokenized words.
        To be used for token classification tasks.
        See: https://huggingface.co/docs/transformers/tasks/token_classification
        Args:
            data_row (dict): A dictionary containing at least two keys:
                - "tokens": A list of words/tokens to be tokenized.
                - "labels": A list of labels corresponding to each word/token.
        Returns:
            dict: A dictionary containing tokenized inputs and aligned labels. The keys include:
                - "input_ids": Tokenized input IDs.
                - "attention_mask": Attention mask for the tokenized inputs.
                - "labels": Aligned labels for the tokenized inputs,
                    where special tokens are set to -100.
        """
        tokenized_inputs = self.tokenizer(
            data_row["tokens"], truncation=True, is_split_into_words=True
        )

        _labels = []
        for i, label in enumerate(data_row["labels"]):
            word_ids = tokenized_inputs.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif (
                    word_idx != previous_word_idx
                ):  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            _labels.append(label_ids)

        tokenized_inputs["labels"] = _labels
        return tokenized_inputs

    def tokenize(self, data_row):
        """
        Tokenizes a given data row using the tokenizer.
        To be used for text classification tasks.
        See: https://huggingface.co/docs/transformers/tasks/sequence_classification
        """
        return self.tokenizer(data_row["text"], truncation=True)

    def compute_metrics(self, p):
        """
        Compute evaluation metrics for model predictions.
        This method computes evaluation metrics for model predictions. It handles
        two cases: when used in the context of the separated dataset, it
        computes metrics for token classification tasks. Otherwise, it computes
        metrics for (multiclass) sequence classification tasks.
        See: https://huggingface.co/docs/transformers/tasks/sequence_classification
        and https://huggingface.co/docs/transformers/tasks/token_classification
        respectively.
        Args:
            p (tuple): A tuple containing predictions and labels. For token
                       classification tasks, predictions and labels are 2D arrays. For
                       text classification tasks, predictions and labels are 1D arrays.
        Returns:
            dict: A dictionary containing the computed metrics. For sequence
                  labeling tasks, the dictionary contains precision, recall, f1,
                  and accuracy. For classification tasks, the dictionary contains
                  weighted precision, recall, f1, and accuracy.
        """
        if not self.seq:
            predictions, labels = p
            # Convert predictions to label IDs
            predictions = np.argmax(predictions, axis=2)

            true_predictions = [
                [self.labels_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [self.labels_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = self.eval_metrics.compute(
                predictions=true_predictions, references=true_labels
            )

            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        return self.eval_metrics.compute(
            predictions=predictions, references=labels, average="macro"
        )

    def train(
        self,
        output_name="intent_recognition_separation",
        learning_rate=2e-5,
        train_batch_size=16,
        eval_batch_size=16,
        num_train_epochs=30,
        weight_decay=0.01,
    ):
        """
        Trains the model using the specified parameters and saves the trained model.
        Args:
            output_name (str): The name to use for the output directory and logs.
            Default is "intent_recognition_separation".
            learning_rate (float): The learning rate for training. Default is 2e-5.
            train_batch_size (int): The batch size for training. Default is 16.
            eval_batch_size (int): The batch size for evaluation. Default is 16.
            num_train_epochs (int): The number of training epochs. Default is 30.
            weight_decay (float): The weight decay to apply. Default is 0.01.
        Returns:
            None
        """
        tokenize = self.tokenize_and_align_labels if not self.seq else self.tokenize
        tokenized_dataset = self.dataset.map(tokenize, batched=True)
        training_args = TrainingArguments(
            output_dir=os.path.join(
                MODELS_DIRECTORY, f"{self.model_name}_{self.prefix}{output_name}_log"
            ),
            learning_rate=learning_rate,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=num_train_epochs,  # 40
            weight_decay=weight_decay,
            logging_strategy="steps",
            logging_steps=250,  # 250
            eval_strategy="steps",
            eval_steps=250,  # 250
            # save_total_limit=1,
            load_best_model_at_end=True,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        self.model.save_pretrained(
            os.path.join(
                MODELS_DIRECTORY, f"{self.model_name}_{self.prefix}{output_name}_final"
            )
        )
