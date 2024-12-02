"""This file contains the Inference classes."""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
)
from config import MODELS_DIRECTORY


class IntentRecognition:
    """IntentRecognition class"""

    def __init__(self, prefix: str, model_name: str, output_name: str, mode: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_name = os.path.join(
            MODELS_DIRECTORY, model_name + "_" + f"{prefix}{output_name}_final"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if mode not in ["logits", "prediction", "score", "labels", "eval"]:
            raise ValueError(
                f"Mode must be one of 'logits', 'prediction', 'score', 'labels' or 'eval', but got {mode}"
            )
        self.mode = mode

    def predict(self, text: str):
        """
        Predicts the label for the given text input using the model.

        Args:
            text (str): The input text to be classified.

        Returns:
            Depending on the mode set for the model, the function returns:
            - logits (torch.Tensor): The raw logits from the model if mode is "logits".
            - result (dict): A dictionary containing the label, score, and input text if mode is "score".
            - prediction (int): The predicted label index if mode is "prediction".
            - label (str): The predicted label if mode is "labels".
            - label_id (int): The label ID if mode is "eval".
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        logits = None
        with torch.no_grad():
            logits = self.model(**inputs).logits
        if self.mode == "logits":
            return logits
        prediction = torch.argmax(logits).item()
        label = self.model.config.id2label[prediction]
        score = torch.softmax(logits, dim=1).tolist()[0]
        result = {
            "label": label,
            "score": score[prediction],
            "input": text,
        }
        if self.mode == "score":
            return result
        if self.mode == "prediction":
            return prediction
        if self.mode == "labels":
            return label
        if self.mode == "eval":
            return self.model.config.label2id[label]


class IntentRecognitionSeparation:
    """IntentRecognitionSeparation class"""

    def __init__(self, prefix: str, model_name: str, output_name: str, mode: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_name = os.path.join(
            MODELS_DIRECTORY, model_name + "_" + f"{prefix}{output_name}_final"
        )
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        if mode not in ["logits", "prediction", "score", "labels", "eval"]:
            raise ValueError(
                f"Mode must be one of 'logits', 'prediction', 'score', 'labels' or 'eval', but got {mode}"
            )
        self.mode = mode

    def predict(self, text: str):
        """
        Predicts the labels for the given text input.

        Args:
            text (str): The input text to be tokenized and predicted.

        Returns:
            Depending on the mode set for the model, the function returns:
                - logits: Raw logits from the model if mode is "logits".
                - results: A list of dictionaries containing labels, scores, and input tokens if mode is "score".
                - prediction: The predicted token IDs if mode is "prediction".
                - labels: A list of predicted labels if mode is "labels".
                - eval: A list of label IDs for tokens starting with "▁" if mode is "eval".
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        logits = None
        with torch.no_grad():
            logits = self.model(**inputs).logits
        if self.mode == "logits":
            return logits
        prediction = torch.argmax(logits, dim=2)
        labels = [
            self.model.config.id2label[label_id.item()]
            for label_id in prediction[0]
            if label_id != -100
        ]
        results = []
        score = torch.softmax(logits, dim=2).tolist()[0]
        results = [
            {
                "label": labels[i],
                "score": score[i][self.model.config.label2id[labels[i]]],
                "input": input,
            }
            for i, input in enumerate(
                self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            )
            if input not in ["[CLS]", "[SEP]"]  # and not input.startswith("▁")
        ]
        if self.mode == "score":
            return results
        if self.mode == "prediction":
            return prediction
        if self.mode == "labels":
            return [result["label"] for result in results]
        if self.mode == "eval":
            return [
                self.model.config.label2id[result["label"]]
                for result in results
                if result["input"].startswith("▁")
            ]


class IntentSeparation(IntentRecognitionSeparation):
    """IntentSeparation class"""
