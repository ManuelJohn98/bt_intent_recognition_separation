import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
)
from config import MODELS_DIRECTORY


class IntentRecognition:
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
    pass
