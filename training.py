import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)
import evaluate
import numpy as np
import json
from config import data_directory, model_directory, load_config


preprocessed_data = {}
with open(
    os.path.join(data_directory, "preprocessed_data.json"), "r", encoding="utf8"
) as f:
    preprocessed_data = json.load(f)


def train_test_split(filename: str, test_size=0.15) -> None:
    data = {}
    with open(filename, "r", encoding="utf8") as f:
        data = json.load(f)
    num_rows = data["num_rows"]
    split_index = int(round(num_rows * (1 - test_size)))
    _train_data = {}
    _train_data["data"] = []
    _test_data = {}
    _test_data["data"] = []
    for row in data["data"]:
        if row["id"] < split_index:
            _train_data["data"].append(row)
        else:
            _test_data["data"].append(row)
    filename = filename.removesuffix(".json")
    with open(
        os.path.join(data_directory, f"{filename}_train.json"), "w", encoding="utf8"
    ) as f:
        json.dump(_train_data, f)
    with open(
        os.path.join(data_directory, f"{filename}_test.json"), "w", encoding="utf8"
    ) as f:
        json.dump(_test_data, f)


train_test_split(os.path.join(data_directory, "preprocessed_data.json"))


dataset = load_dataset(
    "json",
    data_files={
        "train": os.path.join(data_directory, "preprocessed_data_train.json"),
        "test": os.path.join(data_directory, "preprocessed_data_test.json"),
    },
    field="data",
)

print(dataset)

pre_trained_model = load_config()["models"][0]
tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)


def tokenize_and_align_labels(data_row):
    tokenized_inputs = tokenizer(
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


tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer)
seqeval = evaluate.load("seqeval")
labels = preprocessed_data["label2id"].keys()


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


model = AutoModelForTokenClassification.from_pretrained(
    pre_trained_model,
    num_labels=len(labels),
    id2label=preprocessed_data["id2label"],
    label2id=preprocessed_data["label2id"],
)

training_args = TrainingArguments(
    output_dir=os.path.join(
        model_directory, f"{pre_trained_model}_intent_recognition_separation"
    ),
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    # eval_strategy="epoch",
    # load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
