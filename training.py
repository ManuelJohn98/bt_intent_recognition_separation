import os
import json
import concurrent.futures
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)
import evaluate
import numpy as np
from sklearn.model_selection import StratifiedKFold
from config import (
    data_directory,
    models_directory,
    cross_validation_directory,
    load_config,
)
from stats.collectstatistics import StatisticsCollector
from utils.utils import (
    create_proxy_data,
    get_last_checkpoint_dir,
    get_train_eval_stats,
    delete_models,
)


def _shuffle_data(preprocessed_data: dict) -> dict:
    data = preprocessed_data["data"]
    np.random.shuffle(data)
    preprocessed_data["data"] = data
    return preprocessed_data


def _train_test_split(filename: str, test_size=0.15, shuffle=True, fold="") -> None:
    data = {}
    with open(filename, "r", encoding="utf8") as f:
        data = json.load(f)
    if shuffle:
        data = _shuffle_data(data)
    meta_data = {}
    with open(
        os.path.join(data_directory, "preprocessed_data.json"), "r", encoding="utf8"
    ) as f:
        meta_data = json.load(f)
    num_rows = meta_data["num_rows"]
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
        json.dump(_train_data, f, ensure_ascii=False)
    with open(
        os.path.join(data_directory, f"{filename}_test.json"), "w", encoding="utf8"
    ) as f:
        json.dump(_test_data, f, ensure_ascii=False)


def train_model(filename: str, model_name: str, test_size=0.15) -> None:
    # Load dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(data_directory, "preprocessed_data_train.json"),
            "test": os.path.join(data_directory, "preprocessed_data_test.json"),
        },
        field="data",
    )

    # Load metadata
    meta_data = {}
    with open(
        os.path.join(data_directory, "preprocessed_data.json"), "r", encoding="utf8"
    ) as f:
        meta_data = json.load(f)

    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    labels_list = list(meta_data["label2id"].keys())

    def compute_metrics(p):
        predictions, labels = p
        # Convert predictions to label IDs
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [labels_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [labels_list[l] for (p, l) in zip(prediction, label) if l != -100]
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
        model_name,
        num_labels=len(labels_list),
        id2label=meta_data["id2label"],
        label2id=meta_data["label2id"],
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(
            models_directory, f"{model_name}_intent_recognition_separation"
        ),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=30,  # 40
        weight_decay=0.01,
        logging_strategy="steps",
        logging_steps=150,  # 250
        eval_strategy="steps",
        eval_steps=150,  # 250
        # save_total_limit=1,
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
    model.save_pretrained(
        os.path.join(
            models_directory,
            f"{model_name}_intent_recognition_separation",
            "final_model",
        )
    )


def train(filename: str, test_size=0.15) -> None:
    # Set seed for reproducibility
    np.random.seed(load_config()["seed"])

    # preprocessed_data = {}
    # with open(
    #     os.path.join(data_directory, "preprocessed_data.json"), "r", encoding="utf8"
    # ) as f:
    #     preprocessed_data = json.load(f)

    # Split data into train and test
    _train_test_split(filename, test_size)

    # train each model in separate thread
    models_list = load_config()["models"]
    # pool = concurrent.futures.ThreadPoolExecutor()
    # for model in models_list:
    #     pool.submit(train_model, filename, model, test_size)
    # pool.shutdown(wait=True)
    for model in models_list:
        if not os.path.exists(
            os.path.join(models_directory, f"{model}_intent_recognition_separation")
        ):
            train_model(filename, model, test_size)
        if load_config()["statistics"]:
            stats = StatisticsCollector()
            stats.plot_checkpoints(f"{model}_intent_recognition_separation")


def cross_validate_model(model_name: str, splits=5, shuffle=True) -> None:
    # Get seed
    seed = load_config()["seed"]
    tracked_stats = {}
    # Generate proxy data
    preprocessed_data = {}
    with open(
        os.path.join(data_directory, "preprocessed_data.json"), "r", encoding="utf8"
    ) as f:
        preprocessed_data = json.load(f)
    data = create_proxy_data(preprocessed_data["data"])

    # Get splits
    sfk = StratifiedKFold(n_splits=splits, shuffle=shuffle, random_state=seed)
    splits = sfk.split(data, data[:, 1])

    # cross validation
    for i, (train_idx, test_idx) in enumerate(splits):
        train_data = {"data": []}
        test_data = {"data": []}
        for line in preprocessed_data["data"]:
            # Turn ids start at 1
            if line["id"] - 1 in train_idx:
                train_data["data"].append(line)
            elif line["id"] - 1 in test_idx:
                test_data["data"].append(line)
            else:
                raise ValueError("Invalid index")

        with open(
            os.path.join(data_directory, "preprocessed_data_train.json"),
            "w",
            encoding="utf8",
        ) as f:
            json.dump(train_data, f, ensure_ascii=False)
        with open(
            os.path.join(data_directory, "preprocessed_data_test.json"),
            "w",
            encoding="utf8",
        ) as f:
            json.dump(test_data, f, ensure_ascii=False)

        train_model("preprocessed_data", model_name)

        # Track train and eval stats
        tracked_stats[str(i)] = {}
        tracked_stats[str(i)]["train"], tracked_stats[str(i)]["eval"] = (
            get_train_eval_stats(
                get_last_checkpoint_dir(f"{model_name}_intent_recognition_separation")
            )
        )

        delete_models(model_name)

    return tracked_stats


def cross_validate(splits=5, shuffle=True) -> None:
    models_list = load_config()["models"]
    tracked_stats = {}
    for model in models_list:
        tracked_stats[model] = cross_validate_model(model, splits, shuffle)
        cv = {}
        with open(
            os.path.join(cross_validation_directory, "tracked_stats.json"),
            "r",
            encoding="utf8",
        ) as f:
            cv = json.load(f)
        cv[model] = tracked_stats[model]
        with open(
            os.path.join(cross_validation_directory, "tracked_stats.json"),
            "w",
            encoding="utf8",
        ) as f:
            json.dump(cv, f, ensure_ascii=False)
