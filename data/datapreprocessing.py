"""This file contains functionality for data preprocessing.
"""

import os
from collections import defaultdict
import json
import re
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from config import DATA_DIRECTORY, RAW_DATA_DIRECTORY, load_config, FREQUENCY_MAPPING
from stats.collectstatistics import StatisticsCollector
from utils.general import create_proxy_data


def _get_file_names_from_dir(ext: str = "tsv") -> list:
    """This function returns a list of all file names
    with the given extension in the current directory.
    """
    return [file for file in os.listdir(RAW_DATA_DIRECTORY) if file.endswith(ext)]


def _remove_count_from_label(label: str) -> str:
    """This function will return a label without any trailing number."""
    match = re.match(r"([A-Za-z_\/]+)", label)
    return match.group(1)


def _remove_label_from_count(label: str) -> str:
    """This function will return the count of a label."""
    match = re.search(r"\[(\d+)\]", label)
    return match.group(1) if match else ""


def _modify_label(label_with_count: str) -> str:
    """This function will modify the label according to the mapping in config.py.
    Returns new value for label with count.
    """
    label = _remove_count_from_label(label_with_count)
    count = _remove_label_from_count(label_with_count)
    new_label = FREQUENCY_MAPPING.get(label, label)
    return f"{new_label}[{count}]" if count else f"{new_label}"


def _preprocess_raw_data(file: str, prefix: str) -> dict:
    """This function takes a tsv file in the WebAnno
    format to have the data labeled in the BIO format.
    Returns a csv string.
    """
    stat = load_config()["statistics"]
    file = os.path.join(RAW_DATA_DIRECTORY, file)
    labeled_data = {}
    labeled_data["data"] = []
    separated_data_collection = defaultdict(lambda: defaultdict(list))
    bio_labels = set()
    labels = set()
    lexicon = set()
    ablation_goldlable_mapping = {}
    with open(file, "r", encoding="utf8") as f:
        current_turn_no = 0
        current_label = ""
        separated_label = ""
        separated_id = 0
        # Read WebAnno format
        for line_no, line in enumerate(f):
            labeled_data_point = {}
            if line.startswith("#") or line == "\n":
                separated_label = ""
                continue
            elements = line.split("\t")  # line is guaranteed to have 4 elements
            turn_id = elements[0]
            turn_no = int(turn_id.split("-")[0])
            token = elements[2]
            # get the lable which is prefix of last element
            label_with_count = elements[3].split("[")[0].replace("\\", "")
            if "modified" in prefix:
                label_with_count = _modify_label(label_with_count)
            # convert labels to BIO format
            # Unlabeled data point -> O (other)
            if label_with_count == "_":
                labeled_data_point["line_no"] = line_no
                labeled_data_point["turn_no"] = turn_no
                labeled_data_point["label"] = "O"
                labeled_data_point["token"] = token
                labeled_data_point["splitting"] = "I"
                bio_labels.add("O")
                if stat:
                    stats = StatisticsCollector()
                    stats.count_unlabeled("number_of_tokens")
                    lexicon.add(token)
                    if "ablation" in prefix:
                        stats.count_labeled("dataBIO", "I")
                    elif "separated_" in prefix:
                        pass
                        # if current_label != "":
                        #     stats.count_labeled("data", "other")
                    else:
                        stats.count_labeled("data", "other")
                        stats.count_labeled("dataBIO", "O")
                current_label = ""
                if separated_label != "":
                    separated_data_collection[separated_id][separated_label].append(
                        token
                    )
                labeled_data["data"].append(labeled_data_point)
                continue
            # Start of a new turn
            if current_turn_no != turn_no:
                current_turn_no = turn_no
                current_label = label_with_count
                # sometimes the label has a number at the end
                label = _remove_count_from_label(label_with_count)
                labeled_data_point["line_no"] = line_no
                labeled_data_point["turn_no"] = turn_no
                labeled_data_point["label"] = "B-" + label
                labeled_data_point["token"] = token
                labeled_data_point["splitting"] = "B"
                bio_labels.add("B-" + label)
                labels.add(label)
                if stat:
                    stats = StatisticsCollector()
                    stats.count_unlabeled("number_of_tokens")
                    lexicon.add(token)
                    if "ablation" in prefix:
                        stats.count_labeled("dataBIO", "B")
                    elif "separated_" in prefix:
                        stats.count_labeled("data", label)
                    else:
                        stats.count_labeled("data", label)
                        stats.count_labeled("dataBIO", f"B-{label}")
                separated_id += 1
                separated_data_collection[separated_id][label].append(token)
                separated_label = label
                ablation_goldlable_mapping[turn_id] = label
            # New label within same turn
            elif current_label != label_with_count:
                current_label = label_with_count
                label = _remove_count_from_label(label_with_count)
                labeled_data_point["line_no"] = line_no
                labeled_data_point["turn_no"] = turn_no
                labeled_data_point["label"] = "B-" + label
                labeled_data_point["token"] = token
                labeled_data_point["splitting"] = "B"
                bio_labels.add("B-" + label)
                labels.add(label)
                if stat:
                    stats = StatisticsCollector()
                    stats.count_unlabeled("number_of_tokens")
                    stats.count_unlabeled("need_for_separation")
                    lexicon.add(token)
                    if "ablation" in prefix:
                        stats.count_labeled("dataBIO", "B")
                    elif "separated_" in prefix:
                        stats.count_labeled("data", label)
                    else:
                        stats.count_labeled("data", label)
                        stats.count_labeled("dataBIO", f"B-{label}")
                separated_id += 1
                separated_data_collection[separated_id][label].append(token)
                separated_label = label
                ablation_goldlable_mapping[turn_id] = label
            # Label spans multiple tokens
            else:
                # only use I for these intermediary labels without actual label
                labeled_data_point["line_no"] = line_no
                labeled_data_point["turn_no"] = turn_no
                labeled_data_point["label"] = "I"
                labeled_data_point["token"] = token
                labeled_data_point["splitting"] = "I"
                bio_labels.add("I")
                labels.add(label)
                if stat:
                    stats = StatisticsCollector()
                    stats.count_unlabeled("number_of_tokens")
                    lexicon.add(token)
                    if "ablation" in prefix:
                        stats.count_labeled("dataBIO", "I")
                    elif "separated_" in prefix:
                        # stats.count_labeled("data", label)
                        pass
                    else:
                        stats.count_labeled("dataBIO", "I")
                        stats.count_labeled("data", label)
                separated_data_collection[separated_id][label].append(token)
            # add the labeled data point to labeled data
            labeled_data["data"].append(labeled_data_point)
    labeled_data["labels"] = bio_labels
    if stat:
        stats = StatisticsCollector()
        for _ in range(len(lexicon)):
            stats.count_unlabeled("number_of_unique_words")
    if "separated_" in prefix:
        separated_data = {}
        separated_data["data"] = [
            {
                "turn_no": idx,
                "label": label,
                "text": " ".join(tokens),
            }
            for idx in separated_data_collection
            for label, tokens in separated_data_collection[idx].items()
        ]
        separated_data["labels"] = labels
        return separated_data
    if "ablation_" in prefix:
        ablation_data = {}
        ablation_data["data"] = [
            {
                key if key != "splitting" else "label": val
                for key, val in data_point.items()
                if key != "label"
            }
            for data_point in labeled_data["data"]
        ]
        ablation_data["goldlable_mapping"] = ablation_goldlable_mapping
        ablation_data["labels"] = {"B", "I"}
        return ablation_data
    return labeled_data


def _transform_to_huggingface_format(
    labeled_data: list, labels: set, sep=False
) -> tuple:
    huggingface_format = {}
    metadata = {}
    huggingface_format["data"] = []
    labels = sorted(labels)
    metadata["id2label"] = {}
    metadata["label2id"] = {}
    for i, label in enumerate(labels):
        metadata["id2label"][i] = label
        metadata["label2id"][label] = i
    current_turn_no = 0
    data_id = 0
    for data_point in labeled_data:
        turn_no = data_point["turn_no"]
        if current_turn_no != turn_no:
            # Prepare for new turn
            current_turn_no = turn_no
            data_id += 1
            huggingface_format["data"].append({})
            huggingface_format["data"][-1]["id"] = data_id
            if not sep:
                huggingface_format["data"][-1]["labels"] = []

            if not sep:
                huggingface_format["data"][-1]["tokens"] = []

        if not sep:
            huggingface_format["data"][-1]["labels"].append(
                labels.index(data_point["label"])
            )
        else:
            huggingface_format["data"][-1]["label"] = labels.index(data_point["label"])

        if not sep:
            huggingface_format["data"][-1]["tokens"].append(data_point["token"])
        else:
            huggingface_format["data"][-1]["text"] = data_point["text"]
    metadata["num_rows"] = huggingface_format["data"][-1]["id"]
    if load_config()["statistics"]:
        stats = StatisticsCollector()
        stats.add_statistics("number_of_rows", metadata["num_rows"])
        if not sep and "number_of_tokens" in stats.__dict__:
            stats.add_statistics(
                "average_tokens_per_turn", stats.number_of_tokens / stats.number_of_rows
            )
    # huggingface_format["id2label"] = meta_data["id2label"]
    # huggingface_format["label2id"] = meta_data["label2id"]
    # huggingface_format["num_rows"] = meta_data["num_rows"]
    return huggingface_format, metadata


# def _concat_json(json1: dict, json2: dict) -> dict:
#     """This function concatenates two json formatted memory objects."""
#     return {**json1, **json2}


def convert_all_raw_data(prefix: str) -> None:
    """
    This function will convert all raw data in the data directory to a single json file.
    The file will be named {prefix}preprocessed_data.json.

    Args:
        prefix (str): The prefix to be used for the output
    """
    stat = load_config()["statistics"]
    labeled_data = []
    labels = set()
    for file in _get_file_names_from_dir():
        preprocessed_data = _preprocess_raw_data(file, prefix)
        labeled_data = labeled_data + preprocessed_data["data"]
        labels = labels.union(preprocessed_data["labels"])
    transformed_labeled_data, metadata = _transform_to_huggingface_format(
        labeled_data, labels, "separated_" in prefix
    )
    if "ablation_" in prefix:
        metadata["goldlable_mapping"] = preprocessed_data["goldlable_mapping"]
    with open(
        os.path.join(DATA_DIRECTORY, f"{prefix}preprocessed_data.json"),
        "w",
        encoding="utf8",
    ) as f:
        json.dump(transformed_labeled_data, f, ensure_ascii=False)
    with open(
        os.path.join(DATA_DIRECTORY, f"{prefix}metadata.json"), "w", encoding="utf8"
    ) as f:
        json.dump(metadata, f, ensure_ascii=False)
    if stat:
        stats = StatisticsCollector()
        stats.write_to_file(prefix)


def train_test_split_(prefix: str, test_size: 0.15, shuffle=True, seq=False) -> None:
    """
    This function will split the preprocessed data into training and testing data by
    using the train_test_split function from sklearn. The data will be saved in two
    separate files: {prefix}train_data.json and {prefix}test_data.json.

    Args:
        prefix (str): The prefix of the preprocessed data file
        test_size (float): The size of the test data
        shuffle (bool): Whether the data should be shuffled
        seq (bool): Whether the task is a sequence classification task or not

    Raises:
        ValueError: If the number of samples in the proxy_data arrays are inconsistent
    """
    seed = load_config()["seed"]
    data = {}
    with open(
        os.path.join(DATA_DIRECTORY, f"{prefix}preprocessed_data.json"),
        "r",
        encoding="utf8",
    ) as f:
        data = json.load(f)
    y = None
    if not seq:
        proxy_data = create_proxy_data(prefix, data["data"])
        if len(proxy_data[:, 0]) != len(proxy_data[:, 1]):
            raise ValueError("Inconsistent number of samples in proxy_data arrays")
        y = proxy_data
    else:
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(
            [[data["data"][i]["label"]] for i in range(len(data["data"]))]
        )
    train_data, test_data, _, _ = train_test_split(
        data["data"],
        y,
        test_size=test_size,
        shuffle=shuffle,
        stratify=proxy_data[:, 1],
        random_state=seed,
    )
    train_data = {"data": train_data}
    test_data = {"data": test_data}
    with open(
        os.path.join(DATA_DIRECTORY, f"{prefix}train_data.json"),
        "w",
        encoding="utf8",
    ) as f:
        json.dump(train_data, f, ensure_ascii=False)
    with open(
        os.path.join(DATA_DIRECTORY, f"{prefix}test_data.json"),
        "w",
        encoding="utf8",
    ) as f:
        json.dump(test_data, f, ensure_ascii=False)


def prepare_for_cross_validation(
    prefix: str, splits=5, shuffle=True, seq=False
) -> None:
    """
    This function will prepare the data for cross validation by using the StratifiedKFold
    from sklearn. The data will be split into training and testing data for each fold.
    The data will be saved in separate files:
    {fold}_{prefix}train_data.json and {fold}_{prefix}test_data.json.

    Args:
        prefix (str): The prefix of the preprocessed data file
        splits (int): The number of splits
        shuffle (bool): Whether the data should be shuffled
        seq (bool): Whether the task is a sequence classification task or not

    Raises:
        ValueError: If the index that is returned from StratifiedKFold is invalid
    """
    # Get seed
    seed = load_config()["seed"]
    # Generate proxy data
    preprocessed_data = {}
    with open(
        os.path.join(DATA_DIRECTORY, f"{prefix}preprocessed_data.json"),
        "r",
        encoding="utf8",
    ) as f:
        preprocessed_data = json.load(f)
    data = None
    if not seq:
        data = create_proxy_data(prefix, preprocessed_data["data"])
    else:
        data = preprocessed_data["data"]

    # Get splits
    sfk = StratifiedKFold(n_splits=splits, shuffle=shuffle, random_state=seed)
    splits_generator = None
    if not seq:
        splits_generator = sfk.split(data, data[:, 1])
    else:
        splits_generator = sfk.split(data, [data[i]["label"] for i in range(len(data))])

    # Cross validation
    for i, (train_idx, test_idx) in enumerate(splits_generator):
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
            os.path.join(
                DATA_DIRECTORY,
                f"{str(i)}_{prefix}train_data.json",
            ),
            "w",
            encoding="utf8",
        ) as f:
            json.dump(train_data, f, ensure_ascii=False)
        with open(
            os.path.join(
                DATA_DIRECTORY,
                f"{str(i)}_{prefix}test_data.json",
            ),
            "w",
            encoding="utf8",
        ) as f:
            json.dump(test_data, f, ensure_ascii=False)
