import os
import json
from collections import Counter
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from training import ModelTrainer
from inference import IntentRecognition, IntentRecognitionSeparation, IntentSeparation
from config import DATA_DIRECTORY
from utils.utils import get_train_eval_stats, get_last_checkpoint_dir


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


# def cross_validate(splits=5, shuffle=True, ablation=False, sep=False) -> None:
#     if sep and ablation:
#         raise ValueError("sep and ablation cannot be True at the same time")
#     models_list = load_config()["models"]
#     tracked_stats = {}
#     prefix = "ablation_" if ablation else "separated_" if sep else ""

#     for model in models_list:
#         tracked_stats = cross_validate_model(
#             prefix, model, "intent_recognition_separation", splits, sep
#         )
#         cv = {}
#         if os.path.exists(
#             os.path.join(CROSS_VALIDATION_DIRECTORY, f"{prefix}tracked_stats.json")
#         ):
#             with open(
#                 os.path.join(CROSS_VALIDATION_DIRECTORY, f"{prefix}tracked_stats.json"),
#                 "r",
#                 encoding="utf8",
#             ) as f:
#                 cv = json.load(f)
#         cv[model + "_intent_recognition_separation"] = tracked_stats
#         with open(
#             os.path.join(CROSS_VALIDATION_DIRECTORY, f"{prefix}tracked_stats.json"),
#             "w",
#             encoding="utf8",
#         ) as f:
#             json.dump(cv, f, ensure_ascii=False)


class Evaluator:
    def __init__(self, classifier):
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

    def get_per_class_f1(self):
        y_true, y_pred = self._get_y_true_y_pred()
        # mlb = MultiLabelBinarizer()
        # ohe = OneHotEncoder()
        y_true = list(
            map(
                lambda x: (
                    [x["label"]] if self.mode == "IntentRecognition" else x["labels"]
                ),
                y_true,
            )
        )
        # y_true_bin = mlb.fit_transform(y_true)
        # y_pred_bin = mlb.transform(y_pred)
        # y_true_ohe = [ohe.fit_transform([x]) for x in y_true]
        # y_pred_ohe = [ohe.transform([x]) for x in y_pred]
        # # mcm = multilabel_confusion_matrix(y_true_bin, y_pred_bin, samplewise=True)
        # mcm = multilabel_confusion_matrix(y_true_ohe, y_pred_ohe)
        # results = {}
        # for i, label in enumerate(ohe.categories_[0]):
        #     # if all(mcm[i].ravel() == 0):
        #     #     results[label] = "N/A"
        #     #     continue
        #     _, fp, fn, tp = mcm[i].ravel()
        #     results[label] = self._calculate_f1(fp, fn, tp)
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
