import os
import json
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from data.datapreprocessing import prepare_for_cross_validation
from training import ModelTrainer
from inference import IntentRecognition, IntentRecognitionSeparation, IntentSeparation
from config import data_directory, load_config, cross_validation_directory
from utils.utils import get_train_eval_stats, get_last_checkpoint_dir, delete_models


def cross_validate_model(
    prefix: str, model_name: str, output_name: str, splits=5, sep=False
) -> dict:
    tracked_stats = {}
    for i in range(splits):
        # Rename metadata file
        os.rename(
            os.path.join(data_directory, f"{prefix}metadata.json"),
            os.path.join(
                data_directory,
                f"{str(i)}_{prefix}metadata.json",
            ),
        )

        mt = ModelTrainer(f"{str(i)}_{prefix}", model_name, sep)
        mt.train(output_name=output_name, num_train_epochs=30)

        # Track train and eval stats
        tracked_stats[str(i)] = {}
        tracked_stats[str(i)]["train"], tracked_stats[str(i)]["eval"] = (
            get_train_eval_stats(get_last_checkpoint_dir(model_name, output_name))
        )

        # Rename metadata file back
        os.rename(
            os.path.join(
                data_directory,
                f"{str(i)}_{prefix}metadata.json",
            ),
            os.path.join(data_directory, f"{prefix}metadata.json"),
        )

        delete_models(model_name)

    return tracked_stats


def cross_validate(splits=5, shuffle=True, ablation=False, sep=False) -> None:
    if sep and ablation:
        raise ValueError("sep and ablation cannot be True at the same time")
    models_list = load_config()["models"]
    tracked_stats = {}
    prefix = "ablation_" if ablation else "separated_" if sep else ""

    for model in models_list:
        tracked_stats = cross_validate_model(
            prefix, model, "intent_recognition_separation", splits, sep
        )
        cv = {}
        if os.path.exists(
            os.path.join(cross_validation_directory, f"{prefix}tracked_stats.json")
        ):
            with open(
                os.path.join(cross_validation_directory, f"{prefix}tracked_stats.json"),
                "r",
                encoding="utf8",
            ) as f:
                cv = json.load(f)
        cv[model + "_intent_recognition_separation"] = tracked_stats
        with open(
            os.path.join(cross_validation_directory, f"{prefix}tracked_stats.json"),
            "w",
            encoding="utf8",
        ) as f:
            json.dump(cv, f, ensure_ascii=False)


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

    def _calculate_f1(self, fp, fn, tp):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def per_class_f1(self):
        prefix = ""
        if self.mode == "IntentRecognition":
            prefix = "separated_"
        elif self.mode == "IntentSeparation":
            prefix = "ablation_"
        elif self.mode == "IntentRecognitionSeparation":
            prefix = ""
        data = {}
        with open(
            os.path.join(data_directory, f"{prefix}test_data.json"),
            "r",
            encoding="utf8",
        ) as f:
            data = json.load(f)
        y_true = []
        y_pred = []
        for row in data["data"]:
            y_true.append(
                list(map(lambda x: self.model.model.config.id2label[x], row["labels"]))
            )
            y_pred.append(
                list(
                    map(
                        lambda x: self.model.model.config.id2label[x],
                        self.model.predict(" ".join(row["tokens"])),
                    )
                )
            )
        mlb = MultiLabelBinarizer()
        y_true_bin = mlb.fit_transform(y_true)
        y_pred_bin = mlb.transform(y_pred)
        mcm = multilabel_confusion_matrix(y_true_bin, y_pred_bin)
        results = {}
        for i, label in enumerate(mlb.classes_):
            if any(mcm[i].ravel() == 0):
                results[label] = "N/A"
                continue
            _, fp, fn, tp = mcm[i].ravel()
            results[label] = self._calculate_f1(fp, fn, tp)
        return results
