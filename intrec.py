#!bt_env/bin/python3
# intrec.py
"""This is the main file for the project.
It is concerned with handling the user input.
"""
import os
import sys
import json
from data.datapreprocessing import (
    convert_all_raw_data,
    prepare_for_training,
    prepare_for_cross_validation,
)
from config import (
    save_config,
    MODELS_DIRECTORY,
    CROSS_VALIDATION_DIRECTORY,
    MODELS,
)
from utils.utils import (
    delete_all_processed_data,
    delete_all_tracked_stats,
    check_for_splits,
    check_for_train_test,
)
from stats.collectstatistics import StatisticsCollector
from training import ModelTrainer
from inference import IntentRecognition, IntentRecognitionSeparation, IntentSeparation
from evaluation import cross_validate_model, Evaluator
from argparsing import parse_args


def main():
    """Main function of this project."""
    args = parse_args()
    shuffle = args.no_shuffle

    # create config with stat flag
    config = {}
    config["statistics"] = args.s
    config["seed"] = args.seed
    config["dataset_mode"] = args.dataset_mode
    save_config(config)
    shuffle = not args.no_shuffle

    prefix = ""
    if "normal" in args.mode:
        prefix = ""
    elif "ablation" in args.mode:
        prefix = "ablation_"
    elif "separated" in args.mode:
        prefix = "separated_"

    if "modified" in args.dataset_mode:
        prefix = "modified_" + prefix

    seq = False
    if "separated" in args.mode:
        seq = True
    if args.preprocessing:
        if args.clean:
            # delete all preprocessed data
            delete_all_processed_data()
        if not os.path.exists(
            os.path.join(MODELS_DIRECTORY, f"{prefix}processed_data")
        ):
            convert_all_raw_data(prefix)
        if args.train:
            if not os.path.exists(
                os.path.join(MODELS_DIRECTORY, f"{prefix}train_data")
            ) and not os.path.exists(
                os.path.join(MODELS_DIRECTORY, f"{prefix}test_data")
            ):
                prepare_for_training(prefix, args.test_size, shuffle, seq)
        elif args.cv:
            if not check_for_splits(prefix, args.folds):
                prepare_for_cross_validation(prefix, args.folds, shuffle, seq)
        sys.exit(0)
    if args.train:
        for model in MODELS:
            if check_for_train_test(prefix):
                mt = ModelTrainer(prefix, model, seq)
                mt.train(
                    args.output_name,
                    args.learning_rate,
                    args.train_batch_size,
                    args.eval_batch_size,
                    args.num_train_epochs,
                    args.weight_decay,
                )
                pass
            else:
                raise FileNotFoundError(
                    "Train and test data files are missing. Please preprocess the data first."
                )
            if args.s:
                stats = StatisticsCollector()
                stats.plot_checkpoints(f"{model}", prefix, args.output_name)
                inference = None
                if "ablation" in prefix:
                    inference = IntentSeparation(
                        prefix, model, args.output_name, "eval"
                    )
                elif "separated" in prefix:
                    inference = IntentRecognition(
                        prefix, model, args.output_name, "eval"
                    )
                else:
                    inference = IntentRecognitionSeparation(
                        prefix, model, args.output_name, "eval"
                    )
                evaluator = Evaluator(inference)
                per_class_f1 = evaluator.get_per_class_f1()
                for label, f1 in per_class_f1.items():
                    stats.add_statistics(
                        f"{model}_{prefix}{args.output_name}_{label}",
                        (float(f1) if not f1 == "N/A" else 0.0),
                    )
                stats.write_to_file(prefix)

    if args.cv:
        if args.clean:
            # delete all cross validation data
            delete_all_tracked_stats()
        tracked_stats = {}

        for model in MODELS:
            tracked_stats = cross_validate_model(
                prefix,
                model,
                args.output_name,
                args.folds,
                seq,
                args.learning_rate,
                args.train_batch_size,
                args.eval_batch_size,
                args.num_train_epochs,
                args.weight_decay,
            )
            cv = {}
            if os.path.exists(
                os.path.join(CROSS_VALIDATION_DIRECTORY, f"{prefix}tracked_stats.json")
            ):
                with open(
                    os.path.join(
                        CROSS_VALIDATION_DIRECTORY, f"{prefix}tracked_stats.json"
                    ),
                    "r",
                    encoding="utf8",
                ) as f:
                    cv = json.load(f)
            cv[model + f"_{args.output_name}"] = tracked_stats
            with open(
                os.path.join(CROSS_VALIDATION_DIRECTORY, f"{prefix}tracked_stats.json"),
                "w",
                encoding="utf8",
            ) as f:
                json.dump(cv, f, ensure_ascii=False)
        if args.s:
            stats = StatisticsCollector()
            stats.plot_cv(prefix)

    os.remove("config.yaml")


if __name__ == "__main__":
    main()
