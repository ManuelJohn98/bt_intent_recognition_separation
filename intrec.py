#!bt_env/bin/python3
# intrec.py
"""This is the main file for the project.
It is concerned with handling the user input.
"""
import os
from data.datapreprocessing import (
    convert_all_raw_data,
    prepare_for_training,
    prepare_for_cross_validation,
)
from config import (
    data_directory,
    cross_validation_directory,
    save_config,
    models_directory,
)
from utils.utils import (
    delete_models,
    delete_all_processed_data,
    delete_all_tracked_stats,
)
from stats.collectstatistics import StatisticsCollector
from training import ModelTrainer
from evaluation import cross_validate
from argparsing import parse_args


def main():
    """Main function of this project."""
    args = parse_args()
    shuffle = args.no_shuffle

    # create config with stat flag
    config = {}
    config["statistics"] = args.s
    config["seed"] = args.seed
    config["models"] = [
        "ikim-uk-essen/geberta-base",
        # "flair/ner-german-large",
        # "aseifert/distilbert-base-german-cased-comma-derstandard",
        "dbmdz/bert-base-german-cased",
    ]
    save_config(config)
    shuffle = args.no_shuffle

    if "preprocess" in args.mode:
        if args.clean:
            # delete all preprocessed data
            delete_all_processed_data()
        if args.ablation:
            convert_all_raw_data("ablation_")
        elif args.separation:
            convert_all_raw_data("separated_")
        elif not args.ablation and not args.separation:
            convert_all_raw_data("")
        else:
            raise ValueError("Invalid combination of arguments")
    if "train" in args.mode:
        if args.clean:
            # delete all models
            delete_models(*config["models"])
            # delete all cross validation data
            delete_all_tracked_stats()
        prepare_for_training("", args.test_size, shuffle, args.separation)
        for model in config["models"]:
            if not os.path.exists(
                os.path.join(models_directory, f"{model}_{args.output_name}")
            ):
                mt = ModelTrainer("", model, args.separation)
                mt.train(
                    args.output_name,
                    args.learning_rate,
                    args.train_batch_size,
                    args.eval_batch_size,
                    args.num_train_epochs,
                    args.weight_decay,
                )
            if args.s:
                stats = StatisticsCollector()
                stats.plot_checkpoints(f"{model}", args.output_name)
    if "cv" in args.mode:
        if args.clean:
            # delete all models
            delete_models(*config["models"])
            # delete all cross validation data
            delete_all_tracked_stats()
        prepare_for_cross_validation(
            args.folds, shuffle, args.ablation, args.separation
        )
        cross_validate(args.folds, shuffle, args.ablation, args.separation)
        prefix = (
            "ablation_" if args.ablation else "separated_" if args.separation else ""
        )
        if args.s:
            stats = StatisticsCollector()
            stats.plot_cv(prefix)


if __name__ == "__main__":
    main()
