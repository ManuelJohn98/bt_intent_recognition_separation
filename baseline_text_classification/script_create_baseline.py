"""This is a script that can be used within the repository of the TRADR model
project to create a baseline of we can already achieve in terms of text/sentence
classification.
"""

import json
import os
import subprocess


def convert_data(fold: int) -> None:
    for split in ["train", "test"]:
        data = {}
        with open(
            os.path.join(os.curdir, "data", f"{str(fold)}_separated_{split}_data.json"),
            "r",
            encoding="utf8",
        ) as f:
            data = json.load(f)
        metadata = {}
        with open(
            os.path.join(os.curdir, "data", "separated_metadata.json"),
            "r",
            encoding="utf8",
        ) as f:
            metadata = json.load(f)

        os.makedirs(os.path.join(os.curdir, "data", f"data_{str(fold)}"), exist_ok=True)

        with open(
            os.path.join(os.curdir, "data", f"data_{str(fold)}", f"{split}.tsv"),
            "w",
            encoding="utf8",
        ) as f:
            previous = ""
            f.write("speaker\ttokens\ttags\tprevious\n")
            for data_row in data["data"]:
                f.write(
                    f"no_speaker\t{data_row["text"]}\t{metadata["id2label"][str(data_row["label"])]}\t{previous}\n"
                )
                previous = data_row["text"]

        if split == "test":
            with open(
                os.path.join(os.curdir, "data", f"data_{str(fold)}", "dev.tsv"),
                "w",
                encoding="utf8",
            ) as f:
                previous = ""
                f.write("speaker\ttokens\ttags\tprevious\n")
                for data_row in data["data"]:
                    f.write(
                        f"no_speaker\t{data_row["text"]}\t{metadata["id2label"][str(data_row["label"])]}\t{previous}\n"
                    )
                    previous = data_row["text"]


for fold in range(5):
    convert_data(fold)
    subprocess.run(
        ["python", "finetune_tradr.py", f"--data_dir=data/data_{fold}", "--mode=only"],
        check=False,
        shell=True,
    )
    subprocess.run(
        [
            "python",
            "finetune_tradr.py",
            "--evaluation=True",
            f"--data_dir=data/data_{fold}",
            "--mode=only",
            f"--output_dir=outputs/8_{fold}",
        ],
        check=False,
        shell=True,
    )
