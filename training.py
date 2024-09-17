import os
from datasets import load_dataset
from config import data_directory

dataset = load_dataset(
    "json", data_files=os.path.join(data_directory, "processed_data.json"), field="data"
)
print(dataset)
