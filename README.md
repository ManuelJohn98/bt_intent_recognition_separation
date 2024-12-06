# Bachelor Thesis: Intent Recognition and Separation for E-DRZ

This is the accompanying code for the bachelor thesis _Intent Recognition and Separation for E-DRZ_.

## Requirements
This project relies on cuda toolkit version 12.4 to be installed already.
All other requirements can be installed via the _requirements.txt_ file.

## Guide for Setting Up
It is recommended to create a virtual environment, where all required packages can be installed:
```
python -m venv venv
```

For Windows:
```
.\venv\Scripts\activate
```

For Linux:
```
source venv/bin/activate
```

Then installing all required packages:
```
python pip install -r requirements.txt
```

## Usage Guide

There are three modes `-m` that can be chosen from: `basic`, `ablation`, `separated`. Each of these corresponds to an experiment that was conducted in the thesis. Before training a model, the data has to be preprocessed `--preprocess`. To that end, make sure that the raw data files are in the corresponding folder in the data folder. The raw data files should be text files in the WebAnno format. It also needs to be specified, what the data should be specified for: `--train` or `--cv`. The option `-s` enables the collection of some statistics during any part.
```
python intrec.py -m basic --preprocess --train -s
python intrec.py -m basic --train -s
```

The above code will preprocess the raw data and divide it into a randomly split training and testing dataset. In the second step the models that have been specified in the _config.py_ file are fine-tuned with the preprocessed dataset. To perform a cross validation (with a default of 5 folds) the commands look similar:
```
python intrec.py -m basic --preprocess --cv -s
python intrec.py -m basic --cv
```

Note: `python intrec.py --help` can be used to get more information on all command line arguments.
