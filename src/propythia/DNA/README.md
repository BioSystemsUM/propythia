# Note

## Machine Learning Part

* `data` is where the physicochemical indices are stored, which are used to calculate some descriptors.
* `descriptors.py` is the file that contains the calculation of all descriptors for a given sequence.
* `calculate_features.py` is a script that calculates all descriptors for an entire dataset (with the help of `descriptors.py`) and creates a dataframe with all the descriptors.
* `notebooks/quick-start-ML.ipynb` is a notebook that explains how to perform every step of the developed modules. It includes data reading and validation, calculation of descriptors from sequences, descriptors processing and using processed descriptors to train ML models (already implemented in ProPythia).

## Deep Learning Part

* `deep_ml.py` runs a combination of set hyperparameters or performs hyperparameter tuning for the given model, feature mode, and data directory.
* `outputs` is a directory where the output of the hyperparameter tuning is stored. Only the filtered results with the score of each model is stored in the directory.
* `src` is a directory where the source code of the entire DL pipeline is stored.
* `essential_genes` is a directory where all the information about the essential genes is stored since it was needed a lot of data preprocessing to build the dataset.
* `config.json` is a file that contains the configuration of the entire DL pipeline.

## Both Parts

* `utils.py` is a file that contains some useful functions.
* `read_sequence.py` is the file that contains functions to read and validate DNA sequences. They can be read from a *CSV* file, a *FASTA* file, or from a single string.
