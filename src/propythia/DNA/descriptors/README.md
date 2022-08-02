# Note

* `data` is a directory where the *Primer* dataset is stored (both in *CSV* and *FASTA* formats), alongside with physicochemical indices that are used to calculate some descriptors.
* `read_sequence.py` is the file that contains functions to read and validate DNA sequences. They can be read from a *CSV* file, a *FASTA* file, or from a single string.
* `descriptors.py` is the file that contains the calculation of all descriptors for a given sequence.
* `calculate_features.py` is a script that calculates all descriptors for an entire dataset (with the help of `descriptors.py`) and creates a dataframe with all the descriptors.
* `utils.py` is a file that contains some useful functions.
* `quick-start.ipynb` is a notebook that explains how to perform every step of the developed module. It includes the reading of sequences/datasets, calculation and normalization of descriptors, and the usage of the resulting dataframe to train several machine learning models.
