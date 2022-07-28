# Note

* `data` is a directory where the *Primer* dataset is stored, alongside with physicochemical indices that are used to calculate some descriptors.
* `descriptors.py` is the file that contains the calculation of all descriptors for a given sequence.
* `calculate_features.py` is a script that calculates all descriptors for an entire dataset (with the help of `descriptors.py`) and creates a dataframe with all the descriptors.
* `utils.py` is a file that contains some useful functions.
* `validate_descriptors.ipynb` is a notebook that contains an example of how to calculate descriptors and perform **ML** classification for the *Primer* dataset. It uses the `calculate_features.py` script.

