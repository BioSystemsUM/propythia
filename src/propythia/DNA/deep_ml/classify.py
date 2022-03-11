# PyTorch
import torch.nn as nn

# parsing data file
from seq_reader import load_data
from one_hot_rep import get_rep_mats, conv_labels
from utils import to_categorical

import numpy as np
np.random.seed(123)  # for reproducibility

X, y = load_data("./data/promoters.data.txt")   # sequences, labels
X = get_rep_mats(X)

for i in X:
    for idx, j in enumerate(i):
        i[idx] = j[0]

y = conv_labels(y, "promoter")      # convert to integer labels
X = np.asarray(X)       # work with np arrays
y = np.asarray(y)
X_train = X[0:90]
X_test = X[90:]
y_train = y[0:90]
y_test = y[90:]

# 2. Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 1, 55, 64)  # (90, 55, 64) --> (90, 1, 55, 64)
X_test = X_test.reshape(X_test.shape[0], 1, 55, 64)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# 3. Preprocess class labels; i.e. convert 1-dimensional class arrays to 3-dimensional class matrices
Y_train = to_categorical(y_train, 2)
Y_test = to_categorical(y_test, 2)

# 4. Define model architecture

# 5. Compile model

# 6. Fit model on training data

# 7. Evaluate model on test data
