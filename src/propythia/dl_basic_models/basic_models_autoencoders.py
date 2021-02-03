import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
# import logging
# logging.disable(logging.WARNING)
# logging.getLogger("tensorflow").setLevel(logging.FATAL)
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, TimeDistributed, Bidirectional
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import regularizers, Input
from tensorflow.keras.layers import BatchNormalization, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

# from src.mlmodels.utils import timer, summary_loss, saveModel, loadDNN, reportinf, validation, validation_multiclass
from tensorflow.keras.utils import plot_model
from tensorflow import keras
from operator import itemgetter

tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()


############################################################################################
############################################################################################


def create_dnn_simple(input_dim, number_classes,
                      hidden_layers=(128, 64, 32),
                      optimizer='Adam',
                      initial_dropout_value=0.0,
                      dropout_rate=0.3,
                      batchnormalization=True, activation="relu",
                      l1=1e-5, l2=1e-4,
                      final_dropout_value=0.2):
    last_layer = hidden_layers[-1]
    with strategy.scope():
        model = Sequential()
        model.add(Input(shape=(input_dim,)))



        # https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798