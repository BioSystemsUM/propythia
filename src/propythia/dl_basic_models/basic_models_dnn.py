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
                      dropout_rate=(0.3,),
                      batchnormalization=(True,), activation="relu",
                      l1=1e-5, l2=1e-4, loss_fun='binary_crossentropy', activation_fun='sigmoid'):
    last_layer = hidden_layers[-1]
    # todo if give dropout rate not equal to layers, add?
    if len([dropout_rate]) == 1:
        dropout_rate = list(dropout_rate * len(hidden_layers))
    print(dropout_rate)
    if len([batchnormalization]) == 1:
        batchnormalization = list(batchnormalization * len(hidden_layers))

    with strategy.scope():
        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        if initial_dropout_value > 0:
            model.add(Dropout(initial_dropout_value))

        for layer in range(len(hidden_layers)):
            model.add(Dense(units=hidden_layers[layer], activation=activation,
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            if batchnormalization[layer]:
                model.add(BatchNormalization())
            model.add(Dropout(dropout_rate[layer]))

        # # last layer
        # model.add(Dense(units=last_layer, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
        # if batchnormalization: model.add(BatchNormalization())
        # if final_dropout_value > 0: model.add(Dropout(final_dropout_value))

        # Add Classification Dense, Compile model and make it ready for optimization
        model.add(Dense(number_classes, activation=activation_fun))
        model.compile(loss=loss_fun, optimizer=optimizer, metrics=['accuracy'])

        return model


def create_dnn_embedding(input_dim, number_classes,
                         hidden_layers=(128, 64),
                         optimizer='Adam',
                         input_dim_emb=21, output_dim=256, input_length=1000, mask_zero=True,
                         dropout_rate=(0.3,),
                         batchnormalization=True, activation="relu",
                         l1=1e-5, l2=1e-4,
                         loss_fun='binary_crossentropy', activation_fun='sigmoid'):
    if len([dropout_rate]) == 1:
        dropout_rate = list(dropout_rate * len(hidden_layers))
    if len([batchnormalization]) == 1:
        batchnormalization = list(batchnormalization * len(hidden_layers))

    with strategy.scope():
        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        model.add(
            Embedding(input_dim=input_dim_emb, output_dim=output_dim, input_length=input_length, mask_zero=mask_zero))
        model.add(Flatten(data_format=None))

        for layer in range(len(hidden_layers)):
            model.add(Dense(units=hidden_layers[layer], activation=activation,
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            if batchnormalization[layer]:
                model.add(BatchNormalization())
            model.add(Dropout(dropout_rate[layer]))

        # Add Classification Dense, Compile model and make it ready for optimization
        model.add(Dense(number_classes, activation=activation_fun))
        model.compile(loss=loss_fun, optimizer=optimizer, metrics=['accuracy'])

        return model
