import pandas as pd
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
# import logging
# logging.disable(logging.WARNING)
# logging.getLogger("tensorflow").setLevel(logging.FATAL)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, ConvLSTM2D, BatchNormalization

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input

tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()
# todo por isto como deve ser
# https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2
# https://machinelearningmastery.com/cnn-long-short-term-memory-networks/
# https://missinglink.ai/guides/keras/keras-conv1d-working-1d-convolutional-neural-networks-keras/
#
def create_cnn_lstm(input_dim, number_classes,
                    optimizer='Adam',
                    filter_count=(32, 64, 128),
                    padding='same',
                    strides=1,
                    kernel_size=(3,),
                    cnn_activation=None,
                    kernel_initializer='glorot_uniform',
                    dropout_cnn=(0.0,),
                    max_pooling=(True,),
                    pool_size=(2,), strides_pool=1,
                    data_format_pool='channels_first',
                    bilstm=True,
                    lstm_layers=(128, 64),
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    dropout_rate=(0.3,), recurrent_dropout_rate=(0.3,),
                    l1=1e-5, l2=1e-4,
                    dense_layers=(64, 32),
                    dense_activation="relu",
                    dropout_rate_dense=(0.0,),
                    batchnormalization=True,
                    loss_fun='binary_crossentropy', activation_fun='sigmoid'):
    # kernel_initializer='lecun_uniform', activation='relu',

    last_lstm_layer = lstm_layers[-1]
    last_dense_layer = dense_layers[-1]
    # if kernel_size is not list: kernel_size = list([kernel_size]*len(filter_count))
    # if dropout_cnn is not list: dropout_cnn = list([dropout_cnn] * len(filter_count))
    # if max_pooling is not list: max_pooling = list([max_pooling] * len(filter_count))
    # if pool_size is not list: pool_size = list([pool_size] * len(filter_count))
    if len([kernel_size]) == 1:
        kernel_size = list(kernel_size * len(filter_count))
    if len([dropout_cnn]) ==1 : dropout_cnn = list(dropout_cnn * len(filter_count))
    if len([max_pooling]) ==1 : max_pooling = list(max_pooling * len(filter_count))
    if len([pool_size])==1 : pool_size = list(pool_size * len(filter_count))
    if len([dropout_rate]) ==1: dropout_rate = list(dropout_rate * len(lstm_layers))
    # if len([dropout_rate]) == 1:
    #     dropout_rate = list([dropout_rate] * len(lstm_layers))
    if len([recurrent_dropout_rate]) == 1:
        recurrent_dropout_rate = list(recurrent_dropout_rate * len(lstm_layers))
    if len([dropout_rate_dense]) == 1:
        dropout_rate_dense = list(dropout_rate_dense * len(dense_layers))
    if len([batchnormalization]) == 1:
        batchnormalization = list(batchnormalization * len(dense_layers))

    with strategy.scope():
        model = Sequential()
        n_timesteps, n_features = 1, input_dim
        model.add(Input(shape=(n_timesteps,n_features)))
        # model.add(Input(shape=(input_dim,1), dtype='float32', name='main_input'))

        # add convolutional layers
        for cnn in range(len(filter_count)):
            model.add((Conv1D(
                filters=filter_count[cnn],
                kernel_size=kernel_size[cnn],
                strides=strides,
                padding=padding,
                data_format='channels_last',
                activation=cnn_activation,
                dilation_rate=1, use_bias=True,
                kernel_initializer=kernel_initializer, bias_initializer='zeros',
                kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                kernel_constraint=None, bias_constraint=None)))

            max_pool = max_pooling[cnn]
            if max_pool:
                model.add(MaxPool1D(pool_size=pool_size[cnn], strides=strides_pool, padding=padding,
                                    data_format=data_format_pool))
            if dropout_cnn[cnn] > 0:
                model.add(Dropout(dropout_cnn[cnn]))

        # model.add(Flatten())

        # add lstm layers
        if not bilstm:
            # add lstm layers
            for layer in range(len(lstm_layers) - 1):
                model.add(LSTM(units=lstm_layers[layer], return_sequences=True,
                               activation=activation, recurrent_activation=recurrent_activation,
                               dropout=dropout_rate[layer],
                               recurrent_dropout=recurrent_dropout_rate[layer],
                               kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            # add last lstm layer
            model.add(LSTM(units=last_lstm_layer, return_sequences=False,
                           activation=activation, recurrent_activation=recurrent_activation,
                           dropout=dropout_rate[-1],
                           recurrent_dropout=recurrent_dropout_rate[-1],
                           kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))

        elif bilstm:
            # add other lstm layer
            for layer in range(len(lstm_layers) - 1):
                model.add(Bidirectional(LSTM(units=lstm_layers[layer], return_sequences=True,
                                             activation=activation, recurrent_activation=recurrent_activation,
                                             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                                             dropout=dropout_rate[layer], recurrent_dropout=recurrent_dropout_rate[layer])))
            # add last lstm layer
            model.add(Bidirectional(LSTM(units=last_lstm_layer, return_sequences=False,
                                         activation=activation, recurrent_activation=recurrent_activation,
                                         kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                                         dropout=dropout_rate[-1], recurrent_dropout=recurrent_dropout_rate[-1])))

        # add denses
        for layer in range(len(dense_layers)):
            model.add(Dense(units=dense_layers[layer], activation=dense_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            if batchnormalization[layer]:
                model.add(BatchNormalization())
                model.add(Dropout(dropout_rate_dense[layer]))

        # Add Classification Dense, Compile model and make it ready for optimization
        model.add(Dense(number_classes, activation=activation_fun))
        model.compile(loss=loss_fun, optimizer=optimizer, metrics=['accuracy'])
        return model
