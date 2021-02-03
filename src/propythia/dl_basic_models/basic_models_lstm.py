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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import LSTM, ConvLSTM2D, BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool1D, MaxPool3D,TimeDistributed
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input

tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()


def create_lstm_bilstm_simple(input_dim, number_classes,
                              optimizer='Adam',
                              bilstm=True,
                              lstm_layers=(128, 64),
                              activation='tanh',
                              recurrent_activation='sigmoid',
                              dropout_rate=(0.3,), recurrent_dropout_rate=(0.3,),
                              l1=1e-5, l2=1e-4,
                              dense_layers=(64,32),
                              dropout_rate_dense=(0.3,),
                              batchnormalization = (True,),
                              dense_activation="relu", loss_fun='binary_crossentropy', activation_fun='sigmoid'):

    if len([dropout_rate]) == 1:
        dropout_rate = list([dropout_rate] * len(lstm_layers))
    if len([recurrent_dropout_rate]) == 1:
        recurrent_dropout_rate = list([recurrent_dropout_rate] * len(lstm_layers))
    if len([dropout_rate_dense]) == 1:
        dropout_rate_dense = list([dropout_rate_dense] * len(dense_layers))
    if len([batchnormalization]) == 1:
        batchnormalization = list([batchnormalization] * len(dense_layers))


    last_lstm_layer = lstm_layers[-1]
    first_lstm_layer = lstm_layers[0]
    middle_lstm_layer = lstm_layers[1:-1]
    last_dense_layer = dense_layers[-1]

    middle_lstm_dropout=dropout_rate[1:-1]
    middle_lstm_recurrent_dropout=recurrent_dropout_rate[1:-1]
    print(middle_lstm_dropout)


    with strategy.scope():
        model = Sequential()
        model.add(Input(shape=(1, input_dim,), dtype='float32', name='main_input'))

        # # add initial dropout
        # if int(initial_dropout_value) > 0:
        #     model.add(Dropout(initial_dropout_value))

        if not bilstm:
            # add first lstm layers
            model.add(LSTM(units=first_lstm_layer, input_shape=(1, input_dim,), return_sequences=True,
                           activation=activation, recurrent_activation=recurrent_activation,
                           dropout=dropout_rate[0],
                           recurrent_dropout=recurrent_dropout_rate[0],
                           kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))

            # add other lstm layers
            for layer in range(len(middle_lstm_layer) - 1):
                model.add(LSTM(units=middle_lstm_layer[layer], return_sequences=True,
                               activation=activation, recurrent_activation=recurrent_activation,
                               dropout=middle_lstm_dropout[layer],
                               recurrent_dropout=middle_lstm_recurrent_dropout[layer],
                               kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            # add last lstm layer
            model.add(LSTM(units=last_lstm_layer, return_sequences=False,
                           activation=activation, recurrent_activation=recurrent_activation,
                           dropout=dropout_rate[-1],
                           recurrent_dropout=recurrent_dropout_rate[-1],
                           kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))

        elif bilstm:
            # add first bilstm layer
            model.add(Bidirectional(LSTM(units=first_lstm_layer, return_sequences=True, activation=activation,
                                         recurrent_activation=recurrent_activation,
                                         kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                                         dropout=dropout_rate[0], recurrent_dropout=recurrent_dropout_rate[0]),
                                    input_shape=(1, input_dim,)))

            # add other lstm layer
            for layer in range(len(middle_lstm_layer) - 1):
                model.add(Bidirectional(LSTM(units=middle_lstm_layer[layer], return_sequences=True,
                                             activation=activation, recurrent_activation=recurrent_activation,
                                             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                                             dropout=middle_lstm_dropout[layer], recurrent_dropout=middle_lstm_recurrent_dropout[layer])))
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


def create_lstm_embedding(number_classes,
                          optimizer='Adam',
                          input_dim_emb=21, output_dim=128, input_length=1000, mask_zero=True,
                          bilstm=True,
                          lstm_layers=(128, 64),
                          activation='tanh',
                          recurrent_activation='sigmoid',
                          dropout_rate=(0.3,), recurrent_dropout_rate=(0.3,),
                          l1=1e-5, l2=1e-4,
                          dense_layers=(64, 32),
                          dense_activation="relu",
                          dropout_rate_dense = (0.3,),
                          batchnormalization = (True,),
                          loss_fun='binary_crossentropy', activation_fun='sigmoid'):

    if len([dropout_rate]) == 1:
        dropout_rate = list([dropout_rate] * len(lstm_layers))
    print(dropout_rate)

    if len([recurrent_dropout_rate]) == 1:
        recurrent_dropout_rate = list([recurrent_dropout_rate] * len(lstm_layers))
    if len([dropout_rate_dense]) == 1:
        dropout_rate_dense = list([dropout_rate_dense] * len(dense_layers))
    print(dropout_rate_dense)
    if len([batchnormalization]) == 1:
        batchnormalization = list([batchnormalization] * len(dense_layers))


    last_lstm_layer = lstm_layers[-1]

    with strategy.scope():
        model = Sequential()
        # model.add(Input(shape=(input_dim,),dtype='float32', name='main_input'))
        # Add an Embedding layer expecting input vocab of size 1000, and
        model.add(
            Embedding(input_dim=input_dim_emb, output_dim=output_dim, input_length=input_length, mask_zero=mask_zero))
        # model.add(Flatten(data_format=None))
        # model.add(tf.keras.layers.Reshape((input_length,output_dim)))
        # print(model.output_shape) #None 100, 256
        last_layer = LSTM(units=last_lstm_layer, return_sequences=False,
                          activation=activation, recurrent_activation=recurrent_activation,
                          dropout=dropout_rate[-1],
                          recurrent_dropout=recurrent_dropout_rate[-1],
                          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))
        if not bilstm:
            # add lstm layers
            for layer in range(len(lstm_layers) - 1):
                model.add(LSTM(units=lstm_layers[layer], return_sequences=True,
                               activation=activation, recurrent_activation=recurrent_activation,
                               dropout=dropout_rate[layer],
                               recurrent_dropout=recurrent_dropout_rate[layer],
                               kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            # add last lstm layer
            model.add(last_layer)

        elif bilstm:
            # add other lstm layer
            for layer in range(len(lstm_layers) - 1):
                model.add(Bidirectional(LSTM(units=lstm_layers[layer], return_sequences=True,
                                             activation=activation, recurrent_activation=recurrent_activation,
                                             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                                             dropout=dropout_rate[layer], recurrent_dropout=recurrent_dropout_rate[layer])))
            # add last lstm layer
            model.add(Bidirectional(last_layer))

        # add denses
        for layer in range(len(dense_layers)):
            model.add(Dense(units=dense_layers[layer], activation=dense_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            if batchnormalization[layer]:
                model.add(BatchNormalization())
            model.add(Dropout(dropout_rate_dense[layer]))
        # Add Classification Dense, Compile model and make it ready for optimization
        # model binary
        model.add(Dense(number_classes, activation=activation_fun))
        model.compile(loss=loss_fun, optimizer=optimizer, metrics=['accuracy'])
        return model

#
# def create_conv_lstm(number_classes,
#                      maxlen,
#                      input_dim,
#                      optimizer='Adam',
#                      bilstm=True,
#                      filters=(128, 64),
#                      dense_layers=(64, 32),
#                      activation='tanh',
#                      recurrent_activation='sigmoid',
#                      dense_activation="relu",
#                      max_pooling=True,
#                      pool_size=2, strides_pool=1,
#                      data_format_pool='channels_first',
#                      padding='same',
#                      dropout_rate=0.3, recurrent_dropout_rate=0.3,
#                      l1=1e-5, l2=1e-4,
#                      final_dropout_value=0.0):
#     # https://www.kaggle.com/kcostya/convlstm-convolutional-lstm-network-tutorial
#     # https://medium.com/neuronio-br/uma-introdu%C3%A7%C3%A3o-a-convlstm-c14abf9ea84a
#     # https://www.nature.com/articles/s41598-019-46850-0
#     # todo so tem conv2D , mas ate faz sentido. ver se no cnn_lstm tb n fazia diferenca fazer c 2D. ou o stride. pq s o stride for 20 n tiramos nada de la, pq vai sempre buscar o numero 1 q é  o mais alto
#     # ver papaers q usem cnn stm ou convlstm c sequencias proteicas paratentar perceber melhor isto
#     # todo perceber melhor e melhorar a aquitectura do cnn lstm
#     # http://colah.github.io/posts/2015-08-Understanding-LSTMs/
#     # https://keras.io/api/layers/recurrent_layers/time_distributed/
#
#     # https://github.com/keras-team/keras/issues/6150
#     # https://keras.io/api/layers/recurrent_layers/conv_lstm2d/
#     # https://medium.com/neuronio/an-introduction-to-convlstm-55c9025563a7
#
#     last_lstm_layer = filters[-1]
#     last_dense_layer = dense_layers[-1]
#     # maxlen=x_train.shape[1]
#     # input_dim=x_train.shape[2]
#     # trailer_input  = Input(shape=(frames, channels, pixels_x, pixels_y)
#     with strategy.scope():
#         model = Sequential()
#         model.add(Input(shape=(1,input_dim, 1, 1)))
#         # If data_format='channels_first' 5D tensor with shape: (samples, time, channels, rows, cols)
#         # If data_format='channels_last' 5D tensor with shape: (samples, time, rows, cols, channels)
#         print(model.input_shape)
#         for layer in range(len(filters) - 1):
#             model.add(ConvLSTM2D(filters=filters[layer],
#                                  data_format='channels_last',
#                                  kernel_size=(1, 1),
#                                  strides=(1, 1),
#                                  padding='same',
#                                  dilation_rate=(1, 1),
#                                  activation='tanh',
#                                  recurrent_activation='hard_sigmoid',
#                                  use_bias=True,
#                                  kernel_initializer='glorot_uniform',
#                                  recurrent_initializer='orthogonal',
#                                  bias_initializer='zeros',
#                                  unit_forget_bias=True,
#                                  kernel_regularizer=None,
#                                  recurrent_regularizer=None,
#                                  bias_regularizer=None,
#                                  activity_regularizer=None,
#                                  kernel_constraint=None,
#                                  recurrent_constraint=None,
#                                  bias_constraint=None,
#                                  return_sequences=True,
#                                  go_backwards=False,
#                                  stateful=False,
#                                  dropout=0.0,
#                                  recurrent_dropout=0.0))
#             model.add(BatchNormalization())
#             model.add(MaxPool3D(pool_size=pool_size, strides=strides_pool, padding=padding,
#                                              data_format=data_format_pool))
#         model.add(TimeDistributed(Flatten()))
#
#         model.add(TimeDistributed(Dense(512,)))
#         model.add(TimeDistributed(Dense(32,)))
#         model.add(TimeDistributed(Dense(last_dense_layer)))
#         # model.add(Flatten())
#         # # add final dense
#         # for layer in range(len(dense_layers) - 1):
#         #     model.add(Dense(units=dense_layers[layer], activation=dense_activation,
#         #                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
#         #     model.add(Dropout(dropout_rate))
#         #
#         # # add last layer
#         # model.add(Dense(units=last_dense_layer, activation=dense_activation,
#         #                 kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
#         # # add final dropout
#         # if final_dropout_value > 0: model.add(Dropout(final_dropout_value))
#
#         # Add Classification Dense, Compile model and make it ready for optimization
#         # model binary
#         # if not number_classes > 2:
#         #     model.add(Dense(1, activation='sigmoid'))
#         #     model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#         #
#         # # multiclass model
#         # elif number_classes > 2:
#         #     print('multiclass')
#         #     model.add(Dense(number_classes, activation='softmax'))
#         #     model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#         model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#         return model
#
# # model.add(Dense(y_train.shape[1], activation='sigmoid'))





# units=units_lstm, input_shape=(1, 2003),return_sequences=True, activation='tanh', recurrent_activation='sigmoid',
# use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
# bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None,
# bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
# bias_constraint=None, dropout=dropout_rate, recurrent_dropout=0.5, implementation=2, return_state=False,
# go_backwards=False, stateful=False, unroll=False