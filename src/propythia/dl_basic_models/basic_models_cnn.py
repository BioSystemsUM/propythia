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
from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool1D, MaxPool2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input

tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()


# todo por isto como deve ser
# https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2
# https://missinglink.ai/guides/keras/keras-conv1d-working-1d-convolutional-neural-networks-keras/
# https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2
# https://towardsdatascience.com/simple-introduction-to-convolutional-neural-networks-cdf8d3077bac


# 1D networks allow you to use larger filter sizes.
# In a 1D network, a filter of size 7 or 9 contains only 7 or 9 feature vectors.
# Whereas in a 2D CNN, a filter of size 7 will contain 49 feature vectors.
# Can use larger convolution windows with 1D CNNs.
# With a 2D convolution layer, a 3 × 3 convolution window contains 3 × 3 = 9 feature vectors.
# With 1D convolution layer, a window of size 3 contains only 3 feature vectors.
# You can thus easily afford 1D convolution windows of size 7 or 9.

# todo explain really well this list system of mounting a cnn
def create_cnn_1D(input_dim, number_classes,
                  optimizer='Adam',
                  filter_count=(32, 64, 128),  # define number layers
                  padding='same',
                  strides=1,
                  kernel_size=(3,),  # list of kernel sizes per layer. if number will be the same in all numbers
                  cnn_activation='relu',
                  kernel_initializer='glorot_uniform',
                  dropout_cnn=(0.0, 0.2, 0.2),
                  # list of dropout per cnn layer. if number will be the same in all numbers
                  max_pooling=(True,),
                  pool_size=(2,), strides_pool=1,
                  data_format_pool='channels_first',
                  dense_layers=(64, 32),
                  dense_activation="relu",
                  dropout_rate=(0.3,),
                  l1=1e-5, l2=1e-4,
                  loss_fun='binary_crossentropy', activation_fun='sigmoid'):
    # kernel_initializer='lecun_uniform', activation='relu',

    last_dense_layer = dense_layers[-1]
    # todo list !!! they are tuples receive ints in dropouts
    if len(kernel_size) == 1:
        kernel_size = list(kernel_size * len(filter_count))
    if len(dropout_cnn) ==1 : dropout_cnn = list(dropout_cnn * len(filter_count))
    if len(max_pooling) ==1 : max_pooling = list(max_pooling * len(filter_count))
    if len(pool_size)==1 : pool_size = list(pool_size * len(filter_count))
    if len(dropout_rate) ==1: dropout_rate = list(dropout_rate * len(dense_layers))

    with strategy.scope():
        model = Sequential()
        n_timesteps, n_features = 1, input_dim
        model.add(Input(shape=(n_timesteps,n_features)))

        # todo embedding?
        # model.add(layers.Embedding(max_features, 128, input_length=max_len))
        # https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
        # add convolutional layers
        for cnn in range(len(filter_count)):
            model.add(Conv1D(
                filters=filter_count[cnn],
                kernel_size=kernel_size[cnn],
                strides=strides,
                padding=padding,
                data_format='channels_last',
                activation=cnn_activation,
                dilation_rate=1, use_bias=True,
                kernel_initializer=kernel_initializer, bias_initializer='zeros',
                kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                kernel_constraint=None, bias_constraint=None))

            max_pool = max_pooling[cnn]
            if max_pool:
                model.add(MaxPool1D(pool_size=pool_size[cnn], strides=strides_pool, padding=padding,
                                    data_format=data_format_pool))

            if dropout_cnn[cnn] > 0:
                model.add(Dropout(dropout_cnn[cnn]))

        # Flatten
        model.add(Flatten())
        # add dense layers
        for layer in range(len(dense_layers)):
            model.add(Dense(units=dense_layers[layer], activation=dense_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            model.add(Dropout(dropout_rate[layer]))

        # Add Classification Dense, Compile model and make it ready for optimization
        model.add(Dense(number_classes, activation=activation_fun))
        model.compile(loss=loss_fun, optimizer=optimizer, metrics=['accuracy'])
        return model

# todo explain really well this list system of mounting a cnn
def create_cnn_2D(input_dim, number_classes,
                  optimizer='Adam',
                  filter_count=(32, 64, 128),  # define number layers
                  padding='same',
                  strides=1,
                  kernel_size=((3,3),),  # list of kernel sizes per layer. if number will be the same in all numbers
                  cnn_activation='relu',
                  kernel_initializer='glorot_uniform',
                  dropout_cnn=(0.0, 0.2, 0.2),
                  # list of dropout per cnn layer. if number will be the same in all numbers
                  max_pooling=True,
                  pool_size=((2,2),), strides_pool=1,
                  data_format_pool='channels_last',
                  dense_layers=(64, 32),
                  dense_activation="relu",
                  dropout_rate=0.3,
                  l1=1e-5, l2=1e-4,
                  loss_fun='binary_crossentropy', activation_fun='sigmoid'):
    # kernel_initializer='lecun_uniform', activation='relu',

    last_dense_layer = dense_layers[-1]
    if len(kernel_size) == 1:
        kernel_size = list(kernel_size * len(filter_count))
    if len(dropout_cnn) ==1 : dropout_cnn = list(dropout_cnn * len(filter_count))
    if len(max_pooling) ==1 : max_pooling = list(max_pooling * len(filter_count))
    if len(pool_size)==1 : pool_size = list(pool_size * len(filter_count))
    if len(dropout_rate) ==1: dropout_rate = list(dropout_rate * len(dense_layers))
    print(dropout_rate)
    with strategy.scope():
        model = Sequential()

        # input shape = batch size + (image * image, channels)
        model.add(Input(shape=(input_dim)))

        # model.add(Input(shape=(input_dim,), dtype='float32', name='main_input'))

        # add convolutional layers
        for cnn in range(len(filter_count)):
            model.add(Conv2D(
                filters=filter_count[cnn],
                kernel_size=kernel_size[cnn],
                strides=strides,
                padding=padding,
                data_format='channels_last',
                activation=cnn_activation,
                dilation_rate=1, use_bias=True,
                kernel_initializer=kernel_initializer, bias_initializer='zeros',
                kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                kernel_constraint=None, bias_constraint=None))

            max_pool = max_pooling[cnn]
            if max_pool:
                model.add(MaxPool2D(pool_size=pool_size[cnn], strides=strides_pool, padding=padding,
                                    data_format=data_format_pool))
            if dropout_cnn[cnn] > 0:
                model.add(Dropout(dropout_cnn[cnn]))

        # Flatten
        model.add(Flatten())
        # add dense layers
        for layer in range(len(dense_layers)):
            model.add(Dense(units=dense_layers[layer], activation=dense_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            model.add(Dropout(dropout_rate[layer]))


        # Add Classification Dense, Compile model and make it ready for optimization
        # model binary
        model.add(Dense(number_classes, activation=activation_fun))
        model.compile(loss=loss_fun, optimizer=optimizer, metrics=['accuracy'])
        return model

# important parameters to test
# filter size
# filter count this is the most variable parameter,
# it’s a power of two anywhere between 32 and 1024. Using more filters results in a more powerful model,
# but we risk overfitting due to increased parameter count. Usually we start with a small number of
# filters at the initial layers, and progressively increase the count as we go deeper into the network.

# todo if this one works do one for 2D (matrixes)

# # https://github.com/BadrulAlom/Protein-Function-CNN-Model/blob/master/ProteinFunction_CNNModel/code/BioScript.py
# class ProteinSequencingCNNModel:
#     def __init__(self):
#         self.input_Height = 1  # Set to 0 to start with
#         self.input_Width = 1  # Set to 0 to start with
#         self.input_Channels = 1  # (e.g. 3 for RGB)
#         self._input_TrainData = []
#         self._input_TrainLabels = []
#         self._input_TestData = []
#         self._input_PredData = []
#         self._input_TestLabels = []
#
#         self.output_FilePath = ""
#         self.output_Filename = "ProteinSequencingCNNModel.h5"
#         self.model_batchsize = 64
#         self.model_epochs = 1
#         self.model = Sequential()
#
#     def createModel(self):
#         # https://keras.io/getting-started/sequential-model-guide/
#         # Reset model
#
#         model_params = [1, 32, 1, 1]
#
#         self.model = Sequential()
#
#         # Layer 1
#         # From TF Conv2DFunction: input_shape = (128, 128, 3) for 128x128 RGB pictures in `data_format where "channels_last"
#
#         self.model.add(Conv2D(32, (1, 3), activation='relu',
#                               input_shape=(self.input_Height, self.input_Width, self.input_Channels)))  #
#         self.model.add(Conv2D(32, (1, 3), activation='relu'))
#         self.model.add(MaxPooling2D(pool_size=(1, 1)))
#         self.model.add(Dropout(0.25))
#
#         # Layer 2
#         self.model.add(Conv2D(64, (1, 1), activation='relu'))
#         self.model.add(Conv2D(64, (1, 1), activation='relu'))
#         self.model.add(MaxPooling2D(pool_size=(1, 1)))
#         self.model.add(Dropout(0.25))
#
#         # Layer 3
#         self.model.add(Flatten())
#         self.model.add(Dense(256, activation='relu'))
#         self.model.add(Dropout(0.5))
#         self.model.add(Dense(4, activation='softmax'))
#
#         # Layer 4
#         sgd = SGD(lr=model_learningRate, decay=1e-6, momentum=0.9, nesterov=True)
#         self.model.compile(loss='categorical_crossentropy', optimizer=sgd)
#
#         self.model.compile(optimizer='rmsprop',
#                            loss='categorical_crossentropy',
#                            metrics=['accuracy'])
#
#         return self.model
#
#
# # https://missinglink.ai/guides/convolutional-neural-networks/convolutional-neural-network-tutorial-basic-advanced/
# model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(64, (5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(1000, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))
#
#
# layer1 = create_new_conv_layer(x_shaped, 1, 32, [5, 5], [2, 2], name='layer1')
# layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')
# flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])
