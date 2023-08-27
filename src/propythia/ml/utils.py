#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##############################################################################

File with util functions used in classes shallow Learning and DeepLearning
Authors: Ana Marta Sequeira

Date:12/2020

Email:

##############################################################################
"""
import functools
import time
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,matthews_corrcoef,roc_auc_score,f1_score


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer




def validation(model,x_test,y_test):
    print('validation for test data')
    pred = model.predict(x_test)
    pred = np.reshape(pred, (pred.shape[0], 1)) # to put the shape as (number_samples,1)
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, pred))
    print("=== Classification Report ===")
    print(classification_report(y_test, pred))
    print('\n')
    print('test accuracy', accuracy_score(y_test, pred))
    print('f1 score', f1_score(y_test, pred))
    print('roc_auc', roc_auc_score(y_test, pred))
    print('mcc', matthews_corrcoef(y_test, pred))
    print('\n')

def validation_multiclass(model,x_test,y_test):
    print('validation for test data')
    print(x_test)
    pred = model.predict(x_test)
    pred = np.reshape(pred, (pred.shape[0], 1)) # to put the shape as (number_samples,1)
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, pred))
    print("=== Classification Report ===")
    print(classification_report(y_test, pred))
    print('\n')
    print('test accuracy', accuracy_score(y_test, pred))
    print('f1 score weighted', f1_score(y_test, pred, average='weighted'))
    print('f1 score macro', f1_score(y_test, pred, average='macro'))
    print('f1 score micro', f1_score(y_test, pred, average='micro'))
    # print('roc_auc ovr', roc_auc_score(y_test, pred, average= 'weighted',multi_class='ovr'))
    # print('roc_auc ovo', roc_auc_score(y_test, pred, average= 'weighted', multi_class='ovo'))
    print('mcc', matthews_corrcoef(y_test, pred))
    print('\n')


def saveModel(modelType, model, name):
    if modelType != 'dnn':
        pickle.dump(model, open(name, 'wb'))
    else:
        model.model.save(name + '.h5')


def loadDNN(modelname):
    model = load_model(modelname)
    model.compile(loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model




def divide_dataset(file_in, columns_not_x=[], column_label=''):
    # separate labels
    # ec_number = dataset[EC_number]
    # uniprot = dataset[uniprot_ID]
    # sequence = dataset[sequence]
    dataset = pd.read_csv(file_in,low_memory=False)
    dataset_x = dataset.loc[:, dataset.columns[~dataset.columns.isin(columns_not_x)]]
    dataset_y=dataset.loc[:,column_label]
    return dataset, dataset_x, dataset_y
    #
    # x_original=dataset.loc[:, dataset.columns != 'labels']
    # labels=dataset['labels']


def binarize_labels(column_label):
    # print('binarize labels')
    fps_y_bin=[]
    for x in column_label:
        if x == 'non_Enzyme': fps_y_bin.append(int(0))
        elif x == 0 or x == '0': fps_y_bin.append(int(0))
        else: fps_y_bin.append(int(1))
    return fps_y_bin

def show_shapes(x_train, x_test, x_dval, y_train, y_test,y_dval):
    print("Expected: (num_samples, timesteps, channels)")
    print("x_train: {}".format(x_train.shape))
    print("x_test: {}".format(x_test.shape))
    print("x_dval: {}".format(x_dval.shape))

    print("y_train:   {}".format(y_train.shape))
    print("y_test:   {}".format(y_test.shape))
    print("y_dval:   {}".format(y_dval.shape))






#
#
# # def create_dnn(optimizer = 'Adam', dropout_rate=0.0,
# #              batchNormalization = True, embedding=True, hidden_layers=1,
# #              l1 = 0, l2 = 0, units = 512,units1 = 512, units2 = 512,
# #              nb_epoch=500,batch_size=10, input_dim=21, output_dim=100, input_length=None,mask_zero=True):
# def create_dnn(optimizer = 'Adam' , dropout_rate =0.0,
#                batchNormalization = True , embedding=True,hidden_layers =1,
#                l1 = 0, l2 = 0, units = 512 ,units1 = 512 , units2 = 512 ,
#                nb_epoch = 500, batch_size = 10, input_dim = 21, output_dim = 100, input_length = None,mask_zero = True):
#     print('dropout_rate = ', dropout_rate)
#
#     model = Sequential()
#     if embedding:
#         model.add(Embedding(input_dim=input_dim, output_dim=output_dim,input_length=input_length,mask_zero=mask_zero)) #todo input dim? output dim?
#         model.add(Flatten(data_format=None))
#
#     model.add(Dense(units=units, activation="relu"))
#     if batchNormalization:
#         model.add(BatchNormalization())
#
#     model.add(Dropout(dropout_rate))
#     for i in range(hidden_layers):
#         model.add(Dense(units=units, activation="relu", kernel_regularizer = regularizers.l1_l2(l1=l1, l2=l2)))
#         if batchNormalization:
#             model.add(BatchNormalization())
#         model.add(Dropout(dropout_rate))
#
#     model.add(Dense(1, activation='sigmoid'))
#
#     # #Compile model and make it ready for optimization
#     model.compile(loss='binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])
#     # Reduce lr callback
#     # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=50, min_lr=0.00001, verbose=1)
#     return model
#
#
#
# @timer
# def train_dnn(X_train,y_train,X_test,y_test,x_dval,y_dval,
#              optimizer = 'Adam', dropout_rate=0.0,
#              batchNormalization = True, embedding=True, hidden_layers=1,
#              l1 = 0, l2 = 0, units = 512,units1 = 512, units2 = 512,
#              nb_epoch=500,batch_size=10, input_dim=1021, output_dim=100, input_length=400,mask_zero=True,
#              save = True, path = 'savedModels/modelDNN.h1', report = True):
#
#     print(input_dim, output_dim, input_length)
#     model=create_dnn(optimizer=optimizer, dropout_rate=dropout_rate, batchNormalization=batchNormalization,
#                      embedding=embedding,hidden_layers=hidden_layers, l1=l1, l2=l2, units = units, units1=units1,
#                      units2=units2,input_dim=input_dim, output_dim=output_dim, input_length=input_length,
#                      mask_zero=mask_zero, nb_epoch=nb_epoch, batch_size=batch_size)
#
#     print('=== DNN ===')
#     model = KerasClassifier(build_fn=create_dnn, verbose=0)
#     # model = KerasClassifier(build_fn=model,  dropout_rate = dropout_rate, hidden_layers = hidden_layers,
#     #                     l1 = l1, l2 = l2, units = units, verbose=0)
#     # simple early stopping
#     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=10)
#
#     # reduce learning rate
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=50, min_lr=0.00001, verbose=1)
#
#     #checkpoint
#     filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5'
#     cp=ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False,
#                        mode='auto', period=1)
#
#     callbacks = [es, reduce_lr, cp]
#     # callbacks = [reduce_lr]
#
#     #Training
#     dnn = model.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size, callbacks=callbacks,
#                     validation_data = (X_test, y_test),verbose=1)
#
#     if report:
#         print('Training Accuracy mean: ', np.mean(dnn.history['accuracy']))
#         print('Validation Accuracy mean: ', np.mean(dnn.history['val_accuracy']))
#
#         print('Training Loss mean: ', np.mean(dnn.history['loss']))
#         print('Validation Loss mean: ', np.mean(dnn.history['val_loss']))
#
#         summary_accuracy(dnn)
#         summary_loss(dnn)
#
#
#     #todo por aqui os pesos do melhor modelo os parametros
#     # todo por a guardar so o melhor modelo e n de epoc em epoch
#     #todo confirmar q acho q esta a fazer mais hidden layers do q o suposto ou melhor,
#     # adiciona mais uma pelo hidden layers. n sei s a primeira conta como hidden ou nao
#
#     dnn_predict = model.predict(x_dval)
#     print("=== Confusion Matrix ===")
#     print(confusion_matrix(y_dval, dnn_predict))
#     print("=== Classification Report ===")
#     print(classification_report(y_dval, dnn_predict))
#     print('\n')
#     print("=== Test accuracy ===")
#     print(accuracy_score(y_dval, dnn_predict))
#     print('\n')
#
#     print(model.model.summary())
#
#     if save:
#         try:
#             model.model.save(path)
#         except:
#             print('did not save')
#             return model
#
#     return model
#
#
# ############################################ Recurrent neural networks ################################################
# #
# # keras.layers.LSTM(units, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
# #                   kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros',
# #                   unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
# #                   activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
# #                   dropout=0.0, recurrent_dropout=0.0, implementation=2, return_sequences=False, return_state=False,
# #                   go_backwards=False, stateful=False, unroll=False)
# # https://keras.io/layers/recurrent/
#
# # https://adventuresinmachinelearning.com/keras-lstm-tutorial/
# #
#
#
# def create_lstm (input_dim, output_dim, input_length, batchNormalization, embedding, dropout_rate,
#                  optimizer,time_distibuted, hidden_layers, l1, l2, units_lstm, units1,
#                  units2, mask_zero,bidirectional=False):
#     model = Sequential()
#     # Add an Embedding layer expecting input vocab of size 1000, and
#     # output embedding dimension of size 64.
#     if embedding:
#         model.add(Embedding(input_dim=input_dim, output_dim=output_dim,input_length=input_length,mask_zero=mask_zero))
#         model.add(Flatten(data_format=None))
#
#     # Add a LSTM layer with 128 internal units.
#     if bidirectional:
#         model.add(Bidirectional(LSTM(units=units_lstm, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)))
#         model.add(Bidirectional(LSTM(units=units_lstm, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)))
#
#     else:
#         model.add(LSTM(units=units_lstm,return_sequences=True, activation='tanh', recurrent_activation='sigmoid',
#                        use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
#                        bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None,
#                        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
#                        bias_constraint=None, dropout=0.5, recurrent_dropout=0.5, implementation=2, return_state=False,
#                        go_backwards=False, stateful=False, unroll=False))
#
#         model.add(LSTM(units=units_lstm, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
#
#     if batchNormalization:
#         model.add(BatchNormalization())
#
#     if time_distibuted:
#         model.add(TimeDistributed(Dense(input_dim, activation='softmax', name='main_output')))
#         # considering a batch of 32 samples,  where each sample is a sequence  of 10 vectors of 16 dimensions. The batch
#         # input shape of the layer is then(32, 10, 16), and the input_shape, not including the samples dimension, is (10, 16).
#         # TimeDistributed to samples apply a Dense layer to each of the 10 timesteps, independently
#
#     else:
#         model.add(Dense(1, activation='sigmoid'))
#
#     # #Compile model and make it ready for optimization
#     model.compile(loss='binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])
#     # Reduce lr callback
#     # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=50, min_lr=0.00001, verbose=1)
#     model.summary()
#     return model
#
#     # metrics=['categorical_accuracy']
#
# @timer
# def train_lstm( X_train, y_train, X_test, y_test, x_dval, y_dval,
#                 optimizer='Adam', dropout_rate=0.0,
#                 batchNormalization=True, embedding=True, hidden_layers=1,
#                 l1=0, l2=0, units_lstm=512, units1=512, units2=512,
#                 nb_epoch=500, batch_size=10, input_dim=21, output_dim=100, input_length=None, mask_zero=True,
#                 time_distributed=False,
#                 save=True, path='savedModels/modelLSTM.h1', report=True):
#
#     #validation_data data on which to evaluate the loss and any model metrics at the end of each epoch.
#     # tuple (x_val, y_val) of Numpy arrays or tensors - tuple (x_val, y_val, val_sample_weights) of Numpy arrays - dataset or a dataset iterator
#
#     model= create_lstm (optimizer=optimizer, dropout_rate=dropout_rate, batchNormalization=batchNormalization,
#                         embedding=embedding, input_dim=input_dim, output_dim=output_dim, input_length=input_length,
#                         hidden_layers=hidden_layers, l1=l1, l2=l2, units_lstm=units_lstm, units1=units1,
#                         units2=units2, mask_zero=mask_zero, time_distibuted=time_distributed)
#
#     # input_dim=1000, output_dim=64, batchNormalization=True, dropout_rate=0.0,time_distibuted=False
#
#     print('=== LSTM ===') # todo mantive igual ao dnn, n sei se é diferente. se for igual juntar os train
#     model = KerasClassifier(build_fn=create_lstm, dropout_rate=dropout_rate, hidden_layers=hidden_layers,
#                             l1=l1, l2=l2, units=units_lstm, verbose=0)
#
#     # simple early stopping
#     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
#
#     # earlyStopping = EarlyStopping(monitor='val_weighted_accuracy', patience=5, verbose=1, mode='auto') deepaclstm
#
#     # reduce learning rate
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.00001, verbose=1)
#
#     # checkpoint
#     filepath = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
#     cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=0, save_best_only=True,
#                          save_weights_only=False,
#                          mode='auto', period=1)
#
#     callbacks = [es, reduce_lr, cp]
#     # callbacks = [reduce_lr]
#
#     # Training
#     lstm = model.fit(X_train, y_train,epochs=nb_epoch, batch_size=batch_size, callbacks=callbacks,
#                     validation_data=(X_test, y_test), verbose=1)
#
#     if report:
#
#         print('Training Accuracy mean: ', np.mean(lstm.history['accuracy']))
#         print('Validation Accuracy mean: ', np.mean(lstm.history['val_accuracy']))
#
#         print('Training Loss mean: ', np.mean(lstm.history['loss']))
#         print('Validation Loss mean: ', np.mean(lstm.history['val_loss']))
#
#         summary_accuracy(lstm)
#         summary_loss(lstm)
#
#
#     lstm_predict = model.predict(x_dval)
#
#     print("=== Confusion Matrix ===")
#     print(confusion_matrix(y_dval, lstm_predict))
#     print("=== Classification Report ===")
#     print(classification_report(y_dval, lstm_predict))
#     print('\n')
#     print("=== Test accuracy ===")
#     print(accuracy_score(y_dval, lstm_predict))
#     print('\n')
#
#     print(model.model.summary())
#
#     if save:
#         try:
#             model.model.save(path)
#         except:
#             print('did not save')
#             return model
#
#     return model


#generators to feed model wit h data https://medium.com/@anuj_shah/creating-custom-data-generator-for-training-deep-learning-models-part-1-5c62b20cff26

# import numpy as np
# import pandas as pd
# import random
# import tensorflow as tf
# import time as tm
#
# INPUT_SHAPE=[3, 5]
# NUM_POINTS=20
# BATCH_SIZE=7
# EPOCHS=4
#
# def data_gen(num, in_shape):
#     for i in range(num):
#         x = np.random.rand(in_shape[0], in_shape[1])
#         y = random.randint(0,2)
#         yield x, y
#
# train = tf.data.transporter_substrate.from_generator(
#     generator=data_gen,
#     output_types=(tf.float32, tf.int32),
#     output_shapes=([None, INPUT_SHAPE[1]],()),
#     args=([NUM_POINTS, INPUT_SHAPE])
# )
#
# def create_model(input_shape):
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Dense(100, activation="tanh", input_shape=input_shape),
#         tf.keras.layers.LSTM(1, activation="tanh"),
#         tf.keras.layers.Dense(3, activation="softmax")
#     ])
#     return model
#
# model = create_model(input_shape=INPUT_SHAPE)
#
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=1.0),
#     loss= tf.keras.losses.SparseCategoricalCrossentropy()
# )
# print(model.summary())
# model.fit(train.batch(BATCH_SIZE), epochs=EPOCHS, verbose=2)
# model.evaluate(train, steps=None, verbose=1)


#LSTM
# http://colah.github.io/posts/2015-08-Understanding-LSTMs/
# https://towardsdatascience.com/the-fall-of-rnn-lstm-2d1594c74ce0
# https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270
# https://towardsdatascience.com/bert-for-dummies-step-by-step-tutorial-fb90890ffe03
# https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04
# https://towardsdatascience.com/transformers-141e32e69591
# https://distill.pub/2016/augmented-rnns/
# https://skymind.ai/wiki/attention-mechanism-memory-network
# https://bair.berkeley.edu/blog/2019/11/04/proteins/
# https://adventuresinmachinelearning.com/keras-lstm-tutorial/
#  https://www.tensorflow.org/guide/keras/rnn


