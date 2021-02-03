# -*- coding: utf-8 -*-
"""
##############################################################################

File that describes the Antimicrobial peptides case study.
This section will present a comparative analysis to demonstrate the application and performance of proPythia
for addressing sequence-based prediction problems.


do a comparative analysis using the AMPEP article as case study.


The first case study is with antimicorbial peptides and tries to replicate the study made by P. Bhadra and all,
“AMP: Sequence-based prediction of antimicrobial peptides using distribution patterns of amino acid properties
and random forest” which is described to highly perform on AMP prediction methods.
In the publication, Bhadra et al., used a dataset with a positive:negative ratio (AMP/non-AMP) of 1:3
, based on the distribution patterns of aa properties along the sequence (CTD features),
with a 10 fold cross validation RF model. The collection of data with sets of AMP and non-AMP data is freely
available at https://sourceforge.net/projects/axpep/files/).
Their model obtained a sensitivity of 0.95, a specificity and accuracy of 0.96, MCC of 0.9 and AUC-ROC of 0.98.


P. Bhadra, J. Yan, J. Li, S. Fong, and S. W. Siu, “AMP: Sequence-based prediction
of antimicrobial peptides using distribution patterns of amino acid properties and
random forest,” Scientific Reports, vol. 8, no. 1, pp. 1–10, 2018.


Authors: Ana Marta Sequeira

Date: 01/2021

Email:

##############################################################################
"""


import sys
import os
sys.path.append('/home/amsequeira/propythia')

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # tf.config.experimental.set_visible_devices(gpus[:1], 'GPU')

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
import csv
import keras as K
import pandas as pd
from propythia.sequence import ReadSequence
from propythia.descriptors import Descriptor
from propythia.preprocess import Preprocess
from propythia.feature_selection import FeatureSelection
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from propythia.shallow_ml import ShallowML
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, matthews_corrcoef
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import fractions

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
# import logging
# logging.disable(logging.WARNING)
# logging.getLogger("tensorflow").setLevel(logging.FATAL)

from tensorflow import keras
from tensorflow.keras.layers import LSTM, ConvLSTM2D, BatchNormalization
from keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from propythia.deep_ml import DeepML
from propythia.manifold import Manifold
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.utils import shuffle

# GET datasets

from Bio.SeqIO.FastaIO import SimpleFastaParser

amp_eval_file = '/home/amsequeira/propythia/propythia/example/AMP/datasets/AMP.eval.fa'
amp_test_file = '/home/amsequeira/propythia/propythia/example/AMP/datasets/AMP.te.fa'
amp_train_file = '/home/amsequeira/propythia/propythia/example/AMP/datasets/AMP.tr.fa'

non_amp_eval_file = '/home/amsequeira/propythia/propythia/example/AMP/datasets/DECOY.eval.fa'
non_amp_test_file = '/home/amsequeira/propythia/propythia/example/AMP/datasets/DECOY.te.fa'
non_amp_train_file = '/home/amsequeira/propythia/propythia/example/AMP/datasets/DECOY.tr.fa'

def fasta_to_df(file, label):
    identifiers = []
    seq = []
    lab=[]
    with open(file) as fasta_file:  # Will close handle cleanly
        for title, sequence in SimpleFastaParser(fasta_file):
            identifiers.append(title.split(None, 1)[0])  # First word is ID
            seq.append(sequence)
            lab.append(label)
    df = pd.DataFrame(list(zip(identifiers, seq, lab)), columns=['identifiers', 'seq', 'label'])
    return df

amp_eval = fasta_to_df(file=amp_eval_file, label=1)
amp_test = fasta_to_df(file=amp_test_file, label=1)
amp_train = fasta_to_df(file=amp_train_file, label=1)
non_amp_eval = fasta_to_df(file=non_amp_eval_file, label=0)
non_amp_test = fasta_to_df(file=non_amp_test_file, label=0)
non_amp_train = fasta_to_df(file=non_amp_train_file, label=0)

eval = pd.concat([amp_eval, non_amp_eval])
train = pd.concat([amp_train, non_amp_train])
test = pd.concat([amp_test, non_amp_test])

eval = shuffle(eval)
train = shuffle(train)
test = shuffle(test)



print(max(train.seq.apply(len))) # 183 as articl describes

# PAD ZEROS 200 20 aa  X = 0 categorcial encoding
def pad_sequence(df):
    sequences = df['seq'].tolist()
    alphabet = "XARNDCEQGHILKMFPSTWYV"
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    # {'X': 0,
    #  'A': 1,
    #  'R': 2,
    #  'N': 3,
    #  'D': 4,...
    sequences_integer_ecoded = []
    for seq in sequences:
        # seq = seq.replace('X', 0)  # unknown character eliminated
        # define a mapping of chars to integers
        # integer encode input data
        integer_encoded = [char_to_int[char] for char in seq]
        sequences_integer_ecoded.append(integer_encoded)
    fps_x = pad_sequences(sequences_integer_ecoded, maxlen=200, padding='pre', value=0.0)   # (4042, 200)
    return fps_x
    # array([[ 0,  0,  0, ...,  8,  1, 11],
    #        [ 0,  0,  0, ...,  8, 16,  5],
    #        [ 0,  0,  0, ..., 12, 17,  5],
    #        ...,
    #        [ 0,  0,  0, ...,  1, 20,  3],
    #        [ 0,  0,  0, ..., 16, 20,  4],
    #        [ 0,  0,  0, ...,  3, 11,  9]], dtype=int32)


x_eval = pad_sequence(eval)
x_train = pad_sequence(train)
x_test = pad_sequence(test)


# as the model does not resemble any of the basic arquitectures , we built the model accordingly to the article
tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()

def veltri_model():
    # with strategy.scope():
    model = Sequential()
    model.add(Input(shape=(200,)))
    model.add(Embedding(input_dim=21, output_dim=128, input_length=200, mask_zero=True))
    model.add(Conv1D(
        filters=64,
        kernel_size=16,
        strides=1,
        padding='same',
        activation='relu'))
    model.add(MaxPool1D(pool_size=5, strides=1, padding='same'))
    model.add(LSTM(units=100,
                   dropout=0.1,
                   unroll=True,
                   return_sequences=False,
                   stateful=False))

    # Add Classification Dense, Compile model and make it ready for optimization
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model




eval = pd.concat([amp_eval, non_amp_eval])
train = pd.concat([amp_train, non_amp_train])
test = pd.concat([amp_test, non_amp_test])

eval = shuffle(eval)
train = shuffle(train)
test = shuffle(test)

x_eval = pad_sequence(eval)
x_train = pad_sequence(train)
x_test = pad_sequence(test)

dl=DeepML(x_train, train['label'],x_test, test['label'], number_classes=2, problem_type='binary',
          x_dval=x_eval, y_dval=eval['label'], epochs=10, batch_size=32,
          path='/home/amsequeira/propythia/propythia/example/AMP', report_name='veltri', verbose=1)
model = KerasClassifier(build_fn=veltri_model)
veltri = dl.run_model(model)
scores, report, cm, cm2 = dl.model_complete_evaluate()

dl.save_model(model=None, path='model.h5')
c  = dl.load_model(path='')
for lay in c.layers:
    print(lay.name)
    print(lay.get_weights())
embedding_35
conv1d_35

m = Manifold(x_data=c.layers[1].get_weights()[0])
m.run_tsne()
# Conv
# layer embedding weights are first extracted from the trained Keras
# model using all the data. Then, t-distributed stochastic neighbor
# embedding (t-SNE) (Van der Maaten and Hinton, 2008) is applied
# (n_components: 2, init: pca, perplexity: 30, n_iter_
# without_progress: 300, method: exact)
# sem shuffle c os tres datasets separados

# ('Training Accuracy mean: ', 0.8933645486831665)
# ('Validation Accuracy mean: ', 0.8937062919139862)
# ('Training Loss mean: ', 0.2542675271630287)
# ('Validation Loss mean: ', 0.3319994188845158)
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 200, 128)          2688
# _________________________________________________________________
# conv1d (Conv1D)              (None, 200, 64)           131136
# _________________________________________________________________
# max_pooling1d (MaxPooling1D) (None, 200, 64)           0
# _________________________________________________________________
# lstm (LSTM)                  (None, 100)               66000
# _________________________________________________________________
# dense (Dense)                (None, 1)                 101
# =================================================================
# Total params: 199,925
# Trainable params: 199,925
# Non-trainable params: 0
# _________________________________________________________________
# #
# === Confusion Matrix ===
# [[669  43]
#  [107 605]]
# === Classification Report ===
# precision    recall  f1-score   support
# 0       0.86      0.94      0.90       712
# 1       0.93      0.85      0.89       712
# accuracy                           0.89      1424
# macro avg       0.90      0.89      0.89      1424
# weighted avg       0.90      0.89      0.89      1424
#
# # metrics                          scores
# 0   Accuracy                        0.894663
# 1        MCC                        0.792534
# 2   log_loss                        0.302855
# 3   f1 score                        0.889706
# 4    roc_auc                        0.894663
# 5  Precision   [0.5, 0.933641975308642, 1.0]
# 6     Recall  [1.0, 0.8497191011235955, 0.0]

# com shuffle c os tres datasets separados

#
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy', 'lr'])
# ('Training Accuracy mean: ', 0.8969555079936982)
# ('Validation Accuracy mean: ', 0.8881118893623352)
# ('Training Loss mean: ', 0.2516489543020725)
# ('Validation Loss mean: ', 0.29850528538227084)
# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, 200, 128)          2688
# _________________________________________________________________
# conv1d_1 (Conv1D)            (None, 200, 64)           131136
# _________________________________________________________________
# max_pooling1d_1 (MaxPooling1 (None, 200, 64)           0
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 100)               66000
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 101
#                                                        =================================================================
# Total params: 199,925
# Trainable params: 199,925
# Non-trainable params: 0
# _________________________________________________________________
#
# === Confusion Matrix ===
# [[623  89]
#  [ 90 622]]
# === Classification Report ===
# precision    recall  f1-score   support
# 0       0.87      0.88      0.87       712
# 1       0.87      0.87      0.87       712
# accuracy                           0.87      1424
# macro avg       0.87      0.87      0.87      1424
# weighted avg       0.87      0.87      0.87      1424
# metrics                          scores
# 0   Accuracy                        0.874298
# 1        MCC                        0.748596
# 2   log_loss                        0.325068
# 3   f1 score                        0.874209
# 4    roc_auc                        0.874298
# 5  Precision  [0.5, 0.8748241912798875, 1.0]
# 6     Recall  [1.0, 0.8735955056179775, 0.0]


# sem shuffle c juntar os datasets eval e train. e deixar o eval so 10%

# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy', 'lr'])
# ('Training Accuracy mean: ', 0.8956204414367676)
# ('Validation Accuracy mean: ', 0.9098130881786346)
# ('Training Loss mean: ', 0.2570084482431412)
# ('Validation Loss mean: ', 0.2730965778231621)
# Model: "sequential_2"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_2 (Embedding)      (None, 200, 128)          2688
# _________________________________________________________________
# conv1d_2 (Conv1D)            (None, 200, 64)           131136
# _________________________________________________________________
# max_pooling1d_2 (MaxPooling1 (None, 200, 64)           0
# _________________________________________________________________
# lstm_2 (LSTM)                (None, 100)               66000
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 101
#                                                        =================================================================
# Total params: 199,925
# Trainable params: 199,925
# Non-trainable params: 0
# _________________________________________________________________
#
# === Confusion Matrix ===
# [[653  59]
#  [ 84 628]]
# === Classification Report ===
# precision    recall  f1-score   support
# 0       0.89      0.92      0.90       712
# 1       0.91      0.88      0.90       712
# accuracy                           0.90      1424
# macro avg       0.90      0.90      0.90      1424
# weighted avg       0.90      0.90      0.90      1424
# metrics                          scores
# 0   Accuracy                        0.899579
# 1        MCC                         0.79965
# 2   log_loss                        0.267438
# 3   f1 score                        0.897784
# 4    roc_auc                        0.899579
# 5  Precision  [0.5, 0.9141193595342066, 1.0]
# 6     Recall  [1.0, 0.8820224719101124, 0.0]


# com shuffle c juntar os datasets eval e train. e deixar o eval so 10% automatico

# ('Training Accuracy mean: ', 0.9212200224399567)
# ('Validation Accuracy mean: ', 0.9051401793956757)
# ('Training Loss mean: ', 0.20853933393955232)
# ('Validation Loss mean: ', 0.25301963835954666)
# Model: "sequential_3"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_3 (Embedding)      (None, 200, 128)          2688
# _________________________________________________________________
# conv1d_3 (Conv1D)            (None, 200, 64)           131136
# _________________________________________________________________
# max_pooling1d_3 (MaxPooling1 (None, 200, 64)           0
# _________________________________________________________________
# lstm_3 (LSTM)                (None, 100)               66000
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 101
#                                                        =================================================================
# Total params: 199,925
# Trainable params: 199,925
# Non-trainable params: 0
# _________________________________________________________________

# === Confusion Matrix ===
# [[673  39]
#  [ 90 622]]
# === Classification Report ===
# precision    recall  f1-score   support
# 0       0.88      0.95      0.91       712
# 1       0.94      0.87      0.91       712
# accuracy                           0.91      1424
# macro avg       0.91      0.91      0.91      1424
# weighted avg       0.91      0.91      0.91      1424
# metrics                          scores
# 0   Accuracy                         0.90941
# 1        MCC                        0.820929
# 2   log_loss                        0.316688
# 3   f1 score                        0.906045
# 4    roc_auc                         0.90941
# 5  Precision  [0.5, 0.9409984871406959, 1.0]
# 6     Recall  [1.0, 0.8735955056179775, 0.0]

# all data
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy', 'lr'])
# ('Training Accuracy mean: ', 0.9087499976158142)
# ('Validation Accuracy mean: ', 0.9008427202701569)
# ('Training Loss mean: ', 0.23495510816574097)
# ('Validation Loss mean: ', 0.2425863265991211)
# Model: "sequential_4"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_4 (Embedding)      (None, 200, 128)          2688
# _________________________________________________________________
# conv1d_4 (Conv1D)            (None, 200, 64)           131136
# _________________________________________________________________
# max_pooling1d_4 (MaxPooling1 (None, 200, 64)           0
# _________________________________________________________________
# lstm_4 (LSTM)                (None, 100)               66000
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 101
# =================================================================
# Total params: 199,925
# Trainable params: 199,925
# Non-trainable params: 0
# _________________________________________________________________

#
# === Confusion Matrix ===
# [[1714   64]
#  [  76 1702]]
# === Classification Report ===
# precision    recall  f1-score   support
# 0       0.96      0.96      0.96      1778
# 1       0.96      0.96      0.96      1778
# accuracy                           0.96      3556
# macro avg       0.96      0.96      0.96      3556
# weighted avg       0.96      0.96      0.96      3556
# metrics                          scores
# 0   Accuracy                         0.96063
# 1        MCC                        0.921281
# 2   log_loss                        0.120576
# 3   f1 score                        0.960497
# 4    roc_auc                         0.96063
# 5  Precision  [0.5, 0.9637599093997735, 1.0]
# 6     Recall  [1.0, 0.9572553430821147, 0.0]

# all data CV
eval = pd.concat([amp_eval, non_amp_eval])
train = pd.concat([amp_train, non_amp_train])
test = pd.concat([amp_test, non_amp_test])

train_tune = pd.concat([eval, train])
eval = shuffle(eval)
train = shuffle(train)
test = shuffle(test)

all_data = pd.concat([eval, train, test])
x_eval = pad_sequence(eval)
x_train = pad_sequence(train)
x_test = pad_sequence(test)
x_all_data = pad_sequence(all_data)
x_train_tune = pad_sequence(train_tune)

dl=DeepML(x_all_data, all_data['label'],x_test = None, y_test = None, number_classes=2, problem_type='binary',
          x_dval=None, y_dval=None, epochs=10, batch_size=32,
          path='/home/amsequeira/propythia/propythia/example/AMP', report_name='veltri', verbose=1)
model = KerasClassifier(build_fn=veltri_model)
veltri_cv = dl.train_model_cv(x_cv = x_all_data, y_cv = all_data['label'], cv=5, model=model)




# dl.precision_recall_curve(show=True, path_save='try_deep_pre_recall.png', batch_size=1)
# dl.roc_curve(path_save='try_deep_plot_roc_curve.png', show=True, batch_size=1)
dl.plot_learning_curve(path_save='plot_learning_curve', show=True, scalability=True, performance=True)
# # dl.save_model(path=model_name)

# K.clear_session()
# tf.keras.backend.clear_session()




dl=DeepML(x_train_tune, train_tune['label'],x_test = x_test, y_test = test['label'], number_classes=2, problem_type='binary',
          x_dval=None, y_dval=None, epochs=10, batch_size=512,
          path='/home/amsequeira/propythia/propythia/example/AMP', report_name='veltri', verbose=1)
model = KerasClassifier(build_fn=veltri_model)
veltri = dl.run_model(model)
scores, report, cm, cm2 = dl.model_complete_evaluate()

#
# === Confusion Matrix ===
# [[670  42]
#  [124 588]]
# === Classification Report ===
# precision    recall  f1-score   support
# 0       0.84      0.94      0.89       712
# 1       0.93      0.83      0.88       712
# accuracy                           0.88      1424
# macro avg       0.89      0.88      0.88      1424
# weighted avg       0.89      0.88      0.88      1424
# metrics                          scores
# 0   Accuracy                        0.883427
# 1        MCC                        0.771991
# 2   log_loss                        0.293993
# 3   f1 score                        0.876304
# 4    roc_auc                        0.883427
# 5  Precision  [0.5, 0.9333333333333333, 1.0]
# 6     Recall  [1.0, 0.8258426966292135, 0.0]
# 7         sn                        0.825843
# 8         sp                        0.941011



def veltri_model_2(units):
    # with strategy.scope():
    model = Sequential()
    model.add(Input(shape=(200,)))
    model.add(Embedding(input_dim=21, output_dim=128, input_length=200, mask_zero=True))
    model.add(Conv1D(
        filters=64,
        kernel_size=16,
        strides=1,
        padding='same',
        activation='relu'))
    model.add(MaxPool1D(pool_size=5, strides=1, padding='same'))
    # model.add(Conv1D(
    #     filters=32,
    #     kernel_size=16,
    #     strides=1,
    #     padding='same',
    #     activation='relu'))
    # model.add(MaxPool1D(pool_size=5, strides=1, padding='same'))
    model.add(LSTM(units=200,
                   dropout=0.1,
                   unroll=True,
                   return_sequences=False,
                   stateful=False))

    # Add Classification Dense, Compile model and make it ready for optimization
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



dl=DeepML(x_train_tune, train_tune['label'],x_test = x_test, y_test = test['label'], number_classes=2, problem_type='binary',
          x_dval=None, y_dval=None, epochs=500, batch_size=256,
          path='/home/amsequeira/propythia/propythia/example/AMP', report_name='veltri', verbose=1)
model2=KerasClassifier(build_fn=veltri_model_2)
veltri2 = dl.run_model(model2)
scores, report, cm, cm2 = dl.model_complete_evaluate()




########################################################################################################################
########################################################################################################################
# PHYSICO CHEMICAL

amp_eval_file = '/home/amsequeira/propythia/propythia/example/AMP/datasets/AMP.eval.fa'
amp_test_file = '/home/amsequeira/propythia/propythia/example/AMP/datasets/AMP.te.fa'
amp_train_file = '/home/amsequeira/propythia/propythia/example/AMP/datasets/AMP.tr.fa'

non_amp_eval_file = '/home/amsequeira/propythia/propythia/example/AMP/datasets/DECOY.eval.fa'
non_amp_test_file = '/home/amsequeira/propythia/propythia/example/AMP/datasets/DECOY.te.fa'
non_amp_train_file = '/home/amsequeira/propythia/propythia/example/AMP/datasets/DECOY.tr.fa'

def fasta_to_df(file, label):
    identifiers = []
    seq = []
    lab=[]
    with open(file) as fasta_file:  # Will close handle cleanly
        for title, sequence in SimpleFastaParser(fasta_file):
            identifiers.append(title.split(None, 1)[0])  # First word is ID
            seq.append(sequence)
            lab.append(label)
    df = pd.DataFrame(list(zip(identifiers, seq, lab)), columns=['identifiers', 'seq', 'label'])
    return df

amp_eval = fasta_to_df(file=amp_eval_file, label=1)
amp_test = fasta_to_df(file=amp_test_file, label=1)
amp_train = fasta_to_df(file=amp_train_file, label=1)
non_amp_eval = fasta_to_df(file=non_amp_eval_file, label=0)
non_amp_test = fasta_to_df(file=non_amp_test_file, label=0)
non_amp_train = fasta_to_df(file=non_amp_train_file, label=0)

eval = pd.concat([amp_eval, non_amp_eval])
train = pd.concat([amp_train, non_amp_train])
test = pd.concat([amp_test, non_amp_test])


# calculate features
def calculate_feature(data):
    list_feature = []
    count = 0
    for seq in data['seq']:
        count += 1
        res = {'seq': seq}
        sequence = ReadSequence()  # creating sequence object
        ps = sequence.read_protein_sequence(seq)
        protein = Descriptor(ps)  # creating object to calculate descriptors
        # feature = protein.adaptable([32,20,24]) # using the function adaptable. calculate CTD, aac, dpc, paac and  feature
        feature = protein.adaptable([19, 20, 21, 24, 26, 32], lamda_paac=10, lamda_apaac=10) #minimun len = 11
        # feature = protein.get_all(lamda_paac=5, lamda_apaac=5) #minimal seq len = 5
        # lambda should not be larger than len(sequence)
        res.update(feature)
        list_feature.append(res)
        print(count)
    df = pd.DataFrame(list_feature)
    return df

eval_feat = calculate_feature(eval)
train_feat = calculate_feature(train)
test_feat = calculate_feature(test)

eval_feat.to_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/eval_feat.csv', index=False)
train_feat.to_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/train_feat.csv', index=False)
test_feat.to_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/test_feat.csv', index=False)

eval_feat = pd.read_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/eval_feat.csv')
train_feat = pd.read_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/train_feat.csv')
test_feat = pd.read_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/test_feat.csv')


train_tune_feat = pd.concat([eval_feat , train_feat ])
train_tune_feat = train_tune_feat.drop('sequence', axis=1) # [2132 rows x 652 columns]
test_feat = test_feat.drop('sequence', axis=1) # [1424 rows x 652 columns]

train_tune_y = pd.concat([eval, train])['label']
test_y=test['label']


scaler = StandardScaler().fit(train_tune_feat)
X_train = scaler.transform(train_tune_feat)
X_test = scaler.transform(test_feat)

ml = ShallowML(X_train=X_train, X_test=X_test, y_train=train_tune_y, y_test=test_y, report_name=None,
               columns_names=train_tune_feat.columns)
param_grid = [{'clf__n_estimators': [10, 100, 200, 500], 'clf__max_features': ['sqrt', 'log2']}]
rf_all = ml.train_best_model('rf', score=make_scorer(matthews_corrcoef), param_grid=param_grid, cv=10)
scores, report, cm, cm2 = ml.score_testset(rf_all)

# Model with rank: 1
# Mean validation score: 0.818 (std: 0.026)
# Parameters: {'clf__max_features': 'log2', 'clf__n_estimators': 500}
#
# Model with rank: 2
# Mean validation score: 0.813 (std: 0.030)
# Parameters: {'clf__max_features': 'log2', 'clf__n_estimators': 200}
#
# Model with rank: 3
# Mean validation score: 0.810 (std: 0.034)
# Parameters: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# make_scorer(matthews_corrcoef)
# 10
# Best score (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.818
# Parameters:	{'clf__max_features': 'log2', 'clf__n_estimators': 500}
# 0.768419 (0.018745) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 10}
# 0.809771 (0.033761) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.809701 (0.043281) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 200}
# 0.808650 (0.035113) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
# 0.769301 (0.026958) with: {'clf__max_features': 'log2', 'clf__n_estimators': 10}
# 0.807524 (0.026900) with: {'clf__max_features': 'log2', 'clf__n_estimators': 100}
# 0.813030 (0.029605) with: {'clf__max_features': 'log2', 'clf__n_estimators': 200}
# 0.817703 (0.025931) with: {'clf__max_features': 'log2', 'clf__n_estimators': 500}
# clf__max_features  clf__n_estimators     means      stds
# 0              sqrt                 10  0.768419  0.018745
# 1              sqrt                100  0.809771  0.033761
# 2              sqrt                200  0.809701  0.043281
# 3              sqrt                500  0.808650  0.035113
# 4              log2                 10  0.769301  0.026958
# 5              log2                100  0.807524  0.026900
# 6              log2                200  0.813030  0.029605
# 7              log2                500  0.817703  0.025931
# array([[645,  67],
#        [ 67, 645]])
# {'Accuracy': 0.9058988764044944,
#  'MCC': 0.8117977528089888,
#  'log_loss': 0.30931814152031073,
#  'f1 score': 0.9058988764044944,
#  'roc_auc': 0.9058988764044945,
#  'Precision': array([0.5       , 0.90589888, 1.        ]),
#  'Recall': array([1.        , 0.90589888, 0.        ]),
#  'fdr': 0.09410112359550561,
#  'sn': 0.9058988764044944,
#  'sp': 0.9058988764044944}
df = ml.features_importances_df(classifier=rf_all, model_name='rf', top_features=30,
                                column_to_sort='mean_coef')
# importance
# chargedensity                 0.024659
# _ChargeC3                     0.024078
# charge                        0.021781
# IsoelectricPoint              0.018297
# PAAC6                         0.015302
# _ChargeT23                    0.014370
# APAAC6                        0.012385
# _SolventAccessibilityC1       0.012137
# E                             0.011427
# _PolarityC2                   0.010714
# _ChargeD3100                  0.010591
# _SolventAccessibilityD1100    0.010231
# C                             0.009768
# PAAC5                         0.009640
# D                             0.009612
# PAAC4                         0.009547
# APAAC5                        0.009528
# PAAC27                        0.009473
# _ChargeT12                    0.009326
# _ChargeD3075                  0.008843


ml = ShallowML(X_train=X_train, X_test=X_test, y_train=train_tune_y, y_test=test_y, report_name=None,
               columns_names=train_tune_feat.columns)
svm_all = ml.train_best_model('svm', score=make_scorer(matthews_corrcoef), param_grid=None, cv=10)
scores, report, cm, cm2 = ml.score_testset(svm_all)


#
# Model with rank: 1
# Mean validation score: 0.810 (std: 0.031)
# Parameters: {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#
# Model with rank: 2
# Mean validation score: 0.805 (std: 0.044)
# Parameters: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#
# Model with rank: 3
# Mean validation score: 0.804 (std: 0.027)
# Parameters: {'clf__C': 1.0, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
#
# make_scorer(matthews_corrcoef)
# 10
# Best score (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.810
# Parameters:	{'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.739632 (0.024658) with: {'clf__C': 0.01, 'clf__kernel': 'linear'}
# 0.696873 (0.032107) with: {'clf__C': 0.1, 'clf__kernel': 'linear'}
# 0.659751 (0.051784) with: {'clf__C': 1.0, 'clf__kernel': 'linear'}
# 0.659751 (0.051784) with: {'clf__C': 10, 'clf__kernel': 'linear'}
# 0.146457 (0.293212) with: {'clf__C': 0.01, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.136139 (0.291925) with: {'clf__C': 0.01, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.110404 (0.221683) with: {'clf__C': 0.01, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.762038 (0.028498) with: {'clf__C': 0.1, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.776389 (0.027463) with: {'clf__C': 0.1, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.721855 (0.033937) with: {'clf__C': 0.1, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.809741 (0.030626) with: {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.804295 (0.026741) with: {'clf__C': 1.0, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.793914 (0.025133) with: {'clf__C': 1.0, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.804622 (0.043624) with: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.803337 (0.039238) with: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.778536 (0.026553) with: {'clf__C': 10, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# clf__C clf__kernel clf__gamma     means      stds
# 0     0.01      linear        NaN  0.739632  0.024658
# 1     0.10      linear        NaN  0.696873  0.032107
# 2     1.00      linear        NaN  0.659751  0.051784
# 3    10.00      linear        NaN  0.659751  0.051784
# 4     0.01         rbf      scale  0.146457  0.293212
# 5     0.01         rbf      0.001  0.136139  0.291925
# 6     0.01         rbf     0.0001  0.110404  0.221683
# 7     0.10         rbf      scale  0.762038  0.028498
# 8     0.10         rbf      0.001  0.776389  0.027463
# 9     0.10         rbf     0.0001  0.721855  0.033937
# 10    1.00         rbf      scale  0.809741  0.030626
# 11    1.00         rbf      0.001  0.804295  0.026741
# 12    1.00         rbf     0.0001  0.793914  0.025133
# 13   10.00         rbf      scale  0.804622  0.043624
# 14   10.00         rbf      0.001  0.803337  0.039238
# 15   10.00         rbf     0.0001  0.778536  0.026553


# Out[77]:
# {'Accuracy': 0.9044943820224719,
#  'MCC': 0.8099127255330534,
#  'f1 score': 0.902158273381295,
#  'roc_auc': 0.904494382022472,
#  'Precision': array([0.5       , 0.92477876, 1.        ]),
#  'Recall': array([1.        , 0.88061798, 0.        ]),
#  'fdr': 0.0752212389380531,
#  'sn': 0.8806179775280899,
#  'sp': 0.9283707865168539}
#
#
# array([[661,  51],
#        [ 85, 627]])


train_tune_feat = train_tune_feat.filter(regex=r'_.+D\d', axis=1) # 105
test_feat = test_feat.filter(regex=r'_.+D\d', axis=1)

ml = ShallowML(X_train=train_tune_feat, X_test=test_feat, y_train=train_tune_y, y_test=test_y, report_name=None,
               columns_names=train_tune_feat.columns)
rf_all = ml.train_best_model('rf', score=make_scorer(matthews_corrcoef), param_grid=None, cv=10)
scores, report, cm, cm2 = ml.score_testset(rf_all)
df = ml.features_importances_df(classifier=rf_all, model_name='rf', top_features=30,
                                column_to_sort='mean_coef')

# Model with rank: 1
# Mean validation score: 0.739 (std: 0.043)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# Model with rank: 2
# Mean validation score: 0.739 (std: 0.045)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
#
# Model with rank: 3
# Mean validation score: 0.732 (std: 0.042)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 500}
#
# make_scorer(matthews_corrcoef)
# 10
# Best score (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.739
# Parameters:	{'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.668371 (0.044702) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 10}
# 0.739036 (0.042997) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.738829 (0.044627) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
# 0.657992 (0.045032) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 10}
# 0.725472 (0.050117) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 100}
# 0.732455 (0.041697) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 500}
# clf__bootstrap clf__criterion  ...     means      stds
# 0            True           gini  ...  0.668371  0.044702
# 1            True           gini  ...  0.739036  0.042997
# 2            True           gini  ...  0.738829  0.044627
# 3            True           gini  ...  0.657992  0.045032
# 4            True           gini  ...  0.725472  0.050117
# 5            True           gini  ...  0.732455  0.041697
#
# {'Accuracy': 0.8665730337078652,
#  'MCC': 0.7335629303616759,
#  'log_loss': 0.35708389623221887,
#  'f1 score': 0.8642857142857142,
#  'roc_auc': 0.8665730337078652,
#  'Precision': array([0.5       , 0.87936047, 1.        ]),
#  'Recall': array([1.       , 0.8497191, 0.       ]),
#  'fdr': 0.12063953488372094,
#  'sn': 0.8497191011235955,
#  'sp': 0.8834269662921348}
# array([[629,  83],
#        [107, 605]])

# importance
# _SolventAccessibilityD1100    0.056122
# _ChargeD3100                  0.042159
# _ChargeD3075                  0.034739
# _SolventAccessibilityD1075    0.031542
# _ChargeD3025                  0.030993
# _ChargeD3001                  0.030948
# _ChargeD3050                  0.026442
# _NormalizedVDWVD2100          0.025914
# _SolventAccessibilityD1001    0.025158
# _NormalizedVDWVD2075          0.023945
# _ChargeD1025                  0.017918
# _PolarityD2001                0.015563
# _PolarityD2075                0.014489
# _SolventAccessibilityD3100    0.014407
# _PolarityD2100                0.013767
# _NormalizedVDWVD2050          0.012730
# _PolarityD1001                0.012551
# _SolventAccessibilityD1050    0.011980
# _SolventAccessibilityD2075    0.011448
# _PolarityD3025                0.010547
# _PolarityD3100                0.010468
# _HydrophobicityD3001          0.010036
# _PolarityD2050                0.009954
# _HydrophobicityD3100          0.009909
# _SecondaryStrD3001            0.009770
# _PolarityD2025                0.009678
# _HydrophobicityD2001          0.009627
# _SolventAccessibilityD3075    0.009485
# _HydrophobicityD1075          0.009235
# _SolventAccessibilityD2100    0.009221

train_tune_feat = train_tune_feat.filter(regex=r'_.+D\d', axis=1) # 105
test_feat = test_feat.filter(regex=r'_.+D\d', axis=1)
ml = ShallowML(X_train=train_tune_feat, X_test=test_feat, y_train=train_tune_y, y_test=test_y, report_name=None,
               columns_names=train_tune_feat.columns)
svm_all = ml.train_best_model('svm', score=make_scorer(matthews_corrcoef), param_grid=None, cv=10)
scores, report, cm, cm2 = ml.score_testset(svm_all)
df = ml.features_importances_df(classifier=svm_all, model_name='svm', top_features=30,
                                column_to_sort='mean_coef')




eval_feat = pd.read_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/eval_feat.csv')
train_feat = pd.read_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/train_feat.csv')
test_feat = pd.read_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/test_feat.csv')


train_feat = train_feat.drop('sequence', axis=1) # [2132 rows x 652 columns]
eval_feat = eval_feat.drop('sequence', axis=1)
test_feat = test_feat.drop('sequence', axis=1) # [1424 rows x 652 columns]

train_y = train['label']
eval_y = eval['label']
test_y=test['label']

#
# scaler = StandardScaler().fit(train_feat)
# X_train = scaler.transform(train_feat)
# X_test = scaler.transform(test_feat)
# X_val = scaler.transform(eval_feat)


X_test = test_feat.filter(regex=r'_.+D\d', axis=1)
X_train = train_feat.filter(regex=r'_.+D\d', axis=1)
X_val = eval_feat.filter(regex=r'_.+D\d', axis=1)

dl = DeepML(x_train=X_train, y_train=train_y, x_test=X_test, y_test=test_y, x_dval=X_val, y_dval=eval_y,
            number_classes=2, problem_type='binary', epochs=500, batch_size=512,
            path='/home/amsequeira/propythia/propythia/example/AMP', report_name='ampep_veltri', verbose=1)
dnn = dl.run_dnn_simple(
    input_dim=X_train.shape[1],
    optimizer='Adam',
    hidden_layers=(128, 64),
    dropout_rate=(0.3,),
    batchnormalization=(True,),
    l1=1e-5, l2=1e-4,
    final_dropout_value=0.3,
    initial_dropout_value=0.0,
    loss_fun=None, activation_fun=None,
    cv=10, optType='randomizedSearch', param_grid=None, n_iter_search=15, n_jobs=1,
    scoring=make_scorer(matthews_corrcoef))


scores, report, cm, cm2 = dl.model_complete_evaluate()


# dict_keys(['accuracy', 'loss', 'val_accuracy', 'val_loss', 'lr'])
# ('Training Accuracy mean: ', 0.9820794541470326)
# ('Validation Accuracy mean: ', 0.7539176170843361)
# ('Training Loss mean: ', 1.156090444674457)
# ('Validation Loss mean: ', 1.6158131234837274)
# Model: "sequential_662"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense_1938 (Dense)           (None, 128)               83584
# _________________________________________________________________
# batch_normalization_1278 (Ba (None, 128)               512
# _________________________________________________________________
# dropout_1278 (Dropout)       (None, 128)               0
# _________________________________________________________________
# dense_1939 (Dense)           (None, 64)                8256
# _________________________________________________________________
# batch_normalization_1279 (Ba (None, 64)                256
# _________________________________________________________________
# dropout_1279 (Dropout)       (None, 64)                0
# _________________________________________________________________
# dense_1940 (Dense)           (None, 1)                 65
#                                                        =================================================================
# Total params: 92,673
# Trainable params: 92,289
# Non-trainable params: 384
# _________________________________________________________________
# [['Model with rank: 1\n', 'Mean validation score: 0.036 (std: 0.108)\n', "Parameters: {'l2': 0, 'l1': 0.001, 'hidden_layers': (128, 64), 'dropout_rate': 0.3}\n", '\n'], ['Model with rank: 2\n', 'Mean validation score: 0.030 (std: 0.091)\n', "Parameters: {'l2': 0, 'l1': 0, 'hidden_layers': (64,), 'dropout_rate': 0.5}\n", '\n'], ['Model with rank: 3\n', 'Mean validation score: 0.029 (std: 0.086)\n', "Parameters: {'l2': 0.001, 'l1': 0, 'hidden_layers': (64,), 'dropout_rate': 0.4}\n", '\n']]
# Best score (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# df
# means      stds       l2       l1  hidden_layers  dropout_rate
# 14  0.036140  0.108421  0.00000  0.00100      (128, 64)          0.30
# 13  0.030411  0.091232  0.00000  0.00000          (64,)          0.50
# 1   0.028690  0.086071  0.00100  0.00000          (64,)          0.40
# 8   0.027197  0.081591  0.00010  0.00000  (128, 64, 32)          0.30
# 11  0.026502  0.079507  0.00001  0.00001      (128, 64)          0.40
# 10  0.020898  0.062695  0.00000  0.00100      (128, 64)          0.25
# 12  0.020509  0.061528  0.00000  0.00000          (64,)          0.30
# 5   0.017854  0.053563  0.00100  0.00100      (128, 64)          0.25
# 0   0.015032  0.045097  0.00000  0.00010  (128, 64, 32)          0.25
# 4   0.014001  0.042004  0.00100  0.00000       (64, 32)          0.50
# 7   0.010372  0.031117  0.00000  0.00100       (64, 32)          0.10
# 2   0.009503  0.028510  0.00100  0.00000          (64,)          0.50
# 9   0.009097  0.027290  0.00000  0.00000      (128, 64)          0.10
# 3   0.007298  0.021894  0.00001  0.00001       (64, 32)          0.35
# 6  -0.010522  0.031566  0.00100  0.00000  (128, 64, 32)          0.35
#
# metrics                          scores
# 0   Accuracy                        0.854635
# 1        MCC                        0.716515
# 2   log_loss                        0.407887
# 3   f1 score                        0.864262
# 4    roc_auc                        0.854635
# 5  Precision  [0.5, 0.8105781057810578, 1.0]
# 6     Recall   [1.0, 0.925561797752809, 0.0]
# 7         sn                        0.925562
# 8         sp                        0.783708
# array([[558, 154],
#        [ 53, 659]])


# for 105
#
# Epoch 00130: early stopping
# dict_keys(['accuracy', 'loss', 'val_accuracy', 'val_loss', 'lr'])
# ('Training Accuracy mean: ', 0.8464600961941939)
# ('Validation Accuracy mean: ', 0.7761161921116022)
# ('Training Loss mean: ', 0.3493923137967403)
# ('Validation Loss mean: ', 0.4809928114597614)
# Model: "sequential_814"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense_2385 (Dense)           (None, 128)               13568
# _________________________________________________________________
# batch_normalization_1573 (Ba (None, 128)               512
# _________________________________________________________________
# dropout_1573 (Dropout)       (None, 128)               0
# _________________________________________________________________
# dense_2386 (Dense)           (None, 64)                8256
# _________________________________________________________________
# batch_normalization_1574 (Ba (None, 64)                256
# _________________________________________________________________
# dropout_1574 (Dropout)       (None, 64)                0
# _________________________________________________________________
# dense_2387 (Dense)           (None, 32)                2080
# _________________________________________________________________
# batch_normalization_1575 (Ba (None, 32)                128
# _________________________________________________________________
# dropout_1575 (Dropout)       (None, 32)                0
# _________________________________________________________________
# dense_2388 (Dense)           (None, 1)                 33
#                                                        =================================================================
# Total params: 24,833
# Trainable params: 24,385
# Non-trainable params: 448
# _________________________________________________________________
# [['Model with rank: 1\n', 'Mean validation score: 0.013 (std: 0.038)\n', "Parameters: {'l2': 0, 'l1': 0, 'hidden_layers': (128, 64, 32), 'dropout_rate': 0.3}\n", '\n'], ['Model with rank: 2\n', 'Mean validation score: 0.012 (std: 0.036)\n', "Parameters: {'l2': 0.001, 'l1': 1e-05, 'hidden_layers': (128, 64), 'dropout_rate': 0.1}\n", '\n'], ['Model with rank: 3\n', 'Mean validation score: 0.011 (std: 0.034)\n', "Parameters: {'l2': 0.0001, 'l1': 0, 'hidden_layers': (64,), 'dropout_rate': 0.3}\n", '\n']]
# Best score (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# df
# means      stds       l2       l1  hidden_layers  dropout_rate
# 13  0.012826  0.038478  0.00000  0.00000  (128, 64, 32)          0.30
# 1   0.012122  0.036365  0.00100  0.00001      (128, 64)          0.10
# 8   0.011297  0.033891  0.00010  0.00000          (64,)          0.30
# 5   0.008454  0.025362  0.00001  0.00001      (128, 64)          0.35
# 11  0.008150  0.024451  0.00000  0.00000          (64,)          0.40
# 10  0.007355  0.022064  0.00000  0.00100       (64, 32)          0.10
# 7   0.007031  0.021094  0.00100  0.00100  (128, 64, 32)          0.30
# 3   0.005390  0.016171  0.00000  0.00100  (128, 64, 32)          0.10
# 2   0.003953  0.011859  0.00000  0.00000       (64, 32)          0.25
# 6   0.003509  0.010527  0.00001  0.00010          (64,)          0.10
# 12  0.002283  0.006850  0.00100  0.00100          (64,)          0.10
# 9  -0.001756  0.005268  0.00010  0.00001          (64,)          0.10
# 14 -0.001756  0.005268  0.00001  0.00100  (128, 64, 32)          0.10
# 0  -0.007857  0.023570  0.00000  0.00001       (64, 32)          0.25
# 4  -0.008274  0.024821  0.00100  0.00010       (64, 32)          0.25
#
# metrics                          scores
# 0   Accuracy                        0.843399
# 1        MCC                         0.68688
# 2   log_loss                        0.411508
# 3   f1 score                         0.84218
# 4    roc_auc                        0.843399
# 5  Precision  [0.5, 0.8487874465049928, 1.0]
# 6     Recall  [1.0, 0.8356741573033708, 0.0]
# 7         sn                        0.835674
# 8         sp                        0.851124
# Out[110]:
# array([[606, 106],
#        [117, 595]])
######################################################################################################################
# AMP data veltr model
AMP_data = pd.read_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/ampep_feat.csv')
AMP_data['label'] = 1
non_AMP_data_1_3 = pd.read_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/non_ampep_feat.csv')
non_AMP_data_1_3['label'] = 0

from sklearn.utils import shuffle

dataset = pd.concat([AMP_data, non_AMP_data_1_3])
dataset=shuffle(dataset)
fps_y = dataset['label']
fps_x = pad_sequence(dataset)

X_train, X_test, y_train, y_test = train_test_split(fps_x, fps_y, stratify=fps_y)

dl=DeepML(X_train, y_train,X_test, y_test, number_classes=2, problem_type='binary',
          x_dval=None, y_dval=None, epochs=10, batch_size=32,
          path='/home/amsequeira/propythia/propythia/example/AMP', report_name='veltri', verbose=1)
model = KerasClassifier(build_fn=veltri_model)
veltri = dl.run_model(model)
scores, report, cm, cm2 = dl.model_complete_evaluate()



#
#
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy', 'lr'])
# ('Training Accuracy mean: ', 0.9492802798748017)
# ('Validation Accuracy mean: ', 0.9392456710338593)
# ('Training Loss mean: ', 0.14604571908712388)
# ('Validation Loss mean: ', 0.1900278314948082)
# Model: "sequential_815"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_13 (Embedding)     (None, 200, 128)          2688
# _________________________________________________________________
# conv1d_15 (Conv1D)           (None, 200, 64)           131136
# _________________________________________________________________
# max_pooling1d_15 (MaxPooling (None, 200, 64)           0
# _________________________________________________________________
# lstm_15 (LSTM)               (None, 100)               66000
# _________________________________________________________________
# dense_2389 (Dense)           (None, 1)                 101
#                                                        =================================================================
# Total params: 199,925
# Trainable params: 199,925
# Non-trainable params: 0
# _________________________________________________________________
#
# Out[116]: \
# {'Accuracy': 0.8433988764044944,
#  'MCC': 0.6868797316937901,
#  'log_loss': 0.4115075271972859,
#  'f1 score': 0.8421797593772117,
#  'roc_auc': 0.8433988764044944,
#  'Precision': array([0.5       , 0.84878745, 1.        ]),
#  'Recall': array([1.        , 0.83567416, 0.        ]),
#  'sn': 0.8356741573033708,
#  'sp': 0.851123595505618}