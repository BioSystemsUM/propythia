import os
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"] = '3,4,5'
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

import sys

sys.path.append('/home/amsequeira/deepbio')
from src.mlmodels.utils_run_models import divide_dataset, binarize_labels
from src.mlmodels.class_TrainEvalModel import ModelTrainEval
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from collections import Counter
# from prettytable import PrettyTable
from IPython.display import Image

from sklearn.preprocessing import LabelEncoder

from keras.models import Model
from keras.regularizers import l2
from keras.constraints import max_norm
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, Add, MaxPooling1D, BatchNormalization
from keras.layers import Embedding, Bidirectional, LSTM, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
import collections
import logging
from sklearn.metrics import accuracy_score, f1_score, log_loss, matthews_corrcoef, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.metrics import average_precision_score
logging.getLogger("tensorflow").setLevel(logging.FATAL)

tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()


# https://github.com/ronakvijay/Protein_Sequence_Classification/blob/master/Pfam_protein_sequence_classification.ipynb
# https://www.kaggle.com/omarkhald/protein-sequence-classification
# https://www.kaggle.com/googleai/pfam-seed-random-split/notebooks
# file:///C:/Users/anama/Documents/work/DEEPBIO/deeplearning_annotate_protein_universe.pdf
# https://www.youtube.com/watch?v=x-35bDrKfHA&ab_channel=GoogleDevelopers


# Data Overview
# sequence: These are usually the input features to the model.
# Amino acid sequence for this domain. There are 20 very common amino acids (frequency > 1,000,000),
# and 4 amino acids that are quite uncommon: X, U, B, O, Z.

# family_accession: These are usually the labels for the model. Accession number in form PFxxxxx.y (Pfam), where xxxxx is the family accession, and y is the version number. Some values of y are greater than ten, and so 'y' has two digits.
#
# EC number: Labels for the model X.X.X.X has 4 distant numbers. first division has 7 classes


# We have been provided with already done random split(train, val, test) of pfam dataset.
# Train - 80% (For training the models).
# Val - 10% (For hyperparameter tuning/model validation).
# Test - 10% (For acessing the model performance).


# 2.2.1 Type of Machine learning Problem
# It is a multi class classification problem, for a given sequence of amino acids we need to predict its family accession.
#
# 2.2.2 Performance Metric
# Multi class log loss
# Accuracy

def get_list_first_ec_numbers_no_duplicates(ec_number):
    # it get a list of all the first level of ec number of entries. if it has more than one it wills ave both.
    # it will eliminate the duplicates if the same first level of ec number is equal
    # ec_number2=ec_number.replace(['non_Enzyme'], ['0.0'])
    # separate values that have more than one ECnumber (list of ec numbers)

    ec_multiple_list = ec_number.str.split(';')
    # transform nan in list with 0
    ec_multiple_list = ec_multiple_list.apply(lambda d: d if isinstance(d, list) else ['0.0'])

    # get list with lists of values
    firsts = []
    for x in ec_multiple_list:
        [w.replace(" ", "") for w in x]
        first = [w.strip()[0] for w in x]  # list of each sequence
        firsts.append(first)

    # take out this duplicates to get only numbers different. still in alist of list format
    firsts_no_duplicates = []
    for x in firsts:
        first = (set(x))
        firsts_no_duplicates.append(list(first))
    return firsts_no_duplicates


def single_label(firsts_no_duplicates, fps_x, count_negative_as_class=True):
    fps_y = pd.DataFrame(firsts_no_duplicates)
    # x=0
    # for column in range(len(fps_y.columns)):
    #     fps_y.columns = ['ec_number'+str(x) for col_name in fps_y.columns]
    #     x+=1
    columns = ['ec_number1', 'ec_number2', 'ec_number3']  # 284010
    fps_y.columns = columns

    # if classes does not include de zeros (no enzymes)
    if count_negative_as_class is False:
        fps_y = fps_y[fps_y['ec_number1'] != '0']
    # fps_y = fps_y.drop(fps_y[fps_y.ec_number1 == '0'].index)

    # count how many ec_number2 are NAn (more than 1 EC number to verify shape)
    #  print(fps_y['ec_number2'].isna().sum()) # 284009 (1 multiple EC) 283283 ( 727 more than 1 ec number)
    fps_x = pd.DataFrame(fps_x)
    df = fps_y.join(fps_x)
    df = df.dropna(thresh=2)  # at least 2 Nan
    from sklearn.utils import shuffle
    df = shuffle(df)

    # take out the ones that have multiple ec numbers
    new_dataset = df[df[['ec_number2']].isna().any(axis=1)]  # 283283
    fps_y = new_dataset['ec_number1']
    # fps_x=new_dataset.drop(['ec_number1','ec_number2','ec_number3'], axis=1)
    fps_x = new_dataset.loc[:, ~new_dataset.columns.isin(['ec_number1', 'ec_number2', 'ec_number3'])]
    # fps_y = [(map(int, x)) for x in fps_y]
    fps_y = [int(row) for row in fps_y]

    return fps_y, fps_x


########################################################################################################################
# EXPLORATORY DATA ANALYSIS
########################################################################################################################

# lOAD DATA INTO DEV TRAIN TEST
from src.mlmodels.run_deep_model import get_hot_encoded_sequence, get_final_fps_x_y, \
    get_parameter_hot_encoded

ecpred_uniref_90 = '/home/amsequeira/deepbio/datasets/ecpred/ecpred_uniprot_uniref_90.csv'
source_file = pd.read_csv(ecpred_uniref_90, low_memory=False)

file = source_file[source_file.sequence.str.contains(
    '!!!') == False]  # because 2 sequences in ecpred had !!!! to long to EXCEL... warning instead of sequence
file = file[file.notna()]  # have some nas (should be taken out in further datasets)

fps_y = file['ec_number']
fps_y.fillna(0, inplace=True)
fps_x = file[['sequence', 'uniref_90']].dropna()

firsts_no_duplicates = get_list_first_ec_numbers_no_duplicates(fps_y)
fps_y, fps_x = single_label(firsts_no_duplicates, fps_x, count_negative_as_class=False)

x_train_1, x_test, y_train_1, y_test = train_test_split(fps_x, fps_y, test_size=0.10, random_state=42,
                                                        stratify=fps_y)
x_train, x_val, y_train, y_val = train_test_split(x_train_1, y_train_1, test_size=0.11, random_state=42,
                                                  stratify=y_train_1)


def size_data():
    # Given data size
    print('Train size: ', len(x_train))
    print('Val size: ', len(x_val))
    print('Test size: ', len(x_test))
    print(x_train.info())


# # Train size:  121553
# # Val size:  15024
# # Test size:  15176
#
# # Data columns (total 2 columns):
# # #   Column     Non-Null Count   Dtype
# # ---  ------     --------------   -----
# # 0   sequence   121553 non-null  object
# # 1   uniref_90  121553 non-null  object
# # dtypes: object(2)
# # memory usage: 2.8+ MB

def class_count():
    # class count
    # take in consideration that they do not divide in train test set but rather different datasets and
    # counts classes on them

    len(source_file['ec_number'].unique())  # 5315 classes (4 EC numbers) in total dataset

    len(np.unique(y_train))  # 7 pq dividi logo as classes todas
    len(np.unique(y_test))
    len(np.unique(y_val))


def sequence_counts():
    # sequence counts
    # Length of sequence in train data.
    x_train['seq_char_count'] = x_train['sequence'].apply(lambda x: len(x))
    x_val['seq_char_count'] = x_val['sequence'].apply(lambda x: len(x))
    x_test['seq_char_count'] = x_test['sequence'].apply(lambda x: len(x))


def plot_seq_count(df, data_name):
    sns.distplot(df['seq_char_count'].values)
    plt.title(f'Seq char count: {data_name}')
    plt.grid(True)


def plot_seq_count_three_df():
    plt.subplot(1, 3, 1)
    plot_seq_count(x_train, 'Train')

    plt.subplot(1, 3, 2)
    plot_seq_count(x_val, 'Val')

    plt.subplot(1, 3, 3)
    plot_seq_count(x_test, 'Test')

    plt.subplots_adjust(right=3.0)
    plt.show()


# # Most of the unaligned amino acid sequences have character counts in the range of 50-1000.
# # in contrary of the paper much more range of len sequences (paper go around 2000 aa lenght
# # here, sequences with 10000 aa are retrieved


# ### Sequence Code Frequency
# # Amino acid sequences are represented with their corresponding 1 letter code.
# # The complete list of amino acids with there code can be found here.
def get_code_freq(df, data_name):
    df = df.apply(lambda x: " ".join(x))

    codes = []
    for i in df:  # concatenation of all codes
        codes.extend(i)

    codes_dict = Counter(codes)
    codes_dict.pop(' ')  # removing white space

    print(f'Codes: {data_name}')
    print(f'Total unique codes: {len(codes_dict.keys())}')

    df = pd.DataFrame({'Code': list(codes_dict.keys()), 'Freq': list(codes_dict.values())})
    return df.sort_values('Freq', ascending=False).reset_index()[['Code', 'Freq']]


#
# train_code_freq = get_code_freq(x_train['sequence'], 'Train')
# print(train_code_freq)
#
# # Code     Freq
# # 0     L  4875019
# # 1     A  4317357
# # 2     G  3741352
# # 3     V  3554985
# # 4     E  3378716
# # 5     S  3135108
# # 6     I  3071495
# # 7     D  2875165
# # 8     K  2795866
# # 9     R  2776053
# # 10    T  2638318
# # 11    P  2370173
# # 12    N  1989622
# # 13    F  1975864
# # 14    Q  1820636
# # 15    Y  1523917
# # 16    M  1203299
# # 17    H  1186299
# # 18    C   660604
# # 19    W   576484
# # 20    X      940
# # 21    U      119
# # 22    B       20
# # 23    Z       15
# # 24    O        8
# # Codes: Train
# # Total unique codes: 25
#
# val_code_freq = get_code_freq(x_val['sequence'], 'Val')
# print(train_code_freq)
# # Codes: val
# # Total unique codes: 22
# test_code_freq = get_code_freq(x_test['sequence'], 'Test')
# print(train_code_freq)
# # Codes: test
# # Total unique codes: 22
#
# # for x_test and x_val is the exact same order of most appearing aminoacids, but U is the last aa with counts around 12
# # nno B Z O for x_test and x_val
#
def plot_code_freq(df, data_name):
    plt.title(f'Code frequency: {data_name}')
    sns.barplot(x='Code', y='Freq', data=df)


def plot_code_freq_three_df(train_code_freq, val_code_freq, test_code_freq):
    plt.subplot(1, 3, 1)
    plot_code_freq(train_code_freq, 'Train')

    plt.subplot(1, 3, 2)
    plot_code_freq(val_code_freq, 'Val')

    plt.subplot(1, 3, 3)
    plot_code_freq(test_code_freq, 'Test')

    plt.subplots_adjust(right=3.0)
    plt.show()


#
# # The exact same observations that the paper apply here
# # Most frequent amino acid code is L followed by A, V, G.
# # As we can see, that the uncommon amino acids (i.e., X, U, B, O, Z) are present in very less quantity.
# # Therefore we can consider only 20 common natural amino acids for sequence encoding.
#

# # Protein families with most sequences(No. of observations)
def protein_families():
    source_file.groupby('ec_number').size().sort_values(ascending=False).head(20)


# # ec_number
# # 3.1.-.-     2395
# # 7.1.1.-     2237
# # 2.7.11.1    1986
# # 2.7.7.6     1937
# # 2.1.1.-     1529
# # 3.6.4.12    1341
# # 3.6.4.13    1090
# # 3.4.24.-     972
# # 5.2.1.8      968
# # 7.1.2.2      947
# # 2.3.2.27     897
# # 7.1.1.2      829
# # 2.3.1.-      768
# # 6.3.5.-      756
# # 3.4.21.-     710
# # 3.6.5.-      692
# # 3.1.26.4     687
# # 1.-.-.-      678
# # 2.4.1.-      643
# # 4.2.1.33     603
#
# # just for the 7 classes
# pd.DataFrame(collections.Counter(y_train).most_common()).set_index(0)
#
# 2  43441
# 3  27562
# 1  15231
# 6  13032
# 4  10337
# 5   6465
# 7   5485

# y_test
# 2  5423
# 3  3441
# 1  1902
# 6  1627
# 4  1291
# 5   807
# 7   685

# y_val
# 2  5369
# 3  3406
# 1  1883
# 6  1611
# 4  1278
# 5   799
# 7   678

# when do with all classes check if the 20 frequencies are the same across classes (random split)
partitions = {'test': x_test, 'dev': x_val, 'train': x_train}


def plot_family_sizes(partitions):
    for name, partition in partitions.items():
        partition.groupby('family_id').size().hist(bins=50)
    plt.title('Distribution of family sizes for %s' % name)
    plt.ylabel('# Families')
    plt.xlabel('Family size')
    plt.show()


# todo they do an alignment of sequences check biopython
# the aligned sequences are somethng like PPALMDLCA.......LAIQQ.HLGQQRHN.............QI....
# aligned_sequence: Contains a single sequence from the multiple sequence alignment (with the rest of the members of
# the family in seed, with gaps retained.

def find_the_family_with_longest_alignment(df):
    df['alignment_length'] = df.aligned_sequence.str.len()
    df.alignment_length.hist(bins=30)
    plt.title('Distribution of alignment lengths')
    plt.xlabel('Alignment length')
    plt.ylabel('Number of sequences')

# now, they limit to the top 1000 classes due to computational power
########################################################################################################################
########################################################################################################################
# 4. Deep Learning Models
########################################################################################################################
########################################################################################################################
# Text Preprocessing
# https://dmnfarrell.github.io/bioinformatics/mhclearning
# http://www.cryst.bbk.ac.uk/education/AminoAcid/the_twenty.html
# 1 letter code for 20 natural amino acids

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


def create_dict(codes):
    char_dict = {}
    for index, val in enumerate(codes):
        char_dict[val] = index + 1

    return char_dict


char_dict = create_dict(codes)

print(char_dict)
print("Dict Length:", len(char_dict))


def integer_encoding(data):
    """
    - Encodes code sequence to integer values.
    - 20 common amino acids are taken into consideration
      and rest 4 are categorized as 0.
    """

    encode_list = []
    for row in data['sequence'].values:
        row_encode = []
        for code in row:
            row_encode.append(char_dict.get(code, 0))
        encode_list.append(np.array(row_encode))

    return encode_list

# encode x (letter aa --> number)
train_encode = integer_encoding(x_train)
val_encode = integer_encoding(x_val)
test_encode = integer_encoding(x_test)

# padding sequences
max_length = 700
train_pad = pad_sequences(train_encode, maxlen=max_length, padding='post', truncating='post')
val_pad = pad_sequences(val_encode, maxlen=max_length, padding='post', truncating='post')
test_pad = pad_sequences(test_encode, maxlen=max_length, padding='post', truncating='post')

# # train_pad.shape, val_pad.shape, test_pad.shape
# # ((121553, 100), (15024, 100), (15176, 100))


# One hot encoding of sequences
train_ohe = to_categorical(train_pad)
val_ohe = to_categorical(val_pad)
test_ohe = to_categorical(test_pad)

# # train_ohe.shape, test_ohe.shape, test_ohe.shape
# # ((121553, 100, 21), (15176, 100, 21), (15176, 100, 21))


# # # label/integer encoding output variable: (y)
le = LabelEncoder()

# y_train_le = le.fit_transform(train_sm['family_accession'])
# y_val_le = le.transform(val_sm['family_accession'])
# y_test_le = le.transform(test_sm['family_accession'])
# this part i didnt do because separate all the datasets but try as this way !!!!
# i already have encoded as the EC number
# y_train_le = le.fit_transform(y_train)
# y_val_le = le.transform(y_val)
# y_test_le = le.transform(y_test)

# One hot encoding of outputs
y_train_ohe = to_categorical(y_train)
y_val_ohe = to_categorical(y_val)
y_test_ohe = to_categorical(y_test)

# y_train.shape, y_val.shape, y_test.shape
# ((121553, 8), (15024, 8), (15176, 8))

# Utility function: plot model's accuracy and loss

# https://realpython.com/python-keras-text-classification/
plt.style.use('ggplot')
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


# Utility function: Display model score(Loss & Accuracy) across all sets.
def display_model_score(model, train, val, test, batch_size):
    train_score = model.evaluate(train[0], train[1], batch_size=batch_size, verbose=1)
    print('Train loss: ', train_score[0])
    print('Train accuracy: ', train_score[1])
    print('-' * 70)

    val_score = model.evaluate(val[0], val[1], batch_size=batch_size, verbose=1)
    print('Val loss: ', val_score[0])
    print('Val accuracy: ', val_score[1])
    print('-' * 70)

    test_score = model.evaluate(test[0], test[1], batch_size=batch_size, verbose=1)
    print('Test loss: ', test_score[0])
    print('Test accuracy: ', test_score[1])


def bidilstm(input_dim, final_units):
    # Model 1: Bidirectional LSTM
    x_input = Input(shape=(100,))
    emb = Embedding(21, 128, input_length=input_dim)(x_input)
    # bi_rnn = Bidirectional(CuDNNLSTM(64, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))(emb)
    # CuDNNLSTM was giving problems and isthe same as LSTM in newer versions
    bi_rnn = Bidirectional(
        LSTM(64, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))(emb)

    x = Dropout(0.3)(bi_rnn)

    # softmax classifier add dense different 8 classes instead of 1000
    x_output = Dense(final_units, activation='softmax')(x)

    model1 = Model(inputs=x_input, outputs=x_output)
    model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model1.summary()
    return model1


#
# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_3 (InputLayer)         [(None, 100)]             0
# _________________________________________________________________
# embedding_2 (Embedding)      (None, 100, 128)          2688
# _________________________________________________________________
# bidirectional (Bidirectional (None, 128)               98816
# _________________________________________________________________
# dropout (Dropout)            (None, 128)               0
# _________________________________________________________________
# dense (Dense)                (None, 1000)              129000
# =================================================================
# Total params: 230,504
# Trainable params: 230,504
# Non-trainable params: 0
# _________________________________________________________________
input_dim=max_length
final_units=len(y_train_ohe[0])
model1=bidilstm(input_dim, final_units)
# Early Stopping
es = EarlyStopping(monitor='val_loss', patience=30, verbose=1)

history1 = model1.fit(
    train_pad, np.array(y_train_ohe),
    epochs=100, batch_size=256,
    validation_data=(val_pad, np.array(y_val_ohe)),
    callbacks=[es]
)

# # saving model weights.
# model1.save_weights('/home/amsequeira/deepbio/mimic_annotate_protein_universe_model1.h5')
plot_history(history1)

display_model_score(model1,
                    [train_pad, y_train],
                    [val_pad, y_val],
                    [test_pad, y_test],
                    256)

# c 100 len seq e 128 output embedding accuracy around 0.35
# c 1000            256                                 0.28
# tentar c mais parametros. patience len sequences....

# ######################################################################################################################
# Model 2: ProtCNN (https://www.biorxiv.org/content/10.1101/626507v4.full)
# One hot encoded unaligned sequence of amino acids is passed as the input to the network with zero padding.
# This network uses residual blocks inspired from ResNet architecture which also includes dilated convolutions
# offering larger receptive field without increasing number of model parameters.
########################################################################################################################

def residual_block(data, filters, d_rate):
    """
    _data: input
    _filters: convolution filters
    _d_rate: dilation rate
    """

    shortcut = data

    bn1 = BatchNormalization()(data)
    act1 = Activation('relu')(bn1)
    conv1 = Conv1D(filters, 1, dilation_rate=d_rate, padding='same', kernel_regularizer=l2(0.001))(act1)

    # bottleneck convolution
    bn2 = BatchNormalization()(conv1)
    act2 = Activation('relu')(bn2)
    conv2 = Conv1D(filters, 3, padding='same', kernel_regularizer=l2(0.001))(act2)

    # skip connection
    x = Add()([conv2, shortcut])

    return x


def prot_cnn(input_dim, number_classes):
    with strategy.scope():
        # model
        x_input = Input(shape=(int(input_dim), 21))
        # initial conv
        conv = Conv1D(128, 1, padding='same')(x_input)

        # per-residue representation
        res1 = residual_block(conv, 128, 2)
        res2 = residual_block(res1, 128, 3)

        x = MaxPooling1D(3)(res2)
        x = Dropout(0.5)(x)

        # softmax classifier
        x = Flatten()(x)
        x_output = Dense(number_classes, activation='softmax', kernel_regularizer=l2(0.0001))(x)

        model2 = Model(inputs=x_input, outputs=x_output)
        model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # model2.summary()
        return model2


# Early Stopping
# es = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
#
# history2 = model2.fit(
#     train_ohe, y_train,
#     epochs=10, batch_size=256,
#     validation_data=(val_ohe, y_val),
#     callbacks=[es]
# )
#
# # saving model weights.
# model2.save_weights('/home/amsequeira/deepbio/mimic_annotate_protein_universe_model2.h5')
# plot_history(history2)
#
#
# display_model_score(
#     model2,
#     [train_ohe, y_train],
#     [val_ohe, y_val],
#     [test_ohe, y_test],
#     256)

def run_protcnn(model_name, max_length=100):
    ecpred_uniref_90 = '/home/amsequeira/deepbio/datasets/ecpred/ecpred_uniprot_uniref_90.csv'
    source_file = pd.read_csv(ecpred_uniref_90, low_memory=False)

    file = source_file[source_file.sequence.str.contains(
        '!!!') == False]  # because 2 sequences in ecpred had !!!! to long to EXCEL... warning instead of sequence
    file = file[file.notna()]  # have some nas (should be taken out in further datasets)

    fps_y = file['ec_number']
    fps_y.fillna(0, inplace=True)
    fps_x = file[['sequence', 'uniref_90']].dropna()

    firsts_no_duplicates = get_list_first_ec_numbers_no_duplicates(fps_y)
    fps_y, fps_x = single_label(firsts_no_duplicates, fps_x, count_negative_as_class=False)

    x_train_1, x_test, y_train_1, y_test = train_test_split(fps_x, fps_y, test_size=0.10, random_state=42,
                                                            stratify=fps_y)
    x_train, x_val, y_train, y_val = train_test_split(x_train_1, y_train_1, test_size=0.11, random_state=42,
                                                      stratify=y_train_1)

    # get x_train like they do

    train_encode = integer_encoding(x_train)
    val_encode = integer_encoding(x_val)
    test_encode = integer_encoding(x_test)

    # padding sequences
    train_pad = pad_sequences(train_encode, maxlen=max_length, padding='post', truncating='post')
    val_pad = pad_sequences(val_encode, maxlen=max_length, padding='post', truncating='post')
    test_pad = pad_sequences(test_encode, maxlen=max_length, padding='post', truncating='post')

    # One hot encoding of sequences
    train_ohe = to_categorical(train_pad)
    val_ohe = to_categorical(val_pad)
    test_ohe = to_categorical(test_pad)

    # todo attention that they do this way and use categorical crossentropy ( hot encoded labels and not integers)
    # important if consider more than one class
    # y_train = to_categorical(y_train)
    # y_val = to_categorical(y_val)
    # y_test = to_categorical(y_test)
    # print(y_train.shape)

    # Early Stopping
    es = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    model1 = prot_cnn(max_length,len(np.unique(y_train)))
    history1 = model1.fit(
        np.array(train_ohe), np.array(y_train),
        epochs=50, batch_size=256,
        validation_data=(np.array(val_ohe), np.array(y_val)),
        callbacks=[es])



    train = ModelTrainEval(train_ohe, y_train, test_ohe, y_test, val_ohe, y_val, validation_split=0.25,
                           epochs=500, callbacks=None, reduce_lr=True, early_stopping=True, checkpoint=True,
                           early_stopping_patience=int(30),
                           reduce_lr_patience=50, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
                           path='', save=False, plot_model_file='model_plot.png',
                           verbose=2, batch_size=128)

    # if model_name == 'prot_cnn':
    #     protcnn = KerasClassifier(build_fn=prot_cnn, number_classes=len(np.unique(y_train)),
    #                               input_dim=max_length, batch_size=128)
    #     model, history = train.run_model(protcnn)
    #     plot_history(history)
    #
    # elif model_name == 'bidilstm':
    #     bilstm = KerasClassifier(build_fn=bidilstm, number_classes=len(np.unique(y_train)),
    #                              input_dim=max_length, batch_size=128)
    #
    #     model, history = train.run_model(bilstm)
    #     plot_history(history)

    train.model_simple_evaluate(x_test,y_test)
    train.print_evaluation(x_test, y_test)
    # try different optimizers? and other parameters?


# if __name__ == '__main__':
    # run_protcnn('prot_cnn', max_length=500)
    # run_protcnn('bidilstm', max_length = 100)
