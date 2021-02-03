import sys
import os
import pandas as pd
import numpy as np
from tensorflow import keras
import operator
import random
from random import sample
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
# logging.disable(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
import tensorflow as tf
from itertools import chain
from sklearn.feature_selection import mutual_info_classif

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
from collections import Counter
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
import re
from skmultilearn.model_selection import iterative_train_test_split
sys.path.append('/home/amsequeira/propythia/propythia')
sys.path.append('/home/amsequeira/propythia')
sys.path.append('/home/amsequeira/propythia/propythia/src')
sys.path.append('/home/amsequeira/deepbio/datasets')
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
from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import LSTM, TimeDistributed
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool1D, MaxPool2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from propythia.deep_ml import DeepML
from propythia.manifold import Manifold
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, TimeDistributed, Bidirectional, Concatenate
from tensorflow.keras.models import Model
from propythia.adjuv_functions.ml_deep.utils import divide_dataset, binarize_labels
from propythia.deep_ml import DeepML
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from propythia.feature_selection import FeatureSelection

###################################################################################################

########################################################################################################################
################################################## FOR Y ###############################################################
########################################################################################################################

# # select rows with complete ec numbers
# len(data['ec_number'].unique()) # 5315 different classes counts agglomerates
# data.groupby('ec_number').size().sort_values(ascending=False).head(20) # know which classe are most representative

# 1.Top 1000 classes complete classes with only 15. 800 with more than  MULTILABEL
def get_ec_complete_more_than_x_samples(data, x=50, single_label=True):
    # get only EC COMPLETE
    l = []
    for ec_list in data['ec_number']:
        ec_complete = [x.strip() for x in ec_list.split(';') if "-" not in x]
        l.append(list(set(ec_complete)))

    data['ec_number4'] = l
    # remove lines without ec complete
    data = data.loc[data['ec_number4'].apply(len)>0,:]  # (153672, 59)

    if single_label:
        data = turn_single_label('ec_number4', data)
    else:
        pass

    # result = {x for l in data['ec_number4'] for x in l} #4603 differennt unique values
    # result_2 = set(chain(*data['ec_number4'])) # 4603 unique values
    counts = Counter(x for xs in data['ec_number4'] for x in set(xs))
    counts.most_common()
    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    df_sorted = df.sort_values(by=[0], ascending=False)

    # # + de 1000 -
    # df_1000 = df.loc[df[0]>1000] # 5
    # # + de 500 -
    # df_500 = df.loc[df[0]>500] #20
    # # + de 300 -
    # df_300 = df.loc[df[0]>300] #115
    # # + de 100
    # df_100 = df.loc[df[0]>100] # 344
    # # + de 50 -
    # df_50 = df.loc[df[0]>50] # 536
    # # + de 30
    # df_30 = df.loc[df[0]>300] # 709
    # # + de 15
    # df_15 = df.loc[df[0]>15] #1069

    # + de x samples -
    df_15 = df.loc[df[0]>x] # 5
    #select the EC numbers that have more than ....
    l = []
    for ec_list in data['ec_number4']:
        ec = [x for x in ec_list if x in (list(df_15['index']))]
        l.append(ec)
    data['ec_number4'] = l # data 153672
    data = data.loc[data['ec_number4'].apply(len)>0,:] #142092
    counts = Counter(x for xs in data['ec_number4'] for x in set(xs))
    counts.most_common()

    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    df_sorted = df.sort_values(by=[0], ascending=False)
    print(df_sorted)
    return data   # column to consider is 'ec_number

##############
# get all the 3º level ec numbers complete with more than x samples
def get_ec_3_level_more_than_x_samples(data, x, single_label=True):
    # get all until the 3 level (everything until the last dot    175805
    data = data.dropna(subset=['ec_number'])
    l = []
    for ec_list in data['ec_number']:
        print(ec_list)
        ec_3 = [re.match(r'.*(?=\.)',x).group(0) for x in ec_list.split(';') ]
        l.append(list(set(ec_3)))
    data['ec_number3']=l
    # get only 3º level complete (remove dashes)
    l = []
    for ec_list in data['ec_number3']:
        ec_complete = [x.strip() for x in ec_list if "-" not in x]
        l.append(ec_complete)

    data['ec_number3'] = l
    # remove lines without ec complete
    data = data.loc[data['ec_number3'].apply(len)>0,:]  # 170191

    if single_label:
        data = turn_single_label('ec_number3', data)
    else:
        pass


    counts = Counter(x for xs in data['ec_number3'] for x in set(xs))
    counts.most_common()

    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()  #256 classes
    df_sorted = df.sort_values(by=[0], ascending=False)


    # # + de 1000 -
    # df_1000 = df.loc[df[0]>1000] # 41
    # # + de 500 -
    # df_500 = df.loc[df[0]>500] # 67
    # # + de 300 -
    # df_300 = df.loc[df[0]>300] #95
    # # + de 100
    # df_100 = df.loc[df[0]>100] # 131
    # # + de 50 -
    # df_50 = df.loc[df[0]>50] # 160
    # # + de 30
    # df_30 = df.loc[df[0]>300] # 95
    # # + de 15
    # df_15 = df.loc[df[0]>15] #199

    # - de 15
    # df_15 = df.loc[df[0]<15] 55


    # + de x samples -
    df_15 = df.loc[df[0]>x]

    #select the EC numbers that have more than ....
    l = []
    for ec_list in data['ec_number3']:
        ec = [x for x in ec_list if x in(list(df_15['index']))]
        l.append(ec)
    data['ec_number3'] = l # data 153672
    data = data.loc[data['ec_number3'].apply(len)>0,:] #142092
    return data

##############
# get all the 2º level ec numbers complete with more than x samples
def get_ec_2_level_more_than_x_samples(data, x, single_label=True):
    # get all until the 2 level (everything until the last dot    175805
    l = []
    for ec_list in data['ec_number']:
        ec_2 = [re.search(r'[^.]*.[^.]*',x).group(0) for x in ec_list.split(';') ]
        # [^,]* = as many non-dot characters as possible,
        # . = a dot
        # [^.]* = as many non-dot characters as possible
        l.append(list(set(ec_2)))
    data['ec_number2']=l

    # get only 2º level complete (remove dashes)
    l = []
    for ec_list in data['ec_number2']:
        ec_complete = [x.strip() for x in ec_list if "-" not in x]
        l.append(ec_complete)

    data['ec_number2'] = l
    # remove lines without ec complete
    data = data.loc[data['ec_number2'].apply(len)>0,:]  # 174521

    if single_label:
        data = turn_single_label('ec_number2', data)
    else:
        pass


    counts = Counter(x for xs in data['ec_number2'] for x in set(xs))
    counts.most_common()

    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()  #73 classes
    df_sorted = df.sort_values(by=[0], ascending=False)


    # # + de 1000 -
    # df_1000 = df.loc[df[0]>1000] # 27
    # # + de 500 -
    # df_500 = df.loc[df[0]>500] # 39
    # # + de 300 -
    # df_300 = df.loc[df[0]>300] # 45
    # # + de 100
    # df_100 = df.loc[df[0]>100] # 55
    # # + de 50 -
    # df_50 = df.loc[df[0]>50] # 60
    # # + de 30
    # df_30 = df.loc[df[0]>300] # 45
    # # + de 15
    # df_15 = df.loc[df[0]>15] # 64

    # - de 15
    # df_15 = df.loc[df[0]<15] 7


    # + de x samples -
    df_x = df.loc[df[0]>x]

    #select the EC numbers that have more than ....
    l = []
    for ec_list in data['ec_number2']:
        ec = [x for x in ec_list if x in(list(df_x['index']))]
        l.append(ec)
    data['ec_to_keep'] = l # data 153672
    data = data.loc[data['ec_to_keep'].apply(len)>0,:]
    return data


##############
# get all the 1º level ec numbers complete
def get_ec_1_level(data, single_label=True):
    # get all until the 1 level (everything until the last dot    175805
    l = []
    for ec_list in data['ec_number']:
        ec_1 = [x.strip()[0] for x in ec_list.split(';') ]
        # [^,]* = as many non-dot characters as possible,
        # . = a dot
        l.append(list(set(ec_1)))
    data['ec_number1']=l
    if single_label:
        data = turn_single_label('ec_number1', data)
    else:
        pass

    counts = Counter(x for xs in data['ec_number1'] for x in set(xs))
    counts.most_common()
    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    df_sorted = df.sort_values(by=[0], ascending=False)

    # [('2', 54508),
    #  ('3', 34838),
    #  ('0', 23245),
    #  ('1', 19200),
    #  ('6', 16307),
    #  ('4', 13163),
    #  ('5', 8105),
    #  ('7', 6957)]

    return data

#############################3
# get all 4 levels independently 2º, 3º 4º
def get_n_ec_level(n, data, x, single_label=True):
    # get all n level
    n-=1
    l = []
    for ec_list in data['ec_number']:
        ec_l = [x.split('.')[n] for x in ec_list.split(';')]
        l.append(ec_l)
    data['ec_level']=l

    # get only level complete (remove dashes)
    l = []
    for ec_list in data['ec_level']:
        ec_complete = [x.strip() for x in ec_list if "-" not in x]
        l.append(ec_complete)

    data['ec_level'] = l
    # remove lines without ec complete
    data = data.loc[data['ec_level'].apply(len)>0,:]

    if single_label:
        data = turn_single_label('ec_level', data)
    else:
        pass

    #most common
    counts = Counter(x for xs in data['ec_level'] for x in set(xs))
    counts.most_common()
    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()  #73 classes
    df_sorted = df.sort_values(by=[0], ascending=False)

    # + de x samples -
    df_x = df.loc[df[0]>x]

    #select the EC numbers that have more than ....
    l = []
    for ec_list in data['ec_level']:
        ec = [x for x in ec_list if x in(list(df_x['index']))]
        l.append(ec)
    data['ec_to_keep'] = l # data 153672
    data = data.loc[data['ec_to_keep'].apply(len)>0,:] #142092
    return data

# PREDICTING THE SECOND KNOWING THE FIRST SHUFFLE THE REST OF DATASET FOR NEGATIVES
def get_second_knowing_first(data, first_level, single_label=True):
    # get rows with first level
    l = []
    for ec_list in data['ec_number']:
        ec_1 = [x.strip()[0] for x in ec_list.split(';') ]
        l.append(list(set(ec_1)))
    data['ec_number1']=l

    # t exclude the rows without the first level desired
    l = []
    for ec_list in data['ec_number1']:
        ec_1 = [x for x in ec_list if x == str(first_level)]
        l.append(list(set(ec_1)))
    data['ec_number1']=l
    data_negative = data.loc[data['ec_number1'].apply(len)<1,:]
    data_positive = data.loc[data['ec_number1'].apply(len)>0,:]

    # extract the second digit
    l=[]
    for ec_list in data_positive['ec_number']:
        ec_l = [x.split('.')[1] for x in ec_list.split(';')]
        l.append(list(set(ec_l)))
    data_positive['ec_number2']=l

    # remove dashes
    l = []
    for ec_list in data_positive['ec_number2']:
        ec_complete = [x.strip() for x in ec_list if "-" not in x]
        l.append(ec_complete)

    data_positive['ec_number2'] = l
    if single_label:
        data = turn_single_label('ec_number2', data)
    else:
        pass

    # for dataset negative,
    # decide how much negatives
    # to decide how much negatives
    counts = Counter(x for xs in data_positive['ec_number2'] for x in set(xs))
    x = counts.most_common()
    n = int(data_positive.shape[0]/len(x)) # the number negatives has the len of dataset divided by number of classes as if it was a even distribution, although this do not happen in the classes
    # n = x[3][1] # negatives has the third most class

    #create a dataset negative
    data_negative = data_negative.sample(n=n, random_state=5)
    data_negative['ec_number2'] = '0'
    #join the datasets
    final_data = pd.concat([data_positive,data_negative])
    return final_data # column to consider is 'ec_number2


########################################################################################################################
################################################## FOR X ###############################################################
########################################################################################################################
def hot_encoded_sequence(data, column_sequence, seq_len, alphabet, padding_truncating='post'):
    data = data[data[column_sequence].str.contains('!!!') == False]  # because 2 sequences in ecpred had !!!! to long to EXCEL... warning instead of sequence
    data = data[data.notna()]  # have some nas (should be taken out in further datasets)
    sequences = data[column_sequence].tolist()

    sequences_integer_ecoded = []
    for seq in sequences:
        if len(alphabet) < 25:  # alphabet x or alphabet normal
            seq1 = seq.replace('B', 'N')  # asparagine N / aspartic acid  D - asx - B
            seq2 = seq1.replace('Z', 'Q')  # glutamine Q / glutamic acid  E - glx - Z
            seq3 = seq2.replace('U',
                                'C')  # selenocisteina, the closest is the cisteine. but it is a different aminoacid . take care.
            seq4 = seq3.replace('O', 'K')  # Pyrrolysine to lysine
            seq = seq4
            if len(alphabet) == 20:  # alphabet normal substitute every letters
                seq = seq4.replace('X', '')  # unknown character eliminated

        # define a mapping of chars to integers
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        int_to_char = dict((i, c) for i, c in enumerate(alphabet))
        # integer encode input data
        integer_encoded = [char_to_int[char] for char in seq]
        sequences_integer_ecoded.append(integer_encoded)

    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences
    # pad sequences
    # todo pad in middle
    # pad inside the batch ?
    list_of_sequences_length = pad_sequences(sequences_integer_ecoded, maxlen=seq_len, dtype='int32',
                                             padding=padding_truncating, truncating=padding_truncating, value=0.0)

    # one hot encoding
    shape_hot = len(alphabet) * seq_len  # 20000
    encoded = to_categorical(list_of_sequences_length) # shape (samples, 1000,20)
    fps_x = encoded.reshape(encoded.shape[0], shape_hot)  # shape (samples, 20000)
    return data, fps_x, encoded

def hot_encoded_families(data, column_parameter='Cross-reference (Pfam)'):
    # divide parameter
    families = data[column_parameter]

    # remove columns with parameter NAn
    data2 = data[data[column_parameter].notna()] # from 175267 to 167638

    fam = [i.split(';') for i in data2[column_parameter]]  # split dos Pfam 'PF01379;PF03900;'
    fam = [list(filter(None, x)) for x in fam]  # remove '' empty string (because last ;
    # fam = [set(x) for x in fam]

    mlb = MultiLabelBinarizer()
    fam_ho = mlb.fit_transform(fam)
    classes = mlb.classes_
    len(classes)
    return data2, fam_ho, classes

def physchemical(data):
    not_x = ['ec_number_ecpred', 'uniprot', 'sequence', 'uniref_90', 'ec_number']
    fps_x = data.drop(not_x, axis=1)
    columns = fps_x.columns
    return data, fps_x, columns

def nlf(data, seq_len, padding_truncating='post'):
    # transform nlf string to dataframe
    max_length = seq_len * 18 # size of parameters for each aa
    nlf = data['nlf']
    fps_x_encoded = []
    for line in nlf:
        line = [float(x) for x in line.split(',')]
        fps_x_encoded.append(line)

    # pad sequences
    fps_x_nlf = pad_sequences(fps_x_encoded, maxlen=max_length, padding=padding_truncating, truncating=padding_truncating, dtype='float32')
    return data, fps_x_nlf

def blosum(data, seq_len, padding_truncating='post'):
    max_length = seq_len * 24 # size of parameters for each aa
    blosum = data['blosum62']
    fps_x_encoded = []
    for line in blosum:
        line = [float(x) for x in line.split(',')]
        fps_x_encoded.append(line)

    # pad sequences
    fps_x_blosum=pad_sequences(fps_x_encoded, maxlen=max_length, padding=padding_truncating, truncating=padding_truncating, dtype='float32')
    return data, fps_x_blosum


# PAD ZEROS 200 20 aa  X = 0 categorcial encoding
def pad_sequence(df, seq_len=700, padding='pre'):
    sequences_original = df['sequence'].tolist()
    sequences=[]
    for seq in sequences_original:
        seq1 = seq.replace('B', 'N')  # asparagine N / aspartic acid  D - asx - B
        seq2 = seq1.replace('Z', 'Q')  # glutamine Q / glutamic acid  E - glx - Z
        seq3 = seq2.replace('U',
                            'C')  # selenocisteina, the closest is the cisteine. but it is a different aminoacid . take care.
        seq4 = seq3.replace('O', 'K')  # Pyrrolysine to lysine
        sequences.append(seq4)

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
    fps_x = pad_sequences(sequences_integer_ecoded, maxlen=seq_len, padding=padding, value=0.0)   # (4042, 200)
    return fps_x
########################################################################################################################
#################################################### OTHERS ############################################################
########################################################################################################################
def turn_single_label(column, data):
    l = []
    for ec_list in data[column]:
        ec_l = set(ec_list)
        l.append(ec_l)
    data['ec_single_label']=l
    data = data.loc[data['ec_single_label'].apply(len)<2,:]
    return data

def remove_zeros(column, data):
    list_zeros=[0, 0.0, '0', '0.0', '0.0.0', '0.0.0.0']
    l = []
    for ec_list in data[column]:
        ec_l = [x for x in ec_list if x not in (list_zeros)]
        l.append(ec_l)
    data['non_negative'] =l

    data = data.loc[data['non_negative'].apply(len)>0,:]
    return data

# get columns names with number_aa
def column_name_sequence(len_seq, alphabet):
    columns = []
    count=1
    aa=1
    for x in range(len_seq*len(alphabet)):
        s='{}_{}'.format(count, aa)
        count+=1
        aa+=1
        if aa == len(alphabet)+1:
            aa = 1
        columns.append(s)
    return columns

def divide_dataset(fps_x, fps_y, test_size=0.2, val_size=0.1):
    # divide in train, test and validation
    x_train_1, x_test, y_train_1, y_test = train_test_split(fps_x, fps_y, test_size=test_size,random_state=42, shuffle=True, stratify=fps_y)

    # iterative_train_test_split(fps_x, fps_y, test_size=test_size)
    train_percentage = 1 - test_size
    val_size = val_size/train_percentage

    x_train, x_dval, y_train, y_dval = train_test_split(x_train_1, y_train_1, test_size=val_size, random_state=42, shuffle=True, stratify=y_train_1)

    # stratify=y_train_1, shuffle=True)

    return x_train, x_test, x_dval, y_train, y_test, y_dval

def normalization(x_train, x_test, x_dval):
    std_scale = StandardScaler().fit(x_train)
    x_train_std = std_scale.transform(x_train)
    x_test_std = std_scale.transform(x_test)
    x_dval_std = std_scale.transform(x_dval)

    # fps_x_std= StandardScaler().fit_transform(fps_x)
    return x_train_std, x_test_std, x_dval_std


# BINARIZE LABELS
def binarize_labels(fps_y): # for single
    test = pd.Series(fps_y)

    # mlb = MultiLabelBinarizer()
    # hot = mlb.fit_transform(test)
    # res = pd.DataFrame(hot,
    #                    columns=mlb.classes_,
    #                    index=test.index)
    fps_y = [item for sublist in fps_y for item in sublist] # this line is because they are retrieved as a list
    encoder = LabelEncoder()
    encoder.fit(fps_y)
    encoded_Y = encoder.transform(fps_y)
    classes = encoder.classes_
    fps_y = np_utils.to_categorical(encoded_Y) # convert integers to dummy variables (i.e. one hot encoded)

    # print(fps_y)
    # print(fps_y.shape)

    from sklearn.preprocessing import OneHotEncoder
    # creating instance of one-hot-encoder
    # enc = OneHotEncoder(handle_unknown='ignore')
    # # passing bridge-types-cat column (label encoded values of bridge_types)
    # enc_df = pd.DataFrame(enc.fit_transform(bridge_df[['Bridge_Types_Cat']]).toarray())
    # # merge with main df bridge_df on key values
    # bridge_df = bridge_df.join(enc_df)
    # bridge_df
    # bridge_types = list(set(fps_y))
    # bridge_df = pd.DataFrame(bridge_types, columns=['EC'])
    # # generate binary values using get_dummies
    # dum_df = pd.get_dummies(bridge_df, columns=["EC"], prefix=["Type_is"])
    # # merge with main df bridge_df on key values
    # bridge_df = bridge_df.join(dum_df)
    # print(bridge_df)

    return encoded_Y, fps_y, classes


########################################################################################################################
##############################################    RESHAPE   ############################################################
########################################################################################################################

def reshape_data_for_lstm(x_train, x_test, x_dval, y_train, y_test, y_dval):
    # (samples,1,lensequences)
    x_train, x_test, x_dval, y_train, y_test, y_dval = \
        map(lambda x: np.array(x, dtype=np.float), [x_train, x_test, x_dval, y_train, y_test, y_dval])

    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])  # (300455, 1, 20000)
    y_train = y_train.reshape(y_train.shape[0], 1)

    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])  # (100152, 1, 20000)
    y_test = y_test.reshape(y_test.shape[0], 1)

    x_dval = x_dval.reshape(x_dval.shape[0], 1, x_dval.shape[1])  # (100152, 1, 20000)
    y_dval = y_dval.reshape(y_dval.shape[0], 1)

    return x_train, x_test, x_dval, y_train, y_test, y_dval


def reshape_data_for_dense(x_train, x_test, x_dval, y_train, y_test, y_dval):
    # (samples,1,lensequences)
    x_train, x_test, x_dval, y_train, y_test, y_dval = \
        map(lambda x: np.array(x, dtype=np.float), [x_train, x_test, x_dval, y_train, y_test, y_dval])

    return x_train, x_test, x_dval, y_train, y_test, y_dval

def reshape_data_for_cnn1d(x_train, x_test, x_dval, y_train, y_test, y_dval):
    # Input to keras.layers.Conv1D should be 3-d with dimensions
    # (nb_of_examples, timesteps, features). I assume that you have a sequence of length 6000 with 1 feature. In this case:
    x_train, x_test, x_dval, y_train, y_test, y_dval = \
        map(lambda x: np.array(x, dtype=np.float), [x_train, x_test, x_dval, y_train, y_test, y_dval])

    x_train, x_test, x_dval = \
        map(lambda x: x.reshape(x.shape[0], 1, x.shape[1]), [x_train, x_test, x_dval])
    # n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    return x_train, x_test, x_dval, y_train, y_test, y_dval


def reshape_data_for_cnn2d(x_train, x_test, x_dval, y_train, y_test, y_dval,seq_len,alphabet ):
    # Input to keras.layers.Conv1D should be 3-d with dimensions
    # (nb_of_examples, timesteps, features). I assume that you have a sequence of length 6000 with 1 feature. In this case:
    x_train, x_test, x_dval, y_train, y_test, y_dval = \
        map(lambda x: np.array(x, dtype=np.float), [x_train, x_test, x_dval, y_train, y_test, y_dval])

    x_train, x_test, x_dval = \
        map(lambda x: x.reshape(x.shape[0],len(alphabet),seq_len), [x_train, x_test, x_dval])
    x_train, x_test, x_dval = \
        map(lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2],1), [x_train, x_test, x_dval])
    # batchsize + (image,image,channels) the channels be careful if first or last. here last
    return x_train, x_test, x_dval, y_train, y_test, y_dval


######################################################################################################
# Models
tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()
#input is 21 categorical 200 paded aa sequences
from keras.layers.merge import concatenate

def veltri_model(seq_len=700, final_units=8, output_dim = 256):
    with strategy.scope():
        model = Sequential()
        model.add(Input(shape=(seq_len,)))
        model.add(Embedding(input_dim=21, output_dim=output_dim, input_length=seq_len, mask_zero=True))
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
        model.add(Dense(final_units, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

#input is physicochemical enzyme features of size [9920 X 1]
def deepen_model(vector_size, final_units ):
    with strategy.scope():
        model = Sequential()
        n_timesteps, n_features = 1, vector_size
        model.add(Input(shape=(n_timesteps,n_features)))
        model.add(Conv1D(
            filters=46,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu'))
        model.add(MaxPool1D(pool_size=2, strides=1, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(400, activation='sigmoid'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(final_units, activation = 'softmax'))
        print(model.summary())
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model



def deepec_model(vector_size, final_units):
    with strategy.scope():
        input =(Input(shape=(vector_size,21,1)))
        conv1=Conv2D(
            filters=128,
            kernel_size=(16,21),
            strides=1,
            padding='same',
            activation='relu') (input)
        conv1 = MaxPool2D((4,4))(conv1)
        conv1=BatchNormalization()(conv1)
        conv2=Conv2D(
            filters=128,
            kernel_size=(8,21),
            strides=1,
            padding='same',
            activation='relu') (input)
        conv2 = MaxPool2D((4,4))(conv2)
        conv2=BatchNormalization()(conv2)

        conv3 = Conv2D(
            filters=128,
            kernel_size=(4,21),
            strides=1,
            padding='same',
            activation='relu') (input)
        conv3 = MaxPool2D((4,4))(conv3)
        conv3=BatchNormalization()(conv3)

        concat =concatenate([conv1,conv2, conv3])
        flat = Flatten()(concat)
        dr = Dropout(0.5)(flat)
        hidden1 = Dense(512)(dr)
        # hidden2 = Dense(512)(hidden1)
        dr2 = Dropout(0.3)(hidden1)
        dr2=BatchNormalization()(dr2)

        output = Dense(final_units, activation = 'softmax')(dr)
        deepec = Model(inputs=input, outputs=output, name='deepec')
        print(deepec.summary())

        deepec.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return deepec

from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Convolution2D, GRU, TimeDistributed, Reshape,MaxPooling2D,Convolution1D,BatchNormalization



def daclstm_model(shape=(700,21),len_seq=700, final_units=8):
    # with strategy.scope():
    #just physico chemical features

    main_input = Input(shape=shape, name='main_input')
    # concat = main_input

    # design the deepaclstm model
    conv1 = Conv1D(42,1,activation='relu', padding='same', activity_regularizer=regularizers.l2(0.001))(main_input)
    conv1 = Reshape((len_seq,42, 1))(conv1)

    conv2 = Conv2D(42,3,1,activation='relu', padding='same', activity_regularizer=regularizers.l2(0.001))(conv1)
    conv2 = Reshape((len_seq,42*42))(conv2)
    conv2 = Dropout(0.5)(conv2)
    dense = Dense(400, activation='relu')(conv2)

    lstm1 = Bidirectional(LSTM(units=300, return_sequences=True,recurrent_activation='sigmoid',dropout=0.5))(dense)
    lstm2 = Bidirectional(LSTM(units=300, return_sequences=True,recurrent_activation='sigmoid',dropout=0.5))(lstm1)

    # concat_features = Concatenate(axis=-1)([lstm_f2, lstm_b2, conv2_features])

    dr = Dropout(0.4)(lstm2)

    # concat_features = Flatten()(concat_features) #### add this part
    protein_features = Dense(600,activation='relu')(dr)
    fin = Flatten()(protein_features)
    # protein_features = TimeDistributed(Dense(600,activation='relu'))(concat_features)
    # protein_features = TimeDistributed(Dense(100,activation='relu', activity_regularizer=regularizers.l2(0.001)))(protein_features)
    main_output = Dense(final_units, activation='softmax', name='main_output')(fin)
    # main_output = (Dense(1, activation='sigmoid'))(protein_features)


    deepaclstm = Model(inputs=main_input, outputs=main_output)
    deepaclstm.compile(optimizer = 'Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #loss= 'binary_crossentropy'
    print(deepaclstm.summary())
    return deepaclstm

def dense_model(vector_size, final_units ):
    l1=1e-5
    l2=1e-4
    # with strategy.scope():
    model = Sequential()
    model.add(Input(shape=(vector_size,)))
    # model.add(Dense(2048, activation='relu'), kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))
    # model.add(Dropout(0.4))
    # model.add(BatchNormalization())
    # model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    # model.add(Dropout(0.4))
    # model.add(BatchNormalization())
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    # model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    # model.add(Dropout(0.4))
    # model.add(BatchNormalization())
    model.add(Dense(final_units, activation = 'softmax'))
    print(model.summary())
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#
# # # #################################################################################################
# # # # run stuff
# # # #
# # # # 1 . deepen physchemical features 8 classes multiclass
# phys_90 = pd.read_csv('/home/amsequeira/deepbio/datasets/ecpred/ecpred_phys_uniref_90_prepro.csv', low_memory=False)
# # data = get_ec_complete_more_than_x_samples(phys_90, x=30) # column to consider is 'ec_number4
# data = get_ec_complete_more_than_x_samples(phys_90, x=50, single_label=True)
# data, fps_x_phys, columns = physchemical(data)
# fps_x_phys = fps_x_phys.drop(['ec_number4', 'ec_single_label', 'ec_to_keep'], axis=1) #9469 columns
# fps_x_phys = fps_x_phys.astype(np.float32) #reduce memory usage
#
# fps_y = data['ec_single_label']
# y_encoded, fps_y_hot, ecs = binarize_labels(fps_y)
#
#
# # # # counts = Counter(x for xs in fps_y for x in set(xs))
# # # # counts.most_common()
# # # # df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
# # # # df_sorted = df.sort_values(by=[0], ascending=False)
# # # #
# # # # index      0
# # # # 0     2  54307
# # # # 3     3  34394
# # # # 4     0  22855
# # # # 2     1  19402
# # # # 7     6  16288
# # # # 1     4  12914
# # # # 5     5   8077
# # # # 6     7   6494
# # #
# #
# # divide in train, test and validation
# x_train, x_test, x_dval, y_train, y_test, y_dval = \
#     divide_dataset(fps_x_phys, y_encoded, test_size=0.2, val_size=0.2)
#
# #
# # normalization using the Training Set for both Training and Test would be as follows:
# x_train_std, x_test_std, x_dval_std = normalization(x_train, x_test, x_dval)
# # #
# # # # # #feature selection
# # # # # # Select features
# # # # from sklearn.feature_selection import f_classif
# # # #
# # # # fsel = FeatureSelection(x_train_std, y_train, columns_names=fps_x_phys.columns)
# # # # # KBest com *mutual info classif*
# # # # transf, X_fit_uni, X_transf_uni, column_selected, scores, scores_df = \
# # # #     fsel.run_univariate(score_func=f_classif, mode='percentile', param=50)
# # # #
# # # # x_train = X_transf_uni
# # # # x_test = transf.transform(x_test_std)
# # # # x_dval = transf.transform(x_dval_std)
# # # # columns = fps_x_phys.columns[column_selected]
# # # #
# # #
# vector_size = fps_x_phys.shape[1] # 9469
# # vector_size = len(columns)
# final_units = fps_y_hot.shape[1] # 8
# # #
# # # # x_train_std, x_test_std, x_dval_std, y_train, y_test, y_dval = \
# # # #     reshape_data_for_cnn1d(x_train_std, x_test_std, x_dval_std, y_train, y_test, y_dval)
# # #
# dl=DeepML(x_train = x_train_std.astype(np.float32), y_train = y_train,x_test=x_test_std.astype(np.float32), y_test= y_test,
#           number_classes=final_units, problem_type='multiclass',
#           x_dval=x_dval_std.astype(np.float32), y_dval=y_dval, epochs=500, batch_size=512,
#           path='/home/amsequeira/propythia/propythia/example', report_name='deepen_phys_8',
#           verbose=1)
#
# model = KerasClassifier(build_fn=dense_model, vector_size=vector_size, final_units = final_units)
# deepen = dl.run_model(model)
# scores, report, cm, cm2 = dl.model_complete_evaluate()
#
#
# print(ecs)
#
#

# def dense_model(vector_size, final_units ):
#     l1=1e-5
#     l2=1e-4
#     # with strategy.scope():
#     model = Sequential()
#     model.add(Input(shape=(vector_size,)))
#     # model.add(Dense(2048, activation='relu'), kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))
#     # model.add(Dropout(0.4))
#     # model.add(BatchNormalization())
#     model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
#     model.add(Dropout(0.4))
#     model.add(BatchNormalization())
#     model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
#     model.add(Dropout(0.4))
#     model.add(BatchNormalization())
#     model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
#     model.add(Dropout(0.3))
#     model.add(BatchNormalization())
#     model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
#     model.add(Dropout(0.4))
#     model.add(BatchNormalization())
#     model.add(Dense(final_units, activation = 'softmax'))
#     print(model.summary())
#     model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
#
# dl=DeepML(x_train = x_train_std.astype(np.float32), y_train = y_train,x_test=x_test_std.astype(np.float32), y_test= y_test,
#           number_classes=final_units, problem_type='multiclass',
#           x_dval=x_dval_std.astype(np.float32), y_dval=y_dval, epochs=500, batch_size=512,
#           path='/home/amsequeira/propythia/propythia/example', report_name='deepen_phys_8',
#           verbose=1)
#
# model = KerasClassifier(build_fn=dense_model, vector_size=vector_size, final_units = final_units)
# deepen = dl.run_model(model)
# scores, report, cm, cm2 = dl.model_complete_evaluate()


#
# # # 1 . deeEC hot encoding 8 classes multiclass
# hot_90 = pd.read_csv('/home/amsequeira/deepbio/datasets/ecpred/ecpred_uniprot_uniref_90.csv', low_memory=False)
# # data = get_ec_complete_more_than_x_samples(phys_90, x=30) # column to consider is 'ec_number4
# data = get_ec_1_level(hot_90, single_label=True)
# data, fps_x_hot, encoded = hot_encoded_sequence(data, column_sequence='sequence', seq_len=700, alphabet="XARNDCEQGHILKMFPSTWYV"
#                                        , padding_truncating='post')
#
# fps_y = data['ec_number1']
#
# # counts = Counter(x for xs in fps_y for x in set(xs))
# # counts.most_common()
# # df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
# # df_sorted = df.sort_values(by=[0], ascending=False)
# #
# # index      0
# # 0     2  54343
# # 3     3  34484
# # 4     0  22708
# # 2     1  19066
# # 7     6  16290
# # 1     4  12924
# # 5     5   8081
# # 6     7   6860
#
# #binarize labels
# y_encoded, fps_y_hot = binarize_labels(fps_y)
#
# # divide in train, test and validation
# x_train, x_test, x_dval, y_train, y_test, y_dval = \
#     divide_dataset(encoded, y_encoded, test_size=0.2, val_size=0.1)
#
# vector_size = encoded.shape[1] # 700
# final_units = fps_y_hot.shape[1] # 8
#
# # x_train, x_test, x_dval, y_train, y_test, y_dval = \
# #     reshape_data_for_cnn2d(x_train, x_test, x_dval, y_train, y_test, y_dval,seq_len=700, alphabet="XARNDCEQGHILKMFPSTWYV" )
# # x_train, x_test, x_dval, y_train, y_test, y_dval = \
# #     reshape_data_for_cnn1d(x_train, x_test, x_dval, y_train, y_test, y_dval )
# dl=DeepML(x_train = x_train[:30000], y_train = y_train[:30000],x_test=x_test[:10000], y_test= y_test[:10000],
#           number_classes=final_units, problem_type='multiclass',
#           x_dval=x_dval[:10000], y_dval=y_dval[:10000], epochs=500, batch_size=32,
#           path='/home/amsequeira/propythia/propythia/example', report_name='deepen_phys_8',
#           verbose=1)
#
# model = KerasClassifier(build_fn=daclstm_model, shape = (700,21) ,len_seq=700, final_units = final_units)
# deepec = dl.run_model(model)
# scores, report, cm, cm2 = dl.model_complete_evaluate()


#
# #######################################################################################################################
# # CATEGORICAL ENCODING
#
# hot_90 = pd.read_csv('/home/amsequeira/deepbio/datasets/ecpred/ecpred_uniprot_uniref_90.csv', low_memory=False)
# # data = get_ec_complete_more_than_x_samples(phys_90, x=30) # column to consider is 'ec_number4
# data = get_ec_1_level(hot_90, single_label=True)
# fps_x = pad_sequence(data, seq_len=700, padding='pre') #(174756, 700)
# fps_y = data['ec_number1']
#
# # # counts = Counter(x for xs in fps_y for x in set(xs))
# # # counts.most_common()
# # # df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
# # # df_sorted = df.sort_values(by=[0], ascending=False)
# # #
# # # index      0
# # # 0     2  54343
# # # 3     3  34484
# # # 4     0  22708
# # # 2     1  19066
# # # 7     6  16290
# # # 1     4  12924
# # # 5     5   8081
# # # 6     7   6860
# #
# # #binarize labels
# y_encoded, fps_y_hot = binarize_labels(fps_y)
# #
# # # divide in train, test and validation
# x_train, x_test, x_dval, y_train, y_test, y_dval = \
#     divide_dataset(fps_x, y_encoded, test_size=0.2, val_size=0.1)
# #
# vector_size = fps_x.shape[1] # 700
# final_units = fps_y_hot.shape[1] # 8
# #


def dense_model(vector_size, final_units):
    # with strategy.scope():
    l1=1e-5
    l2=1e-4
    # with strategy.scope():
    model = Sequential()
    model.add(Input(shape=(vector_size,)))
    # model.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    # model.add(Dropout(0.4))
    # model.add(BatchNormalization())
    # model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    # model.add(Dropout(0.3))
    # model.add(BatchNormalization())
    # model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    # model.add(Dropout(0.3))
    # model.add(BatchNormalization())
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    # model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    # model.add(Dropout(0.3))
    # model.add(BatchNormalization())
    model.add(Dense(final_units, activation = 'softmax'))
    print(model.summary())
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def cnn_model(vector_size, final_units):
    with strategy.scope():
        model=Sequential()
        input =(Input(shape=(vector_size,21,1)))
        model.add(input)
        model.add(Conv2D(
            filters=128,
            kernel_size=(16,21),
            strides=1,
            padding='same',
            activation='relu'))
        model.add(MaxPool2D((4,4)))
        model.add(BatchNormalization())
        # model.add(Conv2D(
        #     filters=128,
        #     kernel_size=(8,21),
        #     strides=1,
        #     padding='same',
        #     activation='relu') )
        # model.add(MaxPool2D((4,4)))
        # model.add(BatchNormalization())

        # model.add(Conv2D(
        #     filters=128,
        #     kernel_size=(4,21),
        #     strides=1,
        #     padding='same',
        #     activation='relu'))
        # model.add(MaxPool2D((2,2)))
        # model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(512))
        # hidden2 = Dense(512)(hidden1)
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

        model.add(Dense(final_units, activation = 'softmax'))
        print(model.summary())

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
#######################################################################################################################
# HOT ENCODING
seq_len = 500
hot_90 = pd.read_csv('/home/amsequeira/deepbio/datasets/ecpred/ecpred_uniprot_uniref_90.csv', low_memory=False)
# data = get_ec_complete_more_than_x_samples(phys_90, x=30) # column to consider is 'ec_number4
data = get_ec_1_level(hot_90, single_label=True)
# data = get_ec_2_level_more_than_x_samples(hot_90, x=50, single_label=True)
data=data.dropna(subset=['sequence'])
data = data[data['sequence'].str.contains('!!!') == False]  # because 2 sequences in ecpred had !!!! to long to EXCEL... warning instead of sequence
fps_x = pad_sequence(data, seq_len=seq_len, padding='pre')
fps_x = to_categorical(fps_x)
fps_x = fps_x.astype(np.float32) #reduce memory usage
fps_x = fps_x.reshape(fps_x.shape[0], fps_x.shape[1]*fps_x.shape[2])

#
fps_y = data['ec_single_label']
y_encoded, fps_y_hot, ecs = binarize_labels(fps_y)
#
# # divide in train, test and validation
x_train, x_test, x_dval, y_train, y_test, y_dval = \
    divide_dataset(fps_x, y_encoded, test_size=0.2, val_size=0.2)

vector_size = x_train.shape[1] # 700
final_units = fps_y_hot.shape[1] # 8

# x_train, x_test, x_dval, y_train, y_test, y_dval = reshape_data_for_cnn1d(x_train, x_test, x_dval, y_train, y_test, y_dval)

dl=DeepML(x_train = x_train.astype(np.float32), y_train = y_train,x_test=x_test.astype(np.float32),
          y_test= y_test,
          number_classes=final_units, problem_type='multiclass',
          x_dval=x_dval.astype(np.float32), y_dval=y_dval, epochs=500, batch_size=256,
          path='/home/amsequeira/propythia/propythia/example', report_name='deepen_phys_8',
          verbose=1)

# model = KerasClassifier(build_fn=dense_model, vector_size = vector_size, final_units = final_units, batch_size=512)
# model = KerasClassifier(build_fn=deepen_model, vector_size = vector_size, final_units = final_units, batch_size=512)
model = KerasClassifier(build_fn=veltri_model, seq_len = seq_len, final_units = final_units, output_dim = 64, batch_size=512)

dense = dl.run_model(model)
scores, report, cm, cm2 = dl.model_complete_evaluate()

# print(ecs)

# veltri_model(seq_len=700, final_units=8, output_dim = 256)
# deepen_model(vector_size, final_units )
# deepec_model(vector_size, final_units)
# daclstm_model(shape=(700,21),len_seq=700, final_units=8)


########################################################################################################################
# # FAMILIES
# hot90 = pd.read_csv('/home/amsequeira/deepbio/datasets/ecpred/ecpred_uniprot_families_uniref_90.csv', low_memory=False)
# # data = get_ec_1_level(hot90, single_label=True)
# data = get_ec_complete_more_than_x_samples(hot90, x=50, single_label=True)

# data = data.dropna(subset=['Cross-reference (Pfam)']).reset_index()
# data, fps_x, classes = hot_encoded_families(data, column_parameter='Cross-reference (Pfam)')
# fps_x=fps_x.astype(np.float32)
# fps_y = data['ec_number4']
# y_encoded, fps_y_hot, ecs = binarize_labels(fps_y)
#
#
# counts = Counter(x for xs in fps_y for x in set(xs))
# counts.most_common()
# df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
# df_sorted = df.sort_values(by=[0], ascending=False)
# print(df_sorted)
# x_train, x_test, x_dval, y_train, y_test, y_dval = \
#     divide_dataset(fps_x, y_encoded, test_size=0.2, val_size=0.1)
#
# vector_size = x_train.shape[1] # 700
# final_units = fps_y_hot.shape[1]
#

# def dense_model(vector_size, final_units):
#     l1=1e-5
#     l2=1e-4
#     # with strategy.scope():
#     model = Sequential()
#     model.add(Input(shape=(vector_size,)))
#     # model.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
#     # model.add(Dropout(0.4))
#     # model.add(BatchNormalization())
#     # model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
#     # model.add(Dropout(0.3))
#     # model.add(BatchNormalization())
#     # model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
#     # model.add(Dropout(0.3))
#     # model.add(BatchNormalization())
#     model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
#     model.add(Dropout(0.3))
#     model.add(BatchNormalization())
#     model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
#     model.add(Dropout(0.3))
#     model.add(BatchNormalization())
#     model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
#     model.add(Dropout(0.3))
#     model.add(BatchNormalization())
#     # model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
#     # model.add(Dropout(0.3))
#     # model.add(BatchNormalization())
#     model.add(Dense(final_units, activation = 'softmax'))
#     print(model.summary())
#     model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
#
#
#
# dl=DeepML(x_train = x_train, y_train = y_train,x_test=x_test, y_test= y_test,
#           number_classes=final_units, problem_type='multiclass',
#           x_dval=x_dval, y_dval=y_dval, epochs=500, batch_size=256,
#           path='/home/amsequeira/propythia/propythia/example', report_name='deepen_phys_8',
#           verbose=1)
#
# model = KerasClassifier(build_fn=dense_model, vector_size = vector_size, final_units = final_units, batch_size=512)
# dense = dl.run_model(model)
# scores, report, cm, cm2 = dl.model_complete_evaluate()
#
# print(ecs)

#
# def emb_cnn_dense_model(seq_len=700, final_units=8, output_dim = 256):
#     with strategy.scope():
#         model = Sequential()
#         n_timesteps, n_features = 1, seq_len
#         model.add(Input(shape=(n_timesteps, n_features)))
#         # model.add(Input(shape=(seq_len,)))
#         # model.add(Embedding(input_dim=21, output_dim=output_dim, input_length=seq_len, mask_zero=True))
#         model.add(Conv1D(
#             filters=64,
#             kernel_size=16,
#             strides=1,
#             padding='same',
#             activation='relu'))
#         model.add(MaxPool1D(pool_size=5, strides=1, padding='same'))
#         model.add(Flatten())
#         model.add(Dense(256))
#         model.add(Dense(64))
#         # Add Classification Dense, Compile model and make it ready for optimization
#         model.add(Dense(final_units, activation='softmax'))
#         print(model.summary())
#         model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#         return model
# def model():
#     with strategy.scope():
#         model = Sequential()
#         model.add(Bidirectional(LSTM(2048)))
#         model.add(Bidirectional(LSTM(1024)))
#         model.add(Dense(256))
#         model.add(Dense(128))
#         model.add(Dense(64))
#         model.add(Flatten())
#
#         # Add Classification Dense, Compile model and make it ready for optimization
#         model.add(Dense(final_units, activation='softmax'))
#         print(model.summary())
#         model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#         return model

# pad_sequence
K.clear_session()
tf.keras.backend.clear_session()