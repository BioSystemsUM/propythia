# %load_ext autoreload
# #
# %autoreload
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
sys.path.append('/home/amsequeira/deepbio')
from src.mlmodels.utils_run_models import divide_dataset, binarize_labels
from src.mlmodels.deep_ml_TrainEvalModel_propythia import ModelTrainEval
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef


########################################################################################################################
################################################## FOR Y ###############################################################
########################################################################################################################

# # select rows with complete ec numbers
# len(data['ec_number'].unique()) # 5315 different classes counts agglomerates
# data.groupby('ec_number').size().sort_values(ascending=False).head(20) # know which classe are most representative

# 1.Top 1000 classes complete classes with only 15. 800 with more than  MULTILABEL
def get_ec_complete_more_than_x_samples(data, x, single_label=True):
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
    return data   # column to consider is 'ec_number

##############
# get all the 3º level ec numbers complete with more than x samples
def get_ec_3_level_more_than_x_samples(data, x, single_label=True):
    # get all until the 3 level (everything until the last dot    175805
    l = []
    for ec_list in data['ec_number']:
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
    return data, fps_x

def hot_encoded_families(data, column_parameter='Cross-reference (Pfam)'):
    # divide parameter
    families = data[column_parameter]

    # remove columns with parameter NAn
    data = data[data[column_parameter].notna()] # from 175267 to 167638

    fam = [i.split(';') for i in data[column_parameter]]  # split dos Pfam 'PF01379;PF03900;'
    fam = [list(filter(None, x)) for x in fam]  # remove '' empty string (because last ;
    # fam = [set(x) for x in fam]

    mlb = MultiLabelBinarizer()
    fam_ho = mlb.fit_transform(fam)
    classes = mlb.classes_
    len(classes)
    return data, fam_ho, classes

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
    val_size = 1-(train_percentage*val_size*10)

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
    # fps_y = np_utils.to_categorical(encoded_Y) # convert integers to dummy variables (i.e. one hot encoded)
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

    return encoded_Y


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


########################################################################################################################
####################################################### RUN ############################################################
########################################################################################################################
#
# ecpred_uniref_90 = '/home/amsequeira/deepbio/datasets/ecpred/ecpred_uniprot_uniref_90.csv'
# data=pd.read_csv(ecpred_uniref_90, low_memory=False)
# #get y of interest
# data = get_ec_complete_more_than_x_samples(data, x=30) # column to consider is 'ec_number4
# data = get_ec_3_level_more_than_x_samples(data, x=30)  # column to consider is 'ec_number3
# data = get_ec_2_level_more_than_x_samples(data, x=30) # column to consider is 'ec_number2
# data = get_ec_1_level(data)                           # column to consider is 'ec_number1
# data = get_n_ec_level(n=2, data=data, x=30)         # column to consider is 'ec_level
# data = get_n_ec_level(n=3, data=data, x=30)
# data = get_n_ec_level(n=4, data=data, x=30)
# data = get_second_knowing_first(data, first_level)   # column to consider is 'ec_number2
# turn_single_label(column, data)
# remove_zeros(column, data)
#
#
# # HOT ENCODED
# alphabet = "ARNDCEQGHILKMFPSTWYV"
# # alphabet_x = "ARNDCEQGHILKMFPSTWYVX"
# # alphabet_all_characters = "ARNDCEQGHILKMFPSTWYVXBZUO"
# padding_truncating='post'
# padding_truncating='pre'
# # todo pad the middle
# seq_len=500 # check the graphics of len of aa
# data, fps_x_ho = hot_encoded_sequence(data=data, column_sequence='sequence', seq_len=500, alphabet=alphabet, padding_truncating='post')
# columns = column_name_sequence(fps_x, alphabet)
#
#
# # FAMILIES
# # 'Cross-reference (Pfam)',
# # 'Cross-reference (SUPFAM)', 'Gene ontology IDs', 'Protein families',
# # 'Cross-reference (InterPro)', 'Cross-reference (PROSITE)',
# # 'Cross-reference (KEGG)', 'Cross-reference (Reactome)'
# data, fps_x_fam, columns = hot_encoded_families(data, column_parameter='Cross-reference (Pfam)')
#
#
# # PHYSCHEMICAL
# data, fps_x_phys, columns = physchemical(data)
#
# # divide in train, test and validation
# # x_train, x_test, x_dval, y_train, y_test, y_dval = divide_train_test(fps_x, fps_y)
#
# # normalization using the Training Set for both Training and Test would be as follows:
# x_train_std, x_test_std, x_dval_std = normalization(x_train, x_test, x_dval)
#
# # NLF
# data, fps_x_nlf = nlf(data, seq_len, padding_truncating='post')
# columns = column_name_sequence(fps_x_nlf, alphabet)
#
# # divide in train, test and validation
# # x_train, x_test, x_dval, y_train, y_test, y_dval = divide_train_test(fps_x, fps_y)
#
# # normalization using the Training Set for both Training and Test would be as follows:
# x_train_std, x_test_std, x_dval_std = normalization(x_train, x_test, x_dval)
#
#
# # BLOSUM
# data, fps_x_blosum = blosum(data, seq_len, padding_truncating='post')
# columns = column_name_sequence(fps_x_blosum, alphabet)
#
# # divide in train, test and validation
# # x_train, x_test, x_dval, y_train, y_test, y_dval = divide_train_test(fps_x, fps_y)
#
# # normalization using the Training Set for both Training and Test would be as follows:
# x_train_std, x_test_std, x_dval_std = normalization(x_train, x_test, x_dval)
#
# # get input dim
# input_dim = fps_x.shape[1]
#
#######################################################################################################################
######################################################  RUN DEEP ###################################################
########################################################################################################################
def run_deep(fps_x, fps_y, columns, name, test_size=0.2, val_size=0.1):
    path='/home/amsequeira/deepbio/dl_reports/hot_seq'
    fps_y = binarize_labels(fps_y)
    input_dim = fps_x.shape[1]
    # number_classes = fps_y.shape[1]
    number_classes = len(np.unique(fps_y))
    problem_type='multiclass'
    x_train, x_test, x_dval, y_train, y_test, y_dval = divide_dataset(fps_x, fps_y, test_size, val_size)


    x_train_d, x_test_d, x_dval_d, y_train_d, y_test_d, y_dval_d = \
        reshape_data_for_dense(x_train, x_test, x_dval, y_train, y_test, y_dval)

    try:
        report_name = os.path.join(path,str(name+'dense_simple.txt'))
        model_name = os.path.join(path, str(name+'dense_simple.h5'))

        dl = ModelTrainEval(x_train=x_train_d, y_train=y_train_d, x_test=x_test_d, y_test=y_test_d,
                            number_classes=number_classes, problem_type=problem_type,
                            x_dval=x_dval_d, y_dval=y_dval_d,
                            epochs=500, batch_size=512,
                            reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
                            early_stopping_patience=30, reduce_lr_patience=50, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
                            path=path,
                            report_name=report_name)

        dnn_simple_ho = dl.run_dnn_simple(input_dim,
                                          optimizer='Adam',
                                          hidden_layers=(128, 64),
                                          dropout_rate=0.3,
                                          batchnormalization=True,
                                          l1=1e-5, l2=1e-4,
                                          final_dropout_value=0.3,
                                          initial_dropout_value=0.0,
                                          cv=5, optType='randomizedSearch', param_grid=None, n_iter_search=20, n_jobs=1,
                                          scoring=make_scorer(matthews_corrcoef))

        dl.save_model(path=model_name)
        scores, report, cm, cm2 = dl.model_complete_evaluate()
        # dl.precision_recall_curve(show=True, path_save='try_deep_pre_recall.png')
        # dl.roc_curve(path_save='try_deep_plot_roc_curve.png', show=True)
        K.clear_session()
        tf.keras.backend.clear_session()

    except Exception as e:
        print('error dnn_simple')
        print(e)
    try:
        report_name = os.path.join(path,str(name+'dense_embedding.txt'))
        model_name = os.path.join(path, str(name+'dense_embedding.h5'))

        dl = ModelTrainEval(x_train=x_train_d, y_train=y_train_d, x_test=x_test_d, y_test=y_test_d,
                            number_classes=number_classes, problem_type=problem_type,
                            x_dval=x_dval_d, y_dval=y_dval_d,
                            epochs=500, batch_size=128,
                            reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
                            early_stopping_patience=30, reduce_lr_patience=50, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
                            path=path,
                            report_name=report_name)

        dnn_embedding_ho = dl.run_dnn_embedding(input_dim,
                                        optimizer='Adam',
                                        input_dim_emb=21, output_dim=256, input_length=500, mask_zero=True,
                                        hidden_layers=(128, 64),
                                        dropout_rate=(0.3,),
                                        batchnormalization=(True,),
                                        l1=1e-5, l2=1e-4,
                                        final_dropout_value=(0.3,),
                                        cv=5, optType='randomizedSearch', param_grid=None, n_iter_search=20, n_jobs=1,
                                        scoring=make_scorer(matthews_corrcoef))

        dl.save_model(path=model_name)
        scores, report, cm, cm2 = dl.model_complete_evaluate()
        # dl.precision_recall_curve(show=True, path_save='try_deep_pre_recall.png')
        # dl.roc_curve(path_save='try_deep_plot_roc_curve.png', show=True)
        K.clear_session()
        tf.keras.backend.clear_session()

    except Exception as e:
        print('error dnn_embedding')
        print(e)
    try:
        report_name = os.path.join(path,str(name+'lstm_embedding.txt'))
        model_name = os.path.join(path, str(name+'lstm_embedding.h5'))

        dl = ModelTrainEval(x_train=x_train_d, y_train=y_train_d, x_test=x_test_d, y_test=y_test_d,
                            x_dval=x_dval_d, y_dval=y_dval_d,
                            number_classes=number_classes, problem_type=problem_type,
                            epochs=500, batch_size=64,
                            reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
                            early_stopping_patience=30, reduce_lr_patience=50, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
                            path=path,
                            report_name=report_name)

        lstm_embedding_ho = dl.run_lstm_embedding(
            optimizer='Adam',
            input_dim_emb=21, output_dim=128, input_length=500, mask_zero=True,
            bilstm=True,
            lstm_layers=(128, 64),
            activation='tanh',
            recurrent_activation='sigmoid',
            dropout_rate=(0.3,), recurrent_dropout_rate=(0.3,),
            l1=1e-5, l2=1e-4,
            dense_layers=(64, 32),
            dense_activation="relu",
            dropout_rate_dense=(0.3,),
            batchnormalization=(True,),
            cv=5, optType='randomizedSearch', param_grid=None, n_iter_search=5, n_jobs=1,
            scoring=make_scorer(matthews_corrcoef))

        dl.save_model(path=model_name)
        scores, report, cm, cm2 = dl.model_complete_evaluate()
        # dl.precision_recall_curve(show=True, path_save='try_deep_pre_recall.png')
        # dl.roc_curve(path_save='try_deep_plot_roc_curve.png', show=True)
        K.clear_session()
        tf.keras.backend.clear_session()

    except Exception as e:
        print('error lstm_embedding')
        print(e)

    x_train_l, x_test_l, x_dval_l, y_train_l, y_test_l, y_dval_l = \
        reshape_data_for_lstm(x_train, x_test, x_dval, y_train, y_test, y_dval)
    try:
        report_name = os.path.join(path,str(name+'lstm_simple.txt'))
        model_name = os.path.join(path, str(name+'lstm_simple.h5'))

        dl = ModelTrainEval(x_train=x_train_l, y_train=y_train, x_test=x_test_l, y_test=y_test,
                            number_classes=number_classes, problem_type=problem_type,
                            x_dval=x_dval_l, y_dval=y_dval,
                            epochs=500, batch_size=512,
                            reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
                            early_stopping_patience=30, reduce_lr_patience=50, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
                            path=path,
                            report_name=report_name)

        lstm_embedding_ho = dl.run_lstm_embedding(
            optimizer='Adam',
            input_dim_emb=21, output_dim=128, input_length=500, mask_zero=True,
            bilstm=True,
            lstm_layers=(128, 64),
            activation='tanh',
            recurrent_activation='sigmoid',
            dropout_rate=(0.3,), recurrent_dropout_rate=(0.3,),
            l1=1e-5, l2=1e-4,
            dense_layers=(64, 32),
            dense_activation="relu",
            dropout_rate_dense=(0.3,),
            batchnormalization=(True,),
            cv=5, optType='randomizedSearch', param_grid=None, n_iter_search=5, n_jobs=1,
            scoring=make_scorer(matthews_corrcoef))

        dl.save_model(path=model_name)
        scores, report, cm, cm2 = dl.model_complete_evaluate()
        # dl.precision_recall_curve(show=True, path_save='try_deep_pre_recall.png')
        # dl.roc_curve(path_save='try_deep_plot_roc_curve.png', show=True)
        K.clear_session()
        tf.keras.backend.clear_session()

    except Exception as e:
        print('error lstm_simple')
        print(e)

    x_train_c1, x_test_c1, x_dval_c1, y_train_c1, y_test_c1, y_dval_c1 = \
        reshape_data_for_cnn1d(x_train, x_test, x_dval, y_train, y_test, y_dval)
    try:
        report_name = os.path.join(path,str(name+'cnn1d.txt'))
        model_name = os.path.join(path, str(name+'cnn1d.h5'))

        dl = ModelTrainEval(x_train=x_train_c1, y_train=y_train_c1, x_test=x_test_c1, y_test=y_test_c1,
                            number_classes=number_classes, problem_type=problem_type,
                            x_dval=x_dval_c1, y_dval=y_dval_c1,
                            epochs=500, batch_size=512,
                            reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
                            early_stopping_patience=30, reduce_lr_patience=50, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
                            path=path,
                            report_name=report_name)

        cnn1d_ho = dl.run_cnn_1D(input_dim,
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
                         cv=5, optType='randomizedSearch', param_grid=None, n_iter_search=5, n_jobs=1,
                         scoring=make_scorer(matthews_corrcoef))


        dl.save_model(path=model_name)
        scores, report, cm, cm2 = dl.model_complete_evaluate()
        # dl.precision_recall_curve(show=True, path_save='try_deep_pre_recall.png')
        # dl.roc_curve(path_save='try_deep_plot_roc_curve.png', show=True)
        K.clear_session()
        tf.keras.backend.clear_session()

    except Exception as e:
        print('error cnn1d')
        print(e)

    try:
        report_name = os.path.join(path,str(name+'cnn_lstm.txt'))
        model_name = os.path.join(path, str(name+'cnn_lstm.h5'))

        dl = ModelTrainEval(x_train=x_train_c1, y_train=y_train_c1, x_test=x_test_c1, y_test=y_test_c1,
                            number_classes=number_classes, problem_type=problem_type,
                            x_dval=x_dval_c1, y_dval=y_dval_c1,
                            epochs=500, batch_size=128,
                            reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
                            early_stopping_patience=30, reduce_lr_patience=50, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
                            path=path,
                            report_name=report_name)

        cnn_lstm_ho = dl.run_cnn_lstm(input_dim,
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
                                   batchnormalization=(True,),
                                   cv=5, optType='randomizedSearch', param_grid=None, n_iter_search=5, n_jobs=1,
                                   scoring=make_scorer(matthews_corrcoef))

        dl.save_model(path=model_name)
        scores, report, cm, cm2 = dl.model_complete_evaluate()
        # dl.precision_recall_curve(show=True, path_save='try_deep_pre_recall.png')
        # dl.roc_curve(path_save='try_deep_plot_roc_curve.png', show=True)
        K.clear_session()
        tf.keras.backend.clear_session()

    except Exception as e:
        print('error cnn_lstm')
        print(e)

# did not make the CNN2D


########################################################################################################################
######################################################  RUN DATASETS ###################################################
########################################################################################################################
if __name__ == '__main__':
    # SEQUENCE
    ########################################################################################################################
    ecpred_uniref_90 = '/home/amsequeira/deepbio/datasets/ecpred/ecpred_uniprot_uniref_90.csv'
    data=pd.read_csv(ecpred_uniref_90, low_memory=False)
    #get y of interest
    data_till_4 = get_ec_complete_more_than_x_samples(data, x=30, single_label=True)# column to consider is 'ec_number4
    # counts = Counter(x for xs in data_till_4['ec_number4'] for x in set(xs))
    # print(counts.most_common())

    data_till_3 = get_ec_3_level_more_than_x_samples(data, x=30, single_label=True)  # column to consider is 'ec_number3

    data_till_2 = get_ec_2_level_more_than_x_samples(data, x=30, single_label=True) # column to consider is 'ec_number2
    # data_till_2 = turn_single_label('ec_number2', data_till_2)

    data_till_1 = get_ec_1_level(data, single_label=True)                           # column to consider is 'ec_number1
    # data_till_1 = turn_single_label('ec_number1', data_till_1)

    data_2 = get_n_ec_level(n=2, data=data, x=30, single_label=True)         # column to consider is 'ec_level
    # data_2 = turn_single_label('ec_level', data_2)

    data_3 = get_n_ec_level(n=3, data=data, x=30, single_label=True)
    # data_3 = turn_single_label('ec_level', data_3)

    data_4 = get_n_ec_level(n=4, data=data, x=30, single_label=True)
    # data_4 = turn_single_label('ec_level', data_4)

    data_hier1 = get_second_knowing_first(data, 1, single_label=True) # column to consider is 'ec_number2
    # data_hier1 = turn_single_label('ec_number2', data_hier1)

    data_hier2 = get_second_knowing_first(data, 2, single_label=True)
    # data_hier2 = turn_single_label('ec_number2', data_hier2)

    data_hier3 = get_second_knowing_first(data, 3, single_label=True)
    # data_hier3 = turn_single_label('ec_number2', data_hier3)

    data_hier4 = get_second_knowing_first(data, 4, single_label=True)
    # data_hier4 = turn_single_label('ec_number2', data_hier4)

    data_hier5 = get_second_knowing_first(data, 5, single_label=True)
    # data_hier5 = turn_single_label('ec_number2', data_hier5)

    data_hier6 = get_second_knowing_first(data, 6, single_label=True)
    # data_hier6 = turn_single_label('ec_number2', data_hier6)

    data_hier7 = get_second_knowing_first(data, 7, single_label=True)
    # data_hier7 = turn_single_label('ec_number2', data_hier7)

    data_till_1_no_neg = remove_zeros('ec_number1', data_till_1)       # column to consider is 'ec_number1
    data_till_1_no_neg = turn_single_label('ec_number1', data_till_1_no_neg)

    # further transformations
    # turn_single_label(column, data) # todo multilabel. train test slit change.  no startify. and changes in the deep learning model
    # also the y cannot be flatten and need to be multibinarizer
    # remove_zeros(column, data)

    alphabet = "ARNDCEQGHILKMFPSTWYV"
    # alphabet_x = "ARNDCEQGHILKMFPSTWYVX"
    # alphabet_all_characters = "ARNDCEQGHILKMFPSTWYVXBZUO"

    padding_truncating='post'
    # padding_truncating='pre'
    # pad the middle

    seq_len=500 # check the graphics of len of aa
    columns = column_name_sequence(len_seq=seq_len, alphabet=alphabet)

    test_size=0.2
    val_size=0.1

    data, fps_x_ho_till_4 = \
        hot_encoded_sequence(data=data_till_4, column_sequence='sequence', seq_len=seq_len, alphabet=alphabet, padding_truncating=padding_truncating)
    fps_y_ho_till_4 = data['ec_number4']
    run_deep(fps_x_ho_till_4, fps_y_ho_till_4, columns=columns, name = 'ho_till_4', test_size=test_size, val_size=val_size)

    data, fps_x_ho_till_3 = hot_encoded_sequence(data=data_till_3, column_sequence='sequence', seq_len=seq_len, alphabet=alphabet, padding_truncating=padding_truncating)
    fps_y_ho_till_3 = data['ec_number3']
    run_deep(fps_x_ho_till_3, fps_y_ho_till_3, columns=columns, name = 'ho_till_3', test_size=test_size, val_size=val_size)

    data, fps_x_ho_till_2 = \
        hot_encoded_sequence(data=data_till_2, column_sequence='sequence', seq_len=seq_len, alphabet=alphabet, padding_truncating=padding_truncating)
    fps_y_ho_till_2 = data['ec_number2']
    run_deep(fps_x_ho_till_2, fps_y_ho_till_2, columns=columns,  name = 'ho_till_2', test_size=test_size, val_size=val_size)

    data, fps_x_ho_till_1 = \
        hot_encoded_sequence(data=data_till_1, column_sequence='sequence', seq_len=seq_len, alphabet=alphabet, padding_truncating=padding_truncating)
    fps_y_ho_till_1 = data['ec_number1']
    run_deep(fps_x_ho_till_1, fps_y_ho_till_1, columns=columns, name = 'ho_till_1', test_size=test_size, val_size=val_size)

    data, fps_x_ho_till_1_no_neg = \
        hot_encoded_sequence(data=data_till_1_no_neg, column_sequence='sequence', seq_len=seq_len, alphabet=alphabet, padding_truncating=padding_truncating)
    fps_y_ho_till_1_no_neg = data['ec_number1']
    run_deep(fps_x_ho_till_1_no_neg, fps_y_ho_till_1_no_neg, columns=columns, name = 'ho_till_1_no_neg', test_size=test_size, val_size=val_size)

    data, fps_x_ho_2lev = \
        hot_encoded_sequence(data=data_2, column_sequence='sequence', seq_len=seq_len, alphabet=alphabet, padding_truncating=padding_truncating)
    fps_y_ho_2lev = data['ec_level']
    run_deep(fps_x_ho_2lev, fps_y_ho_2lev, columns=columns, name = 'ho_2lev', test_size=test_size, val_size=val_size)

    data, fps_x_ho_3lev = \
        hot_encoded_sequence(data=data_3, column_sequence='sequence', seq_len=seq_len, alphabet=alphabet, padding_truncating=padding_truncating)
    fps_y_ho_3lev = data['ec_level']
    run_deep(fps_x_ho_3lev, fps_y_ho_3lev, columns=columns, name = 'ho_3lev', test_size=test_size, val_size=val_size)

    data, fps_x_ho_4lev = \
        hot_encoded_sequence(data=data_4, column_sequence='sequence', seq_len=seq_len, alphabet=alphabet, padding_truncating=padding_truncating)
    fps_y_ho_4lev = data['ec_level']
    run_deep(fps_x_ho_4lev, fps_y_ho_4lev, columns=columns,  name = 'ho_4lev', test_size=test_size, val_size=val_size)

    data, fps_x_ho_hier1 = \
        hot_encoded_sequence(data=data_hier1, column_sequence='sequence', seq_len=seq_len, alphabet=alphabet, padding_truncating=padding_truncating)
    fps_y_ho_hier1 = data['ec_number2']
    run_deep(fps_x_ho_hier1, fps_y_ho_hier1, columns=columns, name = 'ho_hier1', test_size=test_size, val_size=val_size)

    data, fps_x_ho_hier2 = \
        hot_encoded_sequence(data=data_hier2, column_sequence='sequence', seq_len=seq_len, alphabet=alphabet, padding_truncating=padding_truncating)
    fps_y_ho_hier2 = data['ec_number2']
    run_deep(fps_x_ho_hier2, fps_y_ho_hier2, columns=columns, name = 'ho_hier2', test_size=test_size, val_size=val_size)

    data, fps_x_ho_hier3 = \
        hot_encoded_sequence(data=data_hier3, column_sequence='sequence', seq_len=seq_len, alphabet=alphabet, padding_truncating=padding_truncating)
    fps_y_ho_hier3 = data['ec_number2']
    run_deep(fps_x_ho_hier3, fps_y_ho_hier3, columns=columns, name = 'ho_hier3', test_size=test_size, val_size=val_size)

    data, fps_x_ho_hier4 = \
        hot_encoded_sequence(data=data_hier4, column_sequence='sequence', seq_len=seq_len, alphabet=alphabet, padding_truncating=padding_truncating)
    fps_y_ho_hier4 = data['ec_number2']
    run_deep(fps_x_ho_hier4, fps_y_ho_hier4, columns=columns, name = 'ho_hier4', test_size=test_size, val_size=val_size)

    data, fps_x_ho_hier5 = \
        hot_encoded_sequence(data=data_hier5, column_sequence='sequence', seq_len=seq_len, alphabet=alphabet, padding_truncating=padding_truncating)
    fps_y_ho_hier5 = data['ec_number2']
    run_deep(fps_x_ho_hier5, fps_y_ho_hier5, columns=columns, name = 'ho_hier5', test_size=test_size, val_size=val_size)

    data, fps_x_ho_hier6 = \
        hot_encoded_sequence(data=data_hier6, column_sequence='sequence', seq_len=seq_len, alphabet=alphabet, padding_truncating=padding_truncating)
    fps_y_ho_hier6 = data['ec_number2']
    run_deep(fps_x_ho_hier6, fps_y_ho_hier6, columns=columns, name = 'ho_hier6', test_size=test_size, val_size=val_size)


    data, fps_x_ho_hier7 = \
        hot_encoded_sequence(data=data_hier7, column_sequence='sequence', seq_len=seq_len, alphabet=alphabet, padding_truncating=padding_truncating)
    fps_y_ho_hier7 = data['ec_number2']
    run_deep(fps_x_ho_hier7, fps_y_ho_hier7, columns=columns, name = 'ho_hier7', test_size=test_size, val_size=val_size)

