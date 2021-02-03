import os
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, Add, MaxPooling1D, BatchNormalization
from keras.layers import Embedding, Bidirectional, LSTM, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
logging.getLogger("tensorflow").setLevel(logging.FATAL)
from src.mlmodels.class_TrainEvalModel import ModelTrainEval

tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()


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



def create_dict(codes):
    char_dict = {}
    for index, val in enumerate(codes):
        char_dict[val] = index + 1
    return char_dict

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
char_dict = create_dict(codes)

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


########################################################################################################################
########################################################################################################################
# 4. Deep Learning Models
########################################################################################################################
########################################################################################################################
# Text Preprocessing
########################################################################################################################

def bidilstm(input_dim, final_units):
    with strategy.scope():
        # Model 1: Bidirectional LSTM
        x_input = Input(shape=(input_dim,))
        emb = Embedding(21, 128, input_length=input_dim)(x_input)
        # bi_rnn = Bidirectional(CuDNNLSTM(64, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))(emb)
        # CuDNNLSTM was giving problems and isthe same as LSTM in newer versions
        bi_rnn = Bidirectional(
            LSTM(64, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))(emb)

        x = Dropout(0.3)(bi_rnn)

        # softmax classifier add dense different 8 classes instead of 1000
        x_output = Dense(final_units, activation='softmax')(x)

        model1 = Model(inputs=x_input, outputs=x_output)
        model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model1.summary()
        return model1

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
########################################################################################################################
########################################################################################################################

def run(max_length = 700):

    ecpred_uniref_90 = '/home/amsequeira/deepbio/datasets/ecpred/ecpred_uniprot_uniref_90.csv'
    source_file = pd.read_csv(ecpred_uniref_90, low_memory=False)

    file = source_file[source_file.sequence.str.contains('!!!') == False]  # because 2 sequences in ecpred had !!!! to long to EXCEL... warning instead of sequence
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

    # encode x (letter aa --> number)
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

    # # # label/integer encoding output variable: (y)
    le = LabelEncoder()
    # y_train_le = le.fit_transform(train_sm['family_accession'])
    # y_val_le = le.transform(val_sm['family_accession'])
    # y_test_le = le.transform(test_sm['family_accession'])
    # this part i didnt do because separate all the datasets but try as this way !!!!
    # i already have encoded as the EC number

    # One hot encoding of outputs
    y_train_ohe = to_categorical(y_train)
    y_val_ohe = to_categorical(y_val)
    y_test_ohe = to_categorical(y_test)
    print(train_ohe.shape)
    train = ModelTrainEval(train_ohe, y_train, test_ohe, y_test, val_ohe, y_val, validation_split=0.25,
                           epochs=500, callbacks=None, reduce_lr=True, early_stopping=True, checkpoint=True,
                           early_stopping_patience=int(30),
                           reduce_lr_patience=50, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
                           path='', save_model=False, plot_model_file='model_plot.png',
                           verbose=2, batch_size=16)

    # article loss = categorical crossentropy, onehotencoded train. numberclasses=len(y_train[0])
    # sparse categoical crossentropy y train encoded number classes = len(np.unique(y_train))
    # bilstm = KerasClassifier(build_fn=bidilstm, final_units=len(np.unique(y_train)),
    #                         input_dim=max_length)

    protcnn = KerasClassifier(build_fn=prot_cnn, input_dim=max_length, number_classes=max_length)

    model, history = train.run_model(protcnn)
    plot_history(history)
    display_model_score(model,
         [train_ohe, y_train],
         [val_ohe, y_val],
         [test_ohe, y_test],
         256)
    score= train.model_simple_evaluate(x_test,y_test)
    print(score)
    train.print_evaluation(x_test, y_test)
    # try different optimizers? and other parameters?

run(100)