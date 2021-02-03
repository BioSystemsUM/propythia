import functools
import time
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, TimeDistributed, Bidirectional, Concatenate
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization, Flatten
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
from keras.models import Sequential
from matplotlib import pyplot
from keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Convolution2D, GRU, TimeDistributed, Reshape,MaxPooling2D,Convolution1D,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import metrics
import matplotlib.pyplot as plt
import numpy as np
import random as rn
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = '3,4,5,6,7'
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
tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()

# def read_divide_data_random(file, columns_not_x, column_label):
#     dataset, fps_x, fps_y = divide_dataset(file, columns_not_x, column_label)
#
#     fps_y_bin=binarize_labels(fps_y) # binarize labels EC_number
#     X_train_1, X_test, y_train_1, y_test = train_test_split(fps_x, fps_y_bin, test_size=0.20, random_state=42)
#     X_train, x_dval, y_train, y_dval = train_test_split(X_train_1, y_train_1, test_size=0.20, random_state=42)
#     X_train, X_test, x_dval, y_train, y_test, y_dval = map(lambda x: np.array(x, dtype=np.float), [X_train, X_test, x_dval, y_train, y_test, y_dval])
#
#     show_shapes(X_train,X_test, x_dval, y_train, y_test,y_dval)
#     return X_train,X_test, x_dval, y_train, y_test,y_dval


# def shaping_data( X_train, y_train, X_test, y_test, x_dval, y_dval):
#
#     X_train = X_train.reshape(X_train.shape[1],1,X_train.shape[0])
#     y_train= y_train.to_numpy().reshape(1,y_train.shape[0])
#     X_test=X_test.reshape(X_test.shape[1],1,X_test.shape[0] )
#     y_test=y_test.to_numpy().reshape( 1,y_test.shape[0])
#     x_dval=x_dval.reshape(x_dval.shape[1],1,x_dval.shape[0])
#     y_dval=y_dval.to_numpy().reshape(1,y_dval.shape[0])
#     # LSTM needs input in 3 dimensions (bachsize, timesteps,size)
#     return X_train, y_train, X_test, y_test, x_dval, y_dval
#

def binarize_labels(column_label):
    # print('binarize labels')
    fps_y_bin=[]
    for x in column_label:
        if x == 'non_Enzyme': fps_y_bin.append(int(0))
        elif x == 0 or x == '0': fps_y_bin.append(int(0))
        else: fps_y_bin.append(int(1))
    return fps_y_bin


def plotLoss(history):

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend(['trainloss', 'valloss'], loc='upper left')
    plt.savefig("figures/"+'lstmloss07' +".png" , dpi=600, facecolor='w', edgecolor='w', orientation='portrait',
                papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)
    plt.close()

    ## PLOT CINDEX
    plt.figure()
    plt.title('model  accuracy')
    plt.ylabel('Q8 accuracy')
    plt.xlabel('epoch')
    plt.plot(history.history['weighted_accuracy'])
    plt.plot(history.history['val_weighted_accuracy'])
    plt.legend(['trainaccuracy', 'valaccuracy'], loc='upper left')

    plt.savefig("figures/"+'lstmaccuracy07'+ ".png" , dpi=600, facecolor='w', edgecolor='w', orientation='portrait',
                papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)


def aclstm_simple_feature(input_shape,len_features,trainmain,trainlabel,valmain,vallabel,testmain,test_label):

    def build_model(shape=(700,21),len_seq=700):
        """from daclstm with small changes"""
        np.random.seed(2018)
        rn.seed(2018)
        with strategy.scope():
            #just physico chemical features
            print(input_shape)
            main_input = Input(shape=shape, name='main_input')
            concat = main_input

            # design the deepaclstm model
            conv1_features = Convolution1D(42,1,activation='relu', padding='same', activity_regularizer=regularizers.l2(0.001))(concat)
            print ('conv1_features shape', conv1_features.get_shape())
            conv1_features = Reshape((len_seq,42, 1))(conv1_features)

            conv2_features = Convolution2D(42,3,1,activation='relu', padding='same', activity_regularizer=regularizers.l2(0.001))(conv1_features)
            print ('conv2_features.get_shape()', conv2_features.get_shape())

            conv2_features = Reshape((len_seq,42*42))(conv2_features)
            conv2_features = Dropout(0.5)(conv2_features)
            conv2_features = Dense(400, activation='relu')(conv2_features)

            lstm_f1 = LSTM(units=300, return_sequences=True,recurrent_activation='sigmoid',dropout=0.5)(conv2_features)
            lstm_b1 = LSTM(units=300, return_sequences=True, go_backwards=True,recurrent_activation='sigmoid',dropout=0.5)(conv2_features)

            lstm_f2 = LSTM(units=300, return_sequences=True,recurrent_activation='sigmoid',dropout=0.5)(lstm_f1)
            lstm_b2 = LSTM(units=300, return_sequences=True, go_backwards=True,recurrent_activation='sigmoid',dropout=0.5)(lstm_b1)

            print('merge')

            concat_features = Concatenate(axis=-1)([lstm_f2, lstm_b2, conv2_features])

            concat_features = Dropout(0.4)(concat_features)

            # concat_features = Flatten()(concat_features) #### add this part
            protein_features = Dense(600,activation='relu')(concat_features)
            # protein_features = TimeDistributed(Dense(600,activation='relu'))(concat_features)
            # protein_features = TimeDistributed(Dense(100,activation='relu', activity_regularizer=regularizers.l2(0.001)))(protein_features)
            print('last layer before timedistributed', protein_features.shape)
            main_output = TimeDistributed(Dense(8, activation='softmax', name='main_output'))(protein_features)
            # main_output = (Dense(1, activation='sigmoid'))(protein_features)

            print('main output', main_output.shape)

            print('Model')
            deepaclstm = Model(inputs=[main_input], outputs=[main_output])
            adam = Adam(lr=0.003)
            deepaclstm.compile(optimizer = adam, loss={'main_output': 'categorical_crossentropy'}, metrics=['weighted_accuracy'])
            #loss= 'binary_crossentropy'
            deepaclstm.summary()
            return model

    print('start model')

    ####building the model
    # model = KerasClassifier(build_fn=build_model, nb_epoch=150, batch_size=512, verbose=2)
    model=build_model()
    earlyStopping = EarlyStopping(monitor='val_weighted_accuracy', patience=5, verbose=1, mode='auto')
    #original 'val_weighted_accuracy'

    load_file = "./model/ac_LSTM_best_time_07.h5"
    checkpointer = ModelCheckpoint(filepath=load_file,verbose=1,save_best_only=True)
    #
    # history=model.fit(trainmain,trainlabel, validation_data=(valmain,vallabel),
    #               nb_epoch=150, batch_size=10, callbacks=[checkpointer, earlyStopping], verbose=2, shuffle=True)
    history = model.fit({'main_input': trainmain}, {'main_output': trainlabel},
                        validation_data=({'main_input': valmain}, {'main_output': vallabel}),
                        nb_epoch=150, batch_size=42, callbacks=[checkpointer, earlyStopping], verbose=2, shuffle=True)
    plotLoss(history)
    model.load_weights(load_file)

    print("#########evaluate:##############")
    score = model.evaluate({'main_input': testmain},{'main_output': test_label}, verbose=2, batch_size=2)
    print (score)
    print ('test loss:', score[0])
    print ('test accuracy:', score[1])

    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()

# def aclstm_concat_feature(main_input_shape,main_output_dim, main_input_dim,auxiliary_input_shape, len_features,trainmain,trainlabel,valmain,vallabel,testmain,test_label, auxiliary_input=None, trainaux=None, valaux=None, testaux=None):
#
#     def build_model():
#         """from daclstm with small changes"""
#         np.random.seed(2018)
#         rn.seed(2018)
#
#         # design the deepaclstm model
#         main_input = Input(main_input_shape, dtype='float32', name='main_input')
#         # main_input = Input(shape=(700,), dtype='float32', name='main_input')
#         #main_input = Masking(mask_value=23)(main_input)
#         x = Embedding(output_dim=main_output_dim, input_dim=main_output_dim, input_length=main_input_dim)(main_input)
#         # x = Embedding(output_dim=21, input_dim=21, input_length=700)(main_input)
#         auxiliary_input = Input(shape=auxiliary_input_shape, name='aux_input')  #24
#         # auxiliary_input = Input(shape=(700,21), name='aux_input')  #24
#         #auxiliary_input = Masking(mask_value=0)(auxiliary_input)
#         print(main_input.get_shape())
#         print(auxiliary_input.get_shape())
#         concat = Concatenate([x, auxiliary_input], concat_axis=-1)
#
#         # design the deepaclstm model
#         conv1_features = Convolution1D(42,1,activation='relu', padding='same', activity_regularizer=regularizers.l2(0.001))(concat)
#         resto igual


def divide_train_test(fps_x, fps_y_bin):

    # divide in train, test and validation
    x_train_1, x_test, y_train_1, y_test = train_test_split(fps_x, fps_y_bin, test_size=0.20, random_state=42, stratify=fps_y_bin)
    x_train, x_dval, y_train, y_dval = train_test_split(x_train_1, y_train_1, test_size=0.25, random_state=42, stratify=y_train_1)
    return x_train, x_test,x_dval, y_train, y_test, y_dval


if __name__ == '__main__':

    # ecpred ='/home/amsequeira/deepbio/datasets/ecpred/ecpred_uniprot_the_one.csv'
    # ecpred_silico = '/home/amsequeira/deepbio/datasets/ecpred_silico/ecpred_silico_uniprot.csv'
    # silico = '/home/amsequeira/deepbio/datasets/silico/dataset_silico_separated_labels.csv'
    ecpred = r'/home/martinha/PycharmProjects/deepbio/datasets/datasets/the ones/ecpred_uniprot_the_one.csv'


    alphabet_simple = "ARNDCEQGHILKMFPSTWYV"
    alphabet_x = "ARNDCEQGHILKMFPSTWYVX"
    alphabet_all_characters = "ARNDCEQGHILKMFPSTWYVXBZUO"

    len_seq=700
    alphabet=alphabet_x
    alphabet_size=21
     # for aa sequence input considering sequence lenght of 1000 and just 20 aminoacids
    input_shape = (700,21)
    len_features = 1000

    source_file = pd.read_csv(ecpred, low_memory=False)
    source_file=source_file[source_file.sequence.str.contains('!!!')==False] #because 2 sequences in ecpred had !!!! to long to EXCEL... warning instead of sequence
    source_file=source_file[source_file.notna()] #have some nas (should be taken out in further datasets)
    sequences = source_file['sequence'].dropna().tolist()

    sequences_integer_ecoded=[]
    for seq1 in sequences:
        # seq1=seq.replace('X', '') # unknown character eliminated
        seq2=seq1.replace('B', 'N')  # asparagine N / aspartic acid  D - asx - B
        seq3=seq2.replace('Z', 'Q')  # glutamine Q / glutamic acid  E - glx - Z
        seq4=seq3.replace('U', 'C')  # selenocisteina, the closest is the cisteine. but it is a different aminoacid . take care.
        seq=seq4.replace('O', 'K')  # Pyrrolysine to lysine
        # define a mapping of chars to integers
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        int_to_char = dict((i, c) for i, c in enumerate(alphabet))
        # integer encode input data
        integer_encoded = [char_to_int[char] for char in seq]
        sequences_integer_ecoded.append(integer_encoded)

    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences
    # pad sequences
    list_of_sequences_length=pad_sequences(sequences_integer_ecoded, maxlen=len_seq, dtype='int32', padding='post', truncating='post',
                                           value=0.0)

    fps_x = list_of_sequences_length

    # fps_y
    ec_number = source_file['ec_number']  # loc just ec_number

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

    # to single label
    # take out lines that have multiple EC numbers (in both fps_x and fps_y)

    #single label
    fps_y = pd.DataFrame(firsts_no_duplicates, columns=['ec_number1', 'ec_number2', 'ec_number3'])  # 500759

    # count how many ec_number2 are NAn (more than 1 EC number to verify shape)
    print(fps_y['ec_number2'].isna().sum())  # 500027
    fps_x = pd.DataFrame(fps_x)
    df = fps_y.join(fps_x)
    new_dataset = df[df[['ec_number2']].isna().any(axis=1)]  # 500027
    fps_y = new_dataset['ec_number1']
    fps_x = new_dataset.drop(['ec_number1', 'ec_number2', 'ec_number3'], axis=1)
    # fps_y = [(map(int, x)) for x in fps_y]
    fps_y = [int(row) for row in fps_y]
    # from collections import Counter
    # Counter(fps_y)

    # multilabel
    # from sklearn.preprocessing import MultiLabelBinarizer
    #
    # mlb = MultiLabelBinarizer()
    # multilabeled = mlb.fit_transform(firsts_no_duplicates)
    #
    # fps_y = multilabeled
    # # fps_x= encoded.reshape(encoded.shape[0], 20000)

    x_train, x_test, x_dval, y_train, y_test, y_dval = divide_train_test(fps_x, fps_y)

    # y_train= y_train.to_numpy().reshape(y_train.shape[1],y_train.shape[0])
    # y_test= y_test.to_numpy().reshape(y_test.shape[1],y_test.shape[0])
    # y_dval= y_dval.to_numpy().reshape(y_dval.shape[1],y_dval.shape[0])

    # # loss 'categorical_crossentropy' asks for categorical shapes
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    # y_dval = to_categorical(y_dval)


    # with strategy.scope():
    aclstm_simple_feature(input_shape=input_shape, len_features=len_features,
                          trainmain=x_train, trainlabel=y_train,
                          valmain=x_dval, vallabel=y_dval,
                          testmain=x_test, test_label=y_test)





