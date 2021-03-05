"""
##############################################################################

File containing tests functions to check if all functions from machine_learning module are properly working

Authors: Ana Marta Sequeira

Date: 06/2019 altered 01/2021

Email:

##############################################################################
"""
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical, np_utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, LSTM, Flatten
from tensorflow.keras.layers import MaxPool1D, MaxPool2D, Conv1D, Conv2D
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.model_selection import train_test_split

from propythia.deep_ml import DeepML




# PAD ZEROS 200 20 aa  X = 0 categorcial encoding
def pad_sequence(df, seq_len=200, padding='pre'):
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

def test_deep_ml():
    # split dataset
    hot_90 = pd.read_csv(r'datasets/ecpred_uniprot_uniref_90.csv', delimiter=',')
    alphabet = "XARNDCEQGHILKMFPSTWYV"
    # FPS Yget the first level of enzymes. in multilabel only consider identical labels
    l = []
    for ec_list in hot_90['ec_number']:
        ec_1 = [x.strip()[0] for x in ec_list.split(';') ]
        l.append(list(set(ec_1)))
    hot_90['ec_number1']=l
    l = []
    for ec_list in hot_90['ec_number1']:
        ec_l = set(ec_list)
        l.append(ec_l)
    hot_90['ec_single_label']=l
    data = hot_90.loc[hot_90['ec_single_label'].apply(len)<2,:]
    data=data.dropna(subset=['sequence'])
    data = data[data['sequence'].str.contains('!!!') == False]

    data = data.sample(n=300) # to run faster. just 300 of dataset


    counts = Counter(x for xs in data['ec_single_label'] for x in set(xs))
    counts.most_common()
    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    df_sorted = df.sort_values(by=[0], ascending=False)

# index      0
# 0     2  54343
# 3     3  34484
# 4     0  22708
# 2     1  19066
# 7     6  16290
# 1     4  12924
# 5     5   8081
# 6     7   6860
    fps_y = data['ec_single_label']
    fps_y = [item for sublist in fps_y for item in sublist] # this line is because they are retrieved as a list
    encoder = LabelEncoder()
    encoder.fit(fps_y)
    fps_y_encoded = encoder.transform(fps_y)
    classes = encoder.classes_
    fps_y_bin = np_utils.to_categorical(fps_y_encoded)

    # FPS_X get sequence hot encoded
    seq_len = 100 # lower number to take less time
    fps_x_categ = pad_sequence(data, seq_len=seq_len, padding='pre')
    fps_x_hot2d = to_categorical(fps_x_categ)
    fps_x_hot1d = fps_x_hot2d.reshape(fps_x_hot2d.shape[0], fps_x_hot2d.shape[1]*fps_x_hot2d.shape[2])

    fps_x = fps_x_categ

    # divide dataset in train test validation
    x_train_1, x_test, y_train_1, y_test = train_test_split(fps_x, fps_y_encoded, test_size=0.20, random_state=42,
                                                           stratify=fps_y_encoded, shuffle=True)
    x_train, x_dval, y_train, y_dval = train_test_split(x_train_1, y_train_1, test_size=0.25, random_state=42,
                                                        stratify=y_train_1, shuffle=True)


    vector_size = x_train.shape[1]
    final_units = fps_y_bin.shape[1]

    print(vector_size)
    print(final_units)



    # RUN WITH A USER DEFINED MODEL

    # create DL object
    dl = DeepML(x_train, y_train, x_test, y_test,
                number_classes=final_units, problem_type='multiclass',
                x_dval=x_dval, y_dval=y_dval,
                model=None,
                epochs=5, batch_size=512, callbacks=None,
                reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
                early_stopping_patience=30, reduce_lr_patience=50, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
                path='', report_name=None, verbose=1,  validation_split=0.1, shuffle=True, class_weights=None)


    tf.debugging.set_log_device_placement(True)
    strategy = tf.distribute.MirroredStrategy()
    #input is 21 categorical 200 paded aa sequences
    from keras.layers.merge import concatenate

    def veltri_model(seq_len,final_units=8, output_dim = 32):
        # with strategy.scope():
        model = Sequential()
        model.add(Input(shape=(seq_len,)))
        # model.add(Embedding(input_dim=21, output_dim=output_dim, input_length=seq_len, mask_zero=True))
        # model.add(Conv1D(
        #     filters=32,
        #     kernel_size=8,
        #     strides=1,
        #     padding='same',
        #     activation='relu'))
        # model.add(MaxPool1D(pool_size=5, strides=1, padding='same'))
        # # model.add(Dense(256))
        # model.add(LSTM(units=16,
        #                dropout=0.1,
        #                unroll=True,
        #                return_sequences=False,
        #                stateful=False))
        model.add(Dense(64))
        model.add(Dense(32))
        # Add Classification Dense, Compile model and make it ready for optimization
        model.add(Dense(final_units, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


    model = KerasClassifier(build_fn=veltri_model, seq_len=seq_len)
    # run model
    dm = dl.run_model(model)
    # evaluate the model

    score_simple = dl.model_simple_evaluate()
    scores, report, cm, cm2 = dl.model_complete_evaluate()

    # evaluation curves
    dl.precision_recall_curve(show=False, path_save=None)
    # dl.roc_curve(ylim=(0.0, 1.00), xlim=(0.0, 1.0),
    #       title='Receiver operating characteristic (ROC) curve',
    #       path_save=None, show=True, batch_size=None)

    # other curves
    dl.plot_validation_curve(param_name='output_dim', param_range=[16,32], cv=5,
                          score=make_scorer(matthews_corrcoef), title="Validation Curve",
                          xlab="parameter range", ylab="score", n_jobs=1, show=False,
                          path_save=None)

    dl.plot_learning_curve(title='Learning curve', ylim=None, cv=3,
                    n_jobs=1, path_save=None, show=False, scalability=True, performance=False)

    # predict  (use the test daatset just to demonstrate how to perform. in real work sshould have a separate dataset)
    predict_df = dl.predict(x = x_test, seqs=None, classifier=None, names=None, true_y=y_test, batch = None)

    # save and reconstructed model saved
    dl.save_model(path='model.h5')
    dl.reconstructed_model=dl.load_model(path='model.h5')

    # get the model stored in class
    model = dl.get_model()

    # train a model CV
    scores_cv = dl.train_model_cv(model=model, x_cv=x_train_1, y_cv=y_train_1, cv=5)

    # do hyperparameter optimization . dataXand dataY None, it will use the x_train and y_train
    param_grid = {'output_dim':[32, 16,64]}
    best_model = dl.get_opt_params(param_grid,  model=model, optType='gridSearch', cv=3, dataX=None, datay=None,
                   n_iter_search=15, n_jobs=1, scoring=make_scorer(matthews_corrcoef))


    # based models
    # loss function and activation function can be changed or set automatically
    # parameters of model can be changed (number layers....)
    # dnn simple run



    fps_x = fps_x_hot1d
    input_dim = fps_x_hot1d.shape[1]
    x_train_1, x_test, y_train_1, y_test = train_test_split(fps_x, fps_y_encoded, test_size=0.20, random_state=42,
                                                            stratify=fps_y_encoded, shuffle=True)
    x_train, x_dval, y_train, y_dval = train_test_split(x_train_1, y_train_1, test_size=0.25, random_state=42,
                                                        stratify=y_train_1, shuffle=True)

    dl = DeepML(x_train, y_train, x_test, y_test,
                number_classes=final_units, problem_type='multiclass',
                x_dval=x_dval, y_dval=y_dval,
                model=None,
                epochs=5, batch_size=512, callbacks=None,
                reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
                early_stopping_patience=30, reduce_lr_patience=50, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
                path='', report_name=None, verbose=1,  validation_split=0.1, shuffle=True, class_weights=None)

    # dnn simple  cv = None optType=None
    dnn_simple = dl.run_dnn_simple(
                   input_dim=input_dim,
                   optimizer='Adam',
                   hidden_layers=(32, 16),
                   dropout_rate=(0.3,),
                   batchnormalization=(True,),
                   l1=1e-5, l2=1e-4,
                   final_dropout_value=0.3,
                   initial_dropout_value=0.0,
                   loss_fun=None, activation_fun=None,
                   cv=None, optType=None, param_grid=None, n_iter_search=15, n_jobs=1,
                   scoring=make_scorer(matthews_corrcoef))

    # dnn simple cv cv=something optType =None
    scores_dnn_simple_cv = dl.run_dnn_simple(
        input_dim=input_dim,
        optimizer='Adam',
        hidden_layers=(32, 16),
        dropout_rate=(0.3,),
        batchnormalization=(True,),
        l1=1e-5, l2=1e-4,
        final_dropout_value=0.3,
        initial_dropout_value=0.0,
        loss_fun=None, activation_fun=None,
        cv=3, optType=None, param_grid=None, n_iter_search=15, n_jobs=1,
        scoring=make_scorer(matthews_corrcoef))

    # dnn hyperopt cv and opttype filled. if randomizedSearch, number of iter search is 15 by default but can be changed
    # the grid search can be set or be the one established by default.

    dnn_simple_ho = dl.run_dnn_simple(
        input_dim=input_dim,
        optimizer='Adam',
        hidden_layers=(32,16),
        dropout_rate=(0.3,),
        batchnormalization=(True,),
        l1=1e-5, l2=1e-4,
        initial_dropout_value=0.0,
        loss_fun=None, activation_fun=None,
        cv=3, optType='randomizedSearch', param_grid=None, n_iter_search=3, n_jobs=1,
        scoring=make_scorer(matthews_corrcoef))


    # # same thing but dnn embedding
    # fps_x = fps_x_categ
    #
    # # divide dataset in train test validation
    # x_train_1, x_test, y_train_1, y_test = train_test_split(fps_x, fps_y_encoded, test_size=0.20, random_state=42,
    #                                                         stratify=fps_y_encoded, shuffle=True)
    # x_train, x_dval, y_train, y_dval = train_test_split(x_train_1, y_train_1, test_size=0.25, random_state=42,
    #                                                     stratify=y_train_1, shuffle=True)
    #
    #
    # input_dim = fps_x.shape[1]
    # final_units = fps_y_bin.shape[1]
    #
    # dl = DeepML(x_train, y_train, x_test, y_test,
    #             number_classes=final_units, problem_type='multiclass',
    #             x_dval=x_dval, y_dval=y_dval,
    #             model=None,
    #             epochs=5, batch_size=512, callbacks=None,
    #             reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
    #             early_stopping_patience=30, reduce_lr_patience=50, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
    #             path='', report_name=None, verbose=1,  validation_split=0.1, shuffle=True, class_weights=None)
    #
    # dnn_emb = dl.run_dnn_embedding(input_dim,
    #                   optimizer='Adam',
    #                   input_dim_emb=21, output_dim=32, input_length=seq_len, mask_zero=True,
    #                   hidden_layers=(32,16),
    #                   dropout_rate=(0.3,),
    #                   batchnormalization=(True,),
    #                   l1=1e-5, l2=1e-4,
    #                   loss_fun = None, activation_fun = None,
    #                   cv=None, optType=None, param_grid=None, n_iter_search=15, n_jobs=1,
    #                   scoring=make_scorer(matthews_corrcoef))
    #
    # # dnn_emb_cv = dl.run_dnn_embedding(input_dim,
    # #                                optimizer='Adam',
    # #                                input_dim_emb=21, output_dim=32, input_length=500, mask_zero=True,
    # #                                hidden_layers=(32,16),
    # #                                dropout_rate=(0.3,),
    # #                                batchnormalization=(True,),
    # #                                l1=1e-5, l2=1e-4,
    # #                                loss_fun = None, activation_fun = None,
    # #                                cv=3, optType=None, param_grid=None, n_iter_search=15, n_jobs=1,
    # #                                scoring=make_scorer(matthews_corrcoef))
    # #
    # # dnn_emb_ho = dl.run_dnn_embedding(input_dim,
    # #                                   optimizer='Adam',
    # #                                   input_dim_emb=21, output_dim=32, input_length=500, mask_zero=True,
    # #                                   hidden_layers=(32,16),
    # #                                   dropout_rate=(0.3,),
    # #                                   batchnormalization=(True,),
    # #                                   l1=1e-5, l2=1e-4,
    # #                                   loss_fun = None, activation_fun = None,
    # #                                   cv=3, optType='randomizedSearch', param_grid=None, n_iter_search=5, n_jobs=1,
    # #                                   scoring=make_scorer(matthews_corrcoef))
    #
    # # lstm simple
    # fps_x = fps_x_hot1d
    # input_dim = fps_x_hot1d.shape[1]
    # x_train_1, x_test, y_train_1, y_test = train_test_split(fps_x, fps_y_encoded, test_size=0.20, random_state=42,
    #                                                         stratify=fps_y_encoded, shuffle=True)
    # x_train, x_dval, y_train, y_dval = train_test_split(x_train_1, y_train_1, test_size=0.25, random_state=42,
    #                                                     stratify=y_train_1, shuffle=True)
    #
    # x_train, x_test, x_dval, y_train, y_test, y_dval = \
    #     map(lambda x: np.array(x, dtype=np.float), [x_train, x_test, x_dval, y_train, y_test, y_dval])
    #
    # x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])  # (300455, 1, 20000)
    # y_train = y_train.reshape(y_train.shape[0], 1)
    #
    # x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])  # (100152, 1, 20000)
    # y_test = y_test.reshape(y_test.shape[0], 1)
    #
    # x_dval = x_dval.reshape(x_dval.shape[0], 1, x_dval.shape[1])  # (100152, 1, 20000)
    # y_dval = y_dval.reshape(y_dval.shape[0], 1)
    #
    # dl = DeepML(x_train, y_train, x_test, y_test,
    #             number_classes=final_units, problem_type='multiclass',
    #             x_dval=x_dval, y_dval=y_dval,
    #             model=None,
    #             epochs=5, batch_size=512, callbacks=None,
    #             reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
    #             early_stopping_patience=30, reduce_lr_patience=50, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
    #             path='', report_name=None, verbose=1,  validation_split=0.1, shuffle=True, class_weights=None)
    #
    # lstm = dl.run_lstm_simple(input_dim,
    #                        optimizer='Adam',
    #                        bilstm=True,
    #                        lstm_layers=(32,16),
    #                        dense_layers=(16,),
    #                        activation='tanh',
    #                        recurrent_activation='sigmoid',
    #                        dense_activation="relu",
    #                        l1=1e-5, l2=1e-4,
    #                        dropout_rate=(0.3,), recurrent_dropout_rate=(0.3,),
    #                        dropout_rate_dense=(0.3,),
    #                        batchnormalization = (True,),
    #                        loss_fun = None, activation_fun = None,
    #                        cv=None, optType=None, param_grid=None, n_iter_search=15, n_jobs=1,
    #                        scoring=make_scorer(matthews_corrcoef))
    #
    # # lstm_cv = dl.run_lstm_simple(input_dim,
    # #                           optimizer='Adam',
    # #                           bilstm=True,
    # #                           lstm_layers=(32,16),
    # #                           dense_layers=(16,),
    # #                           activation='tanh',
    # #                           recurrent_activation='sigmoid',
    # #                           dense_activation="relu",
    # #                           l1=1e-5, l2=1e-4,
    # #                           dropout_rate=(0.3,), recurrent_dropout_rate=(0.3,),
    # #                           dropout_rate_dense=(0.3,),
    # #                           batchnormalization = (True,),
    # #                           loss_fun = None, activation_fun = None,
    # #                           cv=3, optType=None, param_grid=None, n_iter_search=15, n_jobs=1,
    # #                           scoring=make_scorer(matthews_corrcoef))
    # # lstm_hot = dl.run_lstm_simple(input_dim,
    # #                              optimizer='Adam',
    # #                              bilstm=False,
    # #                              lstm_layers=(32,16),
    # #                              dense_layers=(16,),
    # #                              activation='tanh',
    # #                              recurrent_activation='sigmoid',
    # #                              dense_activation="relu",
    # #                              l1=1e-5, l2=1e-4,
    # #                              dropout_rate=(0.3,), recurrent_dropout_rate=(0.3,),
    # #                              dropout_rate_dense=(0.3,),
    # #                              batchnormalization = (True,),
    # #                              loss_fun = None, activation_fun = None,
    # #                              cv=3, optType='randomizedSearch', param_grid=None, n_iter_search=3, n_jobs=1,
    # #                              scoring=make_scorer(matthews_corrcoef))
    #
    # # # same thing but lstm embedding
    # fps_x = fps_x_categ
    #
    # # divide dataset in train test validation
    # x_train_1, x_test, y_train_1, y_test = train_test_split(fps_x, fps_y_encoded, test_size=0.20, random_state=42,
    #                                                         stratify=fps_y_encoded, shuffle=True)
    # x_train, x_dval, y_train, y_dval = train_test_split(x_train_1, y_train_1, test_size=0.25, random_state=42,
    #                                                     stratify=y_train_1, shuffle=True)
    #
    #
    # input_dim = fps_x.shape[1]
    # final_units = fps_y_bin.shape[1]
    #
    # dl = DeepML(x_train, y_train, x_test, y_test,
    #             number_classes=final_units, problem_type='multiclass',
    #             x_dval=x_dval, y_dval=y_dval,
    #             model=None,
    #             epochs=3, batch_size=512, callbacks=None,
    #             reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
    #             early_stopping_patience=30, reduce_lr_patience=50, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
    #             path='', report_name=None, verbose=1,  validation_split=0.1, shuffle=True, class_weights=None)
    #
    # lstm_emb = dl.run_lstm_embedding(optimizer='Adam',
    #                    input_dim_emb=21, output_dim=32, input_length=seq_len, mask_zero=True,
    #                    bilstm=True,
    #                    lstm_layers=(32, 16),
    #                    activation='tanh',
    #                    recurrent_activation='sigmoid',
    #                    dropout_rate=(0.3,), recurrent_dropout_rate=(0.3,),
    #                    l1=1e-5, l2=1e-4,
    #                    dense_layers=(16,),
    #                    dense_activation="relu",
    #                    dropout_rate_dense = (0.3,),
    #                    batchnormalization = (True,),
    #                    loss_fun = None, activation_fun = None,
    #                    cv=None, optType=None, param_grid=None, n_iter_search=15, n_jobs=1,
    #                    scoring=make_scorer(matthews_corrcoef))
    # # lstm_emb_cv = dl.run_lstm_embedding(optimizer='Adam',
    # #                    input_dim_emb=21, output_dim=32, input_length=seq_len, mask_zero=True,
    # #                    bilstm=True,
    # #                    lstm_layers=(32, 16),
    # #                    activation='tanh',
    # #                    recurrent_activation='sigmoid',
    # #                    dropout_rate=(0.3,), recurrent_dropout_rate=(0.3,),
    # #                    l1=1e-5, l2=1e-4,
    # #                    dense_layers=(16,),
    # #                    dense_activation="relu",
    # #                    dropout_rate_dense = (0.3,),
    # #                    batchnormalization = (True,),
    # #                    loss_fun = None, activation_fun = None,
    # #                    cv=3, optType=None, param_grid=None, n_iter_search=15, n_jobs=1,
    # #                    scoring=make_scorer(matthews_corrcoef))
    # #
    # # lstm_emb_ho = dl.run_lstm_embedding(optimizer='Adam',
    # #                                     input_dim_emb=21, output_dim=10, input_length=seq_len, mask_zero=True,
    # #                                     bilstm=True,
    # #                                     lstm_layers=(32, 16),
    # #                                     activation='tanh',
    # #                                     recurrent_activation='sigmoid',
    # #                                     dropout_rate=(0.3,), recurrent_dropout_rate=(0.3,),
    # #                                     l1=1e-5, l2=1e-4,
    # #                                     dense_layers=(16,),
    # #                                     dense_activation="relu",
    # #                                     dropout_rate_dense = (0.3,),
    # #                                     batchnormalization = (True,),
    # #                                     loss_fun = None, activation_fun = None,
    # #                                     cv=3, optType='randomizedSearch', param_grid=None, n_iter_search=3, n_jobs=1,
    # #                                     scoring=make_scorer(matthews_corrcoef))
    #
    #
    # # CNN 1D
    # fps_x = fps_x_hot1d
    # input_dim = fps_x_hot1d.shape[1]
    # x_train_1, x_test, y_train_1, y_test = train_test_split(fps_x, fps_y_encoded, test_size=0.20, random_state=42,
    #                                                         stratify=fps_y_encoded, shuffle=True)
    # x_train, x_dval, y_train, y_dval = train_test_split(x_train_1, y_train_1, test_size=0.25, random_state=42,
    #                                                     stratify=y_train_1, shuffle=True)
    #
    # x_train, x_test, x_dval, y_train, y_test, y_dval = \
    #     map(lambda x: np.array(x, dtype=np.float), [x_train, x_test, x_dval, y_train, y_test, y_dval])
    #
    # x_train, x_test, x_dval = \
    #     map(lambda x: x.reshape(x.shape[0], 1, x.shape[1]), [x_train, x_test, x_dval])
    #
    # dl = DeepML(x_train, y_train, x_test, y_test,
    #             number_classes=final_units, problem_type='multiclass',
    #             x_dval=x_dval, y_dval=y_dval,
    #             model=None,
    #             epochs=5, batch_size=512, callbacks=None,
    #             reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
    #             early_stopping_patience=30, reduce_lr_patience=50, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
    #             path='', report_name=None, verbose=1,  validation_split=0.1, shuffle=True, class_weights=None)
    #
    # cnn1d_simple = dl.run_cnn_1D(input_dim,
    #            optimizer='Adam',
    #            filter_count=(16,32, 64),  # define number layers
    #            padding='same',
    #            strides=1,
    #            kernel_size=(3,),  # list of kernel sizes per layer. if number will be the same in all numbers
    #            cnn_activation='relu',
    #            kernel_initializer='glorot_uniform',
    #            dropout_cnn=(0.0, 0.2, 0.2),
    #            # list of dropout per cnn layer. if number will be the same in all numbers
    #            max_pooling=(True,),
    #            pool_size=(2,), strides_pool=1,
    #            data_format_pool='channels_first',
    #            dense_layers=(32,16),
    #            dense_activation="relu",
    #            dropout_rate=(0.3,),
    #            l1=1e-5, l2=1e-4,
    #            loss_fun = None, activation_fun = None,
    #            cv=None, optType=None, param_grid=None, n_iter_search=3, n_jobs=1,
    #            scoring=make_scorer(matthews_corrcoef))

    # cnn1d_cv = dl.run_cnn_1D(input_dim,
    #                              optimizer='Adam',
    #                              filter_count=(32, 64, 16),  # define number layers
    #                              padding='same',
    #                              strides=1,
    #                              kernel_size=(3,),  # list of kernel sizes per layer. if number will be the same in all numbers
    #                              cnn_activation='relu',
    #                              kernel_initializer='glorot_uniform',
    #                              dropout_cnn=(0.0, 0.2, 0.2),
    #                              # list of dropout per cnn layer. if number will be the same in all numbers
    #                              max_pooling=(True,),
    #                              pool_size=(2,), strides_pool=1,
    #                              data_format_pool='channels_first',
    #                              dense_layers=(32,16),
    #                              dense_activation="relu",
    #                              dropout_rate=(0.3,),
    #                              l1=1e-5, l2=1e-4,
    #                              loss_fun = None, activation_fun = None,
    #                              cv=3, optType=None, param_grid=None, n_iter_search=3, n_jobs=1,
    #                              scoring=make_scorer(matthews_corrcoef))
    # cnn1d_ho = dl.run_cnn_1D(input_dim,
    #                              optimizer='Adam',
    #                              filter_count=(32, 64, 128),  # define number layers
    #                              padding='same',
    #                              strides=1,
    #                              kernel_size=(3,),  # list of kernel sizes per layer. if number will be the same in all numbers
    #                              cnn_activation='relu',
    #                              kernel_initializer='glorot_uniform',
    #                              dropout_cnn=(0.0, 0.2, 0.2),
    #                              # list of dropout per cnn layer. if number will be the same in all numbers
    #                              max_pooling=(True,),
    #                              pool_size=(2,), strides_pool=1,
    #                              data_format_pool='channels_first',
    #                              dense_layers=(64, 32),
    #                              dense_activation="relu",
    #                              dropout_rate=(0.3,),
    #                              l1=1e-5, l2=1e-4,
    #                              loss_fun = None, activation_fun = None,
    #                              cv=3, optType='randomizedSearch', param_grid=None, n_iter_search=3, n_jobs=1,
    #                              scoring=make_scorer(matthews_corrcoef))

    # hybrid

    cnn_lstm_simple = dl.run_cnn_lstm(
                 input_dim,
                 optimizer='Adam',
                 filter_count=(64,32, 16),
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
                 lstm_layers=(32,16),
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 dropout_rate=(0.3,), recurrent_dropout_rate=(0.3,),
                 l1=1e-5, l2=1e-4,
                 dense_layers=(16,),
                 dense_activation="relu",
                 dropout_rate_dense=(0.0,),
                 batchnormalization=(True,),
                 loss_fun = None, activation_fun = None,
                 cv=None, optType=None, param_grid=None, n_iter_search=15, n_jobs=1,
                 scoring=make_scorer(matthews_corrcoef))

    # cnn_lstm_cv = dl.run_cnn_lstm(
    #     input_dim,
    #     optimizer='Adam',
    #     filter_count=(32, 64, 16),
    #     padding='same',
    #     strides=1,
    #     kernel_size=(3,),
    #     cnn_activation=None,
    #     kernel_initializer='glorot_uniform',
    #     dropout_cnn=(0.0,),
    #     max_pooling=(True,),
    #     pool_size=(2,), strides_pool=1,
    #     data_format_pool='channels_first',
    #     bilstm=True,
    #     lstm_layers=(32,16),
    #     activation='tanh',
    #     recurrent_activation='sigmoid',
    #     dropout_rate=(0.3,), recurrent_dropout_rate=(0.3,),
    #     l1=1e-5, l2=1e-4,
    #     dense_layers=(16,),
    #     dense_activation="relu",
    #     dropout_rate_dense=(0.0,),
    #     batchnormalization=(True,),
    #     loss_fun = None, activation_fun = None,
    #     cv=3, optType=None, param_grid=None, n_iter_search=15, n_jobs=1,
    #     scoring=make_scorer(matthews_corrcoef))
    #
    # cnn_lstm_ho = dl.run_cnn_lstm(
    #     input_dim,
    #     optimizer='Adam',
    #     filter_count=(32, 64, 16),
    #     padding='same',
    #     strides=1,
    #     kernel_size=(3,),
    #     cnn_activation=None,
    #     kernel_initializer='glorot_uniform',
    #     dropout_cnn=(0.0,),
    #     max_pooling=(True,),
    #     pool_size=(2,), strides_pool=1,
    #     data_format_pool='channels_first',
    #     bilstm=True,
    #     lstm_layers=(32,16),
    #     activation='tanh',
    #     recurrent_activation='sigmoid',
    #     dropout_rate=(0.3,), recurrent_dropout_rate=(0.3,),
    #     l1=1e-5, l2=1e-4,
    #     dense_layers=(16,),
    #     dense_activation="relu",
    #     dropout_rate_dense=(0.0,),
    #     batchnormalization=(True,),
    #     loss_fun = None, activation_fun = None,
    #     cv=3, optType='randomizedSearch', param_grid=None, n_iter_search=3, n_jobs=1,
    #     scoring=make_scorer(matthews_corrcoef))

if __name__ == '__main__':
    test_dl()