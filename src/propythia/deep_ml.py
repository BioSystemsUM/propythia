# -*- coding: utf-8 -*-
"""
##############################################################################

File containing a class intend to facilitate deep learning
The functions are based on the package tensorflow and keras.

Authors: Ana Marta Sequeira

Date: 12/2020

Email:

##############################################################################
"""

import os
import pickle
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.disable(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.FATAL)
import tensorflow as tf
import keras
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, log_loss, matthews_corrcoef, classification_report, \
    multilabel_confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import ShuffleSplit, train_test_split

from propythia.dl_basic_models.basic_models_dnn import create_dnn_embedding, create_dnn_simple
from propythia.dl_basic_models.basic_models_lstm import create_lstm_embedding, create_lstm_bilstm_simple
from propythia.dl_basic_models.basic_models_cnn import create_cnn_1D, create_cnn_2D
from propythia.dl_basic_models.basic_models_hybrid import create_cnn_lstm
from propythia.adjuv_functions.ml_deep.utils import timer, saveModel, loadDNN, validation, validation_multiclass
from propythia.adjuv_functions.ml_deep.parameters_deep import param_deep
from propythia.adjuv_functions.ml_deep.plot_curves_ml_propythia import plot_roc_curve, plot_precision_recall_curve, \
    plot_validation_curve, plot_learning_curve, plot_summary_loss, plot_summary_accuracy
from propythia.param_optimizer import ParamOptimizer

tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()


class DeepML:
    """
    Class used for performing train and evaluation of different DL models.
    """

    def __init__(self,
                 x_train, y_train,
                 x_test, y_test, number_classes=2, problem_type='binary',
                 x_dval=None, y_dval=None,
                 model=None,
                 epochs=100, batch_size=512,
                 callbacks=None,
                 reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
                 early_stopping_patience=30, reduce_lr_patience=50, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
                 path='',
                 report_name=None,
                 verbose=1,  validation_split=0.1,
                 shuffle=True, class_weights=None):

        """
        https://www.tensorflow.org/api_docs/python/tf/keras/Model
        :param x_train: dataset/array with features or encodings for training (x_train)
        :param y_train: labels for training (y_train)
        :param x_test: dataset/array with features or encodings for testing (x_test)
        :param y_test: labels for testing (y_test)
        :param number_classes: number of classes of the problem
        :param problem_type: problem type. 'binary', 'multiclass', 'multilabel'. Only used to run base models.
        Not necessary when running outside architectures.
        :param x_dval: dataset/array with features or encodings for vlidation (x_dval). If none,
        will retrive from train data using the param validation split.
        :param y_dval: labels for validation (y_dval)
        :param model: trained model. None by default.
        :param epochs: number of epochs to use when training DL model. default 100
        :param batch_size: batch size to use when trianing DL models. 512 default.
        :param callbacks: callbacks for training the model. If None (default), callbacks will be assigned.
        :param reduce_lr: if True (default),  Reduce learning rate will be added to callbacks
        :param early_stopping: If True (default) early stopping will be added to callbacks.
        :param checkpoint: If True (default) model checkpoint will be added to callbacks.
        :param tensorboard: If True (default) tensorboard extension will be added to callbacks. False by default
        :param early_stopping_patience: Early stopping parameter. number of epochs patience of no improvements
        to model stop. 30 by default.
        :param reduce_lr_patience: Reduce lr parameter. number of epochs with no improvement after which learning rate
        will be reduced 50 default.
        :param reduce_lr_factor: factor to reduce learning rate. 0.2 by default
        :param reduce_lr_min: lower bound on the learning rate. 0.00001 by default
        :param path: os path to store graphics and reports.
        :param report_name: if None will generate a txt file reporting results for the functions called inside class.
        :param verbose: verbose for model training . 1 default
        :param validation_split: fraction indicating the percentage to split train and validation. 0.1 by default.
        Only used if no validation (x_dval) is given.  The validation is retrieved from the last training examples.
        :param shuffle: whether to shuffle or not the data in traiing process. True by default.
        :param class_weights: class weights if to be used. None by default.
        """

        self.x_train = x_train
        self.x_test = x_test

        self.y_train = y_train
        self.y_test = y_test

        self.x_dval = x_dval
        self.y_dval = y_dval

        self.validation_split = validation_split

        self.callbacks = callbacks
        self.reduce_lr = reduce_lr
        self.early_stopping = early_stopping
        self.checkpoint = checkpoint
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_min = reduce_lr_min
        self.tensorboard = tensorboard

        self.path = path

        self.model = model
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.class_weights = class_weights
        self.history = None

        self.final_units = number_classes
        # self.final_units = len(y_train[0])
        # self.final_units = len(np.unique(y_train))
        # https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
        # https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
        # https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff
        if problem_type == 'binary': # binary problem ou se for multilabel
            self.loss = 'binary_crossentropy'
            self.activation = 'sigmoid'
            self.metrics = ['accuracy']
            self.final_units = 1
        elif problem_type == 'multiclass': # multiclass sem ser multilabel
            # if any([item for sublist in self.y_train for item in sublist if item not in [0,1]]): # labeled # todo put here fastest way
            #     self.loss = 'sparse_categorical_crossentropy' # labeled encoder
            # if not hot encoded but labeled encoder
            # else:
            self.loss = 'sparse_categorical_crossentropy'
            self.activation = 'softmax'
            self.metrics = ['accuracy']
        elif problem_type == 'multilabel':
            self.loss = 'binary_crossentropy'
            self.activation = 'sigmoid'
            self.metrics = ['accuracy']
        # print(problem_type)
        # print(self.loss)
        # print(self.activation)
        # print(self.metrics)
        # print(self.final_units)
        self.report_name = report_name
        self.y_test_pred = None
        self.y_test_proba = None
        saved_args = locals()
        if self.report_name:
            self._report(str(self.report_name))
            self._report(saved_args)

    def _report(self, info, dataframe=False, **kwargs):
        filename = str(self.report_name)
        with open(filename, 'a+') as file:
            if dataframe is True:
                info.to_csv(file, sep='\t', mode='a', **kwargs)
            elif isinstance(info, str):
                file.writelines(info)
            else:
                for l in info:
                    file.writelines('\n{}'.format(l))

    # todo create methods to get all this values?

    def get_model(self):
        """
        Getter function for returning the keras compile model
        :return: keras compiled model
        """
        if self.model:
            return self.model
        else:
            raise ValueError("No model was compiled.")

    # ==================================================================================================================
    #  TRAIN AND RUN MODELS
    # ==================================================================================================================
    def train_model(self):
        """
        Function to train model
        :return: None
        """
        if self.x_dval is not None:
            validation_data = (self.x_dval, self.y_dval)
        else:
            validation_data = None

        self.history = self.model.fit(x=self.x_train, y=self.y_train,
                                 batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose,
                                 callbacks=self.callbacks,
                                 validation_split=self.validation_split, validation_data=validation_data,
                                 shuffle=self.shuffle, class_weight=self.class_weights,
                                 sample_weight=None, initial_epoch=0, steps_per_epoch=None,
                                 validation_steps=None, validation_batch_size=None, validation_freq=1,
                                 max_queue_size=10, workers=1, use_multiprocessing=False)

        # if data comes from generator
        # model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
        # nb_val_samples=len(validation_samples), nb_epoch=self.epochs)

        # # write report
        # if self.report_name is not None:
        #     self._report(['===TRAIN MODELS===\n', self.train_model.__name__, saved_args])

    def generate_callbacks(self):
        """
        Creates the callback instances to be used in the model training process
        :return: None
        """
        if self.callbacks is None:
            callbacks_list = []
            if self.early_stopping:
                # simple early stopping
                # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                   patience=self.early_stopping_patience)
                callbacks_list.append(es)

            if self.reduce_lr:
                # reduce learning rate reduce_lr_factor=0.5
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=self.reduce_lr_factor,
                                              patience=self.reduce_lr_patience, min_lr=self.reduce_lr_min, verbose=1)
                callbacks_list.append(reduce_lr)

            if self.checkpoint:
                # checkpoint
                filepath = os.path.join(self.path, 'weights-{{epoch:02d}}-{{val_loss:.2f}}.hdf5')
                cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=0,
                                     save_best_only=True, save_weights_only=False,
                                     mode='auto', save_freq='epoch', period=1)
                callbacks_list.append(cp)

            if self.tensorboard:
                # TensorBoard is a tool for providing the measurements and visualizations needed during the machine
                # learning workflow. It enables tracking experiment metrics like loss and accuracy, visualizing the
                # model graph, projecting embeddings to a lower dimensional space, and much more.
                # https://www.tensorflow.org/tensorboard/get_started
                log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,
                                                                      write_graph=True,
                                                                      write_grads=False)
                callbacks_list.append(tensorboard_callback)
                # Start TensorBoard through the command line or within a notebook experience. The two interfaces are generally the same. In notebooks, use the %tensorboard line magic. On the command line, run the same command without "%".
                # %tensorboard --logdir logs/fit
            self.callbacks = callbacks_list
        # write report
        if self.report_name is not None:
            self._report(['===Callbacks===\n', self.generate_callbacks.__name__, self.callbacks])

    def run_model(self, model=None):
        """
        Function used to run DL model. It will train the DL model using callbacks.
        It will retrieve training and validation accuracy mean and  training and validation loss mean.
        It retrieves the performance plots for the model ( training and validation accuracy mean and
        training and validation loss mean) and the summary plot.
        :param model: Keras classifier object.
        :return: model compiled and history object
        """
        start = time.perf_counter()
        if model is None:
            model = self.model
        if self.model is None:
            self.model = model
        self.generate_callbacks()
        self.train_model()

        s1 = 'Training Accuracy mean: ', np.mean(self.history.history['accuracy'])
        history_dict = self.history.history
        # print(history_dict.keys())
        s2 = 'Validation Accuracy mean: ', np.mean(self.history.history['val_accuracy'])
        s3 = 'Training Loss mean: ', np.mean(self.history.history['loss'])
        s4 = 'Validation Loss mean: ', np.mean(self.history.history['val_loss'])

        print(s1,'\n', s2,'\n', s3,'\n',s4)

        self.model.model.summary()


        final = time.perf_counter()
        run_time = final - start
        # write report
        if self.report_name is not None:
            self._report(['===TRAIN MODELS===\n', self.run_model.__name__])
            self._report([s1,s2,s3,s4])
            l=[]
            self.model.model.summary(print_fn=lambda x: l.append(x))
            self._report(l)
            self._report(f"Finished {self.run_model.__name__} in {run_time:.4f} secs\n\n")

        # plot model
        if self.report_name:
            path_save_acc = str(self.report_name+'accuracy.png')
            path_save_loss = str(self.report_name+'loss.png')
        else:
            path_save_acc = None
            path_save_loss = None


        plot_summary_accuracy(self.history, path_save=path_save_acc, show=True)
        plot_summary_loss(self.history,path_save=path_save_loss, show=True)
        path_plot = str(self.path + str('plot_model_file.png'))
        plot_model(self.model.model, to_file=path_plot, show_shapes=True, show_layer_names=True)

        return self.model, self.history

    # @timer
    def train_model_cv(self, x_cv, y_cv, cv=5, model=None):
        """
        Function used to cross validate DL models.
        :param x_cv:  data to train model
        :param y_cv: labels of training data
        :param cv: number of folds to cross validation. 5 by default
        :param model: Keras classifier. If None, retrieves model from the class.
        :return: dataframe with scores per fold, and mean and standard deviation of those scores.
        """
        start = time.perf_counter()
        # https://www.kaggle.com/stefanie04736/simple-keras-model-with-k-fold-cross-validation
        if self.model is None:
            self.model = model
        # Define per-fold score containers
        scores_complete = []
        accuracy_list = []
        loss_list = []
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=22).split(x_cv, y_cv)
        # print(folds)
        for j, (train_idx, val_idx) in enumerate(folds):
            print('\nFold ', j)
            self.x_train = x_cv[train_idx]
            self.x_test = x_cv[val_idx]
            self.y_train = y_cv[train_idx]
            self.y_test = y_cv[val_idx]
            self.validation_split = 0.10

            self.generate_callbacks()
            self.train_model()
            # s1 = 'Training Accuracy mean: ', np.mean(self.history['accuracy'])
            # s2 = 'Validation Accuracy mean: ', np.mean(self.history['val_accuracy'])
            # s3 = 'Training Loss mean: ', np.mean(self.history['loss'])
            # s4 = 'Validation Loss mean: ', np.mean(self.history['val_loss'])
            #
            # print(s1,'\n', s2,'\n', s3,'\n',s4)
            # summary = model.model.summary()
            # score = self.model.model.evaluate(self.x_test, self.y_test, batch_size=None, verbose=1, sample_weight=None,
            #                                   steps=None,
            #                                   callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)

            scores, report, cm, cm2 = self.model_complete_evaluate(self.x_test, self.y_test)

            scores_complete.append(scores)
            metrics = scores.keys()
            # # write report
            # if self.report_name is not None:
            #     self._report(['\nFold ', j, '\n'])
            #     # self._report([s1,s2,s3,s4])
            #     # self._report(summary)
            #     self._report('test_score for {} fold'.format(j))
            #     self._report(score, dataframe=True)

        # melhorar esta parte. n chamar o complete evaluate. calcular scores fora. adicionar sensitivity and sensibility
        df = pd.DataFrame(scores_complete)
        df.loc['mean'] = df.mean()
        df.loc['std'] = df.std()


        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
        # # evaluate using n-fold cross validation uses the wrapper scikit learn, n consigo testar num dataset a parte
        # kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=7)
        # results = cross_val_score(self.model, x_cv, y_cv, cv=kfold, fit_params={'callbacks': self.callbacks})
        # print(results)
        # print(results.mean())
        # print("MLP: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
        #
        # # reportinf(self.history)

        final = time.perf_counter()
        run_time = final - start
        print(f"Finished {self.train_model_cv.__name__} in {run_time:.4f} secs\n\n")
        # write report
        if self.report_name is not None:
            self._report(['===TRAIN MODELS with CV===\n', self.train_model_cv.__name__])
            self._report(df, dataframe=True)
            self._report(df)
            self._report(f"Finished {self.train_model_cv.__name__} in {run_time:.4f} secs\n\n")
        return df

    def get_opt_params(self, param_grid,  model, optType='randomizedSearch', cv=10, dataX=None, datay=None,
                       n_iter_search=15, n_jobs=1,
                       scoring=make_scorer(matthews_corrcoef)):
        """
        Function that allows to perform gridSearchCV or randomizedSearchCV in DL models
        :param param_grid: param grid to use in the optimization.
        :param model: keras classifier. model to optimize
        :param optType: optimization type. accept 'randomizedSearch' (default) or 'gridSearch'
        :param cv: number of folds for cross validation. 10 by default.
        :param dataX: data to train the model. If None (default) it will use the train data defined in class
        :param datay: labels to train the model. If None (default) it will use the train labels defined in class
        :param n_iter_search: number of iterations to do when randomizedSearch is performed
        :param n_jobs: n_jobs CPUs to use. 1 by default
        :param scoring: {metrics instance} scoring function built from make_scorer() or a predefined value in string form
            (choose from the scikit-learn`scoring-parameters
            <http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_).
        :return: best model fit and transformed to train data
        """
        start = time.perf_counter()
        if dataX is None:
            dataX=self.x_train
            datay=self.y_train

        po = ParamOptimizer(estimator=model, optType=optType, paramDic=param_grid,
                            dataX=dataX, datay=datay,
                            cv=cv, n_iter_search=n_iter_search, n_jobs=n_jobs, scoring=scoring)

        gs = po.get_opt_params()
        # summaries
        list_to_write_top_3_models = po.report_top_models(gs)
        s1, s2, s3, df = po.report_all_models(gs) # metrics for the best model. dataframe with all
        # Set the best parameters to the best estimator
        best_classifier = gs.best_estimator_
        self.model = best_classifier
        model_optimized, history_optimized = self.run_model()
        # best_classifier_fit = best_classifier.fit(self.x_train, self.y_train)
        # dnn_simple = best_classifier_fit
        # self.model=dnn_simple
        final = time.perf_counter()
        run_time = final - start

        for line in list_to_write_top_3_models:
            print(line)
        print(s1)
        print('df')
        print(df)
        # write report
        if self.report_name is not None:
            self._report(list_to_write_top_3_models)
            self._report([s1, s2, s3,f"Finished get_opt_params in {run_time:.4f} secs\n\n"])
            self._report(df, dataframe=True, float_format='%.3f')
        return model_optimized

    def save_model(self, model=None, path='model.h5'):
        """
        Function to save model
        :param model: model compiled to save
        :param path: path where to save model. 'model.h5' by default.
        :return:  None
        """
        if model is None:
            model=self.model

        try:
            model.model.save(path)
            print('model saved at {}'.format(path))
        except:
            print('model not saved')

    def load_model(self, path=''):
        """
        Function to load saved models
        :param path: path where model is saved
        :return:  reconstructed model
        """
        reconstructed_model = keras.models.load_model(path)
        return reconstructed_model

    ####################################################################################################################
    # EVALUATE
    ####################################################################################################################
    # join this functions equal to shalow and deep
    def conf_matrix_seaborn_table(self, conf_matrix=None, classifier=None, path_save='', show=True,
                                  square=True, annot=True, fmt='d', cbar=False,batch_size=None, **params):
        """
        Function to retrieve confusion matrix using seaborn
        :param conf_matrix:
        :param classifier:
        :param path_save:
        :param show:
        :param square:
        :param annot:
        :param fmt:
        :param cbar:
        :param batch_size:
        :param params:
        :return:
        """

        plt.clf()
        if conf_matrix is None:
            y_pred = classifier.predict(self.x_test, batch_size=batch_size)
            mat = confusion_matrix(self.y_test, y_pred)
        else:
            mat = conf_matrix

        sns.heatmap(mat.T, square=square, annot=annot, fmt=fmt, cbar=cbar, **params)
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        if path_save is not None:
            plt.savefig(fname=path_save)
        if show is True:
            plt.show()
        plt.clf()

    def model_simple_evaluate(self, x_test=None, y_test=None, model=None):
        """
        Performs model evaluation with model evaluate function. Returns the loss value & metrics values for the model
        in test mode.
        :param x_test: x dataset where to test. If none will retrieve the dataset from the class.
        :param y_test: labels dataset to test. If none will retrieve the y_test from the class.
        :return: returns the score of the model evaluation list test loss, test acc
        e.g.[0.1309928148984909, 0.963699996471405]
        """
        if x_test is None:
            x_test = self.x_test
            y_test = self.y_test
        if model is None:
            model = self.model
        score = model.model.evaluate(x_test, y_test, batch_size=None, verbose=1,
                                          sample_weight=None, steps=None, callbacks=None,
                                          max_queue_size=10, workers=1, use_multiprocessing=False)
        s1 = ("test loss, test acc:", score)
        print(s1)
        # write report
        if self.report_name is not None:
            self._report(['===simple evaluate===\n', self.model_simple_evaluate.__name__, '\n', s1])

        return score

    def model_complete_evaluate(self, x_test=None, y_test=None, model=None):
        """
        Performs complete model evaluation using multiple scoring functions.
        :param x_test: x dataset where to test. If none will retrieve the dataset from the class.
        :param y_test: labels dataset to test. If none will retrieve the y_test from the class.
        :param model: classifier to evaluate. None by default. If None, it will retrieves the trained model from the class
        :return: dictionary containing scores, classification report, simple confusion matrix, confusion matrix by
        class (in binary classification this last object will retrieve empty)
        """
        saved_args = locals()
        if x_test is None:
            x_test = self.x_test
            y_test = self.y_test
        if model is None:
            model = self.model
        scores = {}
        cm2 = []

        # if batch_size == 1:
        #     y_pred = []
        #     y_prob = []
        #     # online forecast
        #     # https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/
        #     for i in range(x_test.shape[0]):
        #         testX, testy = x_test[i], np.array(y_test)[i]
        #         y_pre = model.predict(testX, batch_size=1)
        #         print(y_pre)
        #         y_pred.append(y_pre[0][0])
        #         y_pro = model.predict_proba(testX, batch_size=1)
        #         y_prob.append(y_pro[0][0])
        #     y_pred = list(chain.from_iterable(y_pred))
        #     y_prob = list(chain.from_iterable(y_prob))
        #
        #     print(y_pred)
        #     print(y_test)
        # else:
        y_pred = model.predict(x_test)
        try:
            y_prob = model.predict_proba(x_test)
        except:
            y_prob = model.predict(x_test)
            y_pred= np.argmax(y_prob,axis=1)

        scores['Accuracy'] = accuracy_score(y_test, y_pred)
        scores['MCC'] = matthews_corrcoef(y_test, y_pred)
        scores['log_loss'] = log_loss(y_test, y_prob)

        if self.final_units > 2:
            # multiclass
            scores['f1 score weighted'] = f1_score(y_test, y_pred, average='weighted')
            scores['f1 score macro'] = f1_score(y_test, y_pred, average='macro')
            scores['f1 score micro'] = f1_score(y_test, y_pred, average='micro')
            scores['roc_auc ovr'] = roc_auc_score(y_test, y_prob, average='weighted', multi_class='ovr')
            y_test_reshape = y_test.reshape(y_test.shape[0])  # roc auc ovo was giving error
            scores['roc_auc ovo'] = roc_auc_score(y_test_reshape, y_prob, average='weighted', multi_class='ovo')
            scores['precision'] = precision_score(y_test, y_pred, average='weighted')
            scores['recall'] = recall_score(y_test, y_pred, average='weighted')
            cm2 = multilabel_confusion_matrix(y_test, y_pred)

        else:
            # binary
            scores['f1 score'] = f1_score(y_test, y_pred)
            scores['roc_auc'] = roc_auc_score(y_test, y_pred)
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
            scores['Precision'] = precision
            scores['Recall'] = recall
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
            # scores['fdr'] = float(fp) / (tp + fp)
            scores['sn'] = float(tp) / (tp + fn)
            scores['sp'] = float(tn) / (tn + fp)

        report = classification_report(y_test, y_pred, output_dict=False)
        cm = confusion_matrix(y_test, y_pred)
        scores_df = pd.DataFrame(scores.keys(), columns=['metrics'])
        scores_df['scores'] = scores.values()

        if self.report_name is not None:
            self._report(['===SCORING TEST SET ===\n', self.model_complete_evaluate.__name__, saved_args,
                          'report\n', report,
                          '\n===confusion_matrix===\n', cm,
                          '\n===multilabel confusion matrix===\n', cm2,
                          '\n===scores report===\n'])

            self._report(scores_df, dataframe=True, float_format='%.4f', index=False)
            self.conf_matrix_seaborn_table(conf_matrix=cm, path_save=str(self.report_name + 'confusion_matrix.png'),
                                           show=False)

        # print("=== Confusion Matrix ===")
        # print(cm)
        # print("=== Classification Report ===")
        # print(report)
        # print('\n')
        # print(scores_df)
        self.y_test_pred = y_pred
        self.y_test_proba = y_prob

        return scores, report, cm, cm2

    # https://hackernoon.com/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier-2ecc6c73115a
    # ROC Curves summarize the trade-off between the true positive rate and false positive rate for a predictive model using different probability thresholds.
    # Precision-Recall curves summarize the trade-off between the true positive rate and the positive predictive value for a predictive model using different probability thresholds.
    # ROC curves are appropriate when the observations are balanced between each class, whereas precision-recall curves are appropriate for imbalanced datasets.

    @timer
    def precision_recall_curve(self, show=False, path_save='precision_recall_curve.png', batch_size=None):
        """
        Plot a Precision-Recall curve.
        PRC curves summarize the trade-off between the true positive rate and the positive predictive value for a
        predictive model using different probability thresholds. While ROC curves are appropriate when the observations
        are balanced between each class, PRC are appropriate for imbalanced datasets.
        :param show: whether to show the PRC. false by default
        :param path_save: path to save the plot. if None, will not be stored.
        :param batch_size: batch size to calculate the PRC. None by default
        :return:
        """
        y = self.y_test
        # y_pred = self.model.predict(self.x_test)
        if batch_size == 1:
            y_pred = []
            y_prob = []
            # online forecast
            # https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/
            for i in range(self.x_test.shape[0]):
                testX, testy = self.x_test[i], np.array(y)[i]
                y_pro = self.model.predict_proba(testX, batch_size=1)
                y_prob.append(y_pro)
        else:
            y_prob = self.model.predict_proba(self.x_test, batch_size=batch_size)
        # y_prob = self.model.predict_proba(self.x_test, batch_size=batch_size)
        # todo attentio to this ! isnecessary to binarize. and if it is already binarize? should force the input to be binarized or just through an error in the functions that needed to so
        # take this out and receive binarize in the begining
        # checkk the functions iin the models the sparse categorical.
        # Use label_binarize to be multi-label like settings
        # and careful with the binary. take this out of here
        Y = label_binarize(y, classes=np.unique(self.y_train))

        plot_precision_recall_curve(Y, y_prob, self.final_units, show=show, path_save=path_save)

    @timer
    def roc_curve(self, classifier=None, ylim=(0.0, 1.00), xlim=(0.0, 1.0),
                  title='Receiver operating characteristic (ROC) curve',
                  path_save='plot_roc_curve.png', show=False, batch_size=None):
        """
        Plot receiver operating characteristic (ROC) curve.
        ROC Curves summarize the trade-off between the true positive rate and false positive rate for a predictive
        model using different probability thresholds.
        ROC curves are appropriate when the observations are balanced between each class.
        :param classifier: classifier instance. If None, will retrieve the trained one from the class.
        :param ylim: y axis limits
        :param xlim: x axis limits
        :param title: title of the plot. 'Receiver operating characteristic (ROC) curve' by default
        :param path_save: path where to save the plot.  'plot_roc_curve.png' by default
        :param show: Whether to display the produced plot or not. False by default.
        :param batch_size: batch size to use. None by default.
        :return: None
        """
        if classifier is None:
            classifier = self.model

        # attention that do the binarizer inside the function

        plot_roc_curve(classifier, self.final_units, self.x_test, self.y_test, self.x_train, self.y_train, ylim=ylim,
                       xlim=xlim,
                       title=title,
                       path_save=path_save, show=show)

    ###################################################################################################################
    # PREDICT
    ###################################################################################################################

    @timer
    def predict(self, x, seqs=None, classifier=None, names=None, true_y=None, batch = None):
        """
        This function can be used to predict with a trained classifier model. The function returns a
        'pandas.DataFrame' with predictions using the specified estimator and tests data.
        :param x: {array} descriptor values of the peptides to be predicted.
        :param seqs: {list} sequences of the peptides in ``x``.
        :param classifier: {classifier instance} classifier used for predictions. When None ( by default)will use the
        trained one inside class.
        :param names: {list} (optional) names of the peptides in ``x``.
        :param true_y: {array} (optional) true (known) classes of the peptides.
        :param batch: batch size to perform the prediction. None by default.
        :return: ``pandas.DataFrame`` containing predictions for ``x``
       """
        saved_args = locals()
        if classifier is None:
            classifier = self.model

        predict = classifier.predict(x, batch_size=batch)

        predict_df = pd.DataFrame(columns=['names', 'sequence', 'class predicted', 'True classes'])
        predict_df['names'] = names
        predict_df['True classes'] = true_y
        predict_df['sequence'] = seqs
        predict_df['class predicted'] = predict
        # remove columns with Nans all (cols that were None in entry
        predict_df = predict_df.dropna(axis=1)

        try:
            preds = classifier.predict_proba(x, batch_size=batch)
            cols_prob = ['prob_class_{}'.format(x+1) for x in range(self.final_units)]
            preds_df = pd.DataFrame(preds, columns=cols_prob)
            predict_df[cols_prob] = preds
            predict_df = predict_df.round(4)
        except Exception as e:
            print('classifier does not support predict probabilities')
            print(str(e))
            return predict_df

        if self.report_name is not None:
            self._report([self.predict.__name__, saved_args])
            predict_df.to_csv(str(self.report_name + 'predict' + time.strftime("-%Y%m%-%H%M%S.csv")))

        return predict_df

    ####################################################################################################################
    # MODEL CURVES
    # ##################################################################################################################
    # if tensorboard on jupyter is on dont know if it is necessary

    # plot validation curve makes sense?
    @timer
    def plot_validation_curve(self, param_name, param_range,
                              classifier=None,
                              cv=5,
                              score=make_scorer(matthews_corrcoef), title="Validation Curve",
                              xlab="parameter range", ylab="score", n_jobs=1, show=False,
                              path_save='plot_validation_curve', **params):

        """
        This function plots a cross-validation curve for the specified classifier on all tested parameters given in the
        option 'param_range'.

        :param classifier: {classifier instance} classifier or validation curve. If None will use the classifier trianed inside class.
        :param param_name: {string} parameter to assess in the validation curve plot. For example,
        deep learning examples
        :param param_range: {list} parameter range for the validation curve.
        :param cv: {int} number of folds for cross-validation.
        :param score: {metrics instance} scoring function built from make_scorer() or a predefined value in string form
            `sklearn.model_evaluation.scoring-parameter
            <http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_.
        :param title: {str} graphic title. "Validation Curve" by default
        :param xlab: {str} x axis label. "parameter range" by default.
        :param ylab: {str} y axis label. "score" by default
        :param n_jobs: {int} number of parallel jobs to use for calculation. if -1, all available cores are used. 1 by default.
        :param path_save: str. path to save the plot. 'plot_validation_curve' by default.
        :return: plot of the validation curve.
        """
        if classifier is None:
            classifier = self.model

        plot_validation_curve(classifier, self.x_train, self.y_train, param_name, param_range,
                              cv=cv, score=score,
                              title=title, xlab=xlab, ylab=ylab,
                              n_jobs=n_jobs,
                              show=show, path_save=path_save,
                              **params)

    @timer
    def plot_learning_curve(self, classifier=None, title='Learning curve', ylim=None,
                            cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5),
                            path_save='plot_learning_curve', show=False, scalability=True, performance=True):
        """
        Plot a learning curve to determine cross validated training and tests scores for different training set sizes.
        It retrieves graphic representing learning curves, numbers of trainig examples, scores on training sets,
        and scores on tests set and scalability and performance plots if set to True.
        :param classifier: {classifier instance} classifier. If None (default) uses the classifier inside the class.
        :param title: title of the plot
        :param ylim: None by default.
        :param cv: cross validation to use. ShuffleSplit(n_splits=10, test_size=0.2, random_state=0) by default.
        :param n_jobs:  number of parallel jobs to use for calculation. if -1, all available cores are used. default 1.
        :param train_sizes: train sizes to tests
        :param path_save: path to save the plot. If None , lot is not saved. 'plot_learning_curve' by default.
        :param show: Whether to display the graphic. False by default.
        :param scalability: whether to retrieve scalability plots. Fit times per training examples.
        :param performance: whether to retrieve performance plots . Plot fit_time vs score
        :return: None
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
        """
        if classifier is None:
            classifier = self.model

        plot_learning_curve(classifier, self.x_train, self.y_train, title=title, ylim=ylim,
                            cv=cv,
                            n_jobs=n_jobs, train_sizes=train_sizes,
                            path_save=path_save, show=show, scalability=scalability, performance=performance)

    ####################################################################################################################
    # RUN PREDEFINED MODELS
    # ##################################################################################################################
    # todo passar para outra class?
    def _get_opt_params(self, param_grid, func_name, model, optType, cv, n_iter_search, n_jobs, scoring):
        start = time.perf_counter()
        # retrieve default grids
        if param_grid is None:
            param = param_deep()
            if optType == 'gridSearch':
                param_grid = param[func_name]['param_grid']
            else:
                param_grid = param[func_name]['distribution']

        po = ParamOptimizer(estimator=model, optType=optType, paramDic=param_grid,
                            dataX=self.x_train, datay=self.y_train,
                            cv=cv, n_iter_search=n_iter_search, n_jobs=n_jobs, scoring=scoring)

        gs = po.get_opt_params()
        # summaries
        list_to_write_top_3_models = po.report_top_models(gs)
        s1, s2, s3, df = po.report_all_models(gs) # metrics for the best model. dataframe with all
        # Set the best parameters to the best estimator
        best_classifier = gs.best_estimator_
        self.model = best_classifier
        model_optimized, history_optimized = self.run_model()
        # best_classifier_fit = best_classifier.fit(self.x_train, self.y_train)
        # dnn_simple = best_classifier_fit
        # self.model=dnn_simple
        final = time.perf_counter()
        run_time = final - start

        print(list_to_write_top_3_models)
        print(s1)
        print('df')
        print(df)
        # write report
        if self.report_name is not None:
            self._report(list_to_write_top_3_models)
            self._report([s1, s2, s3,f"Finished {func_name} in {run_time:.4f} secs\n\n"])
            self._report(df, dataframe=True, float_format='%.3f')
        return model_optimized

    def run_dnn_simple(self,
                       input_dim,
                       optimizer='Adam',
                       hidden_layers=(128, 64),
                       dropout_rate=(0.3,),
                       batchnormalization=(True,),
                       l1=1e-5, l2=1e-4,
                       final_dropout_value=0.3,
                       initial_dropout_value=0.0,
                       loss_fun=None, activation_fun=None,
                       cv=None, optType=None, param_grid=None, n_iter_search=15, n_jobs=1,
                       scoring=make_scorer(matthews_corrcoef)):

        """
        It runs a home network based on stack of Dense layers. It is possible to run a single network, a cross validation
        score or to hyperparameter optimized the network ( with user defined grids or the default ones)
        :param input_dim: input dim for the network
        :param optimizer: optimizer to use when compiling the model. 'Adam' by default.
        :param hidden_layers: tuple. hidden layers to consider. Default (128,64) will run a etwork with 2 hidden layers
        with 128, followed by one with 64.
        :param dropout_rate: tuple. dropout rate to be used after each layer of dense. If dropout rate is only one number,
        it will be propagated to the dense layers existent. By default (0.3,). It will introduces 0.3 dropout rate after
         the first and after the second dense layer.
        :param batchnormalization: tuple. wheteher to use or not batch normalizzation after each dense layer. If
        is only one value it will be propagated to the dense layers existent. By default (True,). It will introduces
        bactchNorm after the first and after the second dense layer.
        :param l1: l1 regularization value
        :param l2: l2 regularization value
        :param final_dropout_value: whether to introduce a final dropout rate before classificaton layer. 0.3 by default.
        :param initial_dropout_value: whether to introduce an initial dropout rate before first layer. 0.0 by default
        :param loss_fun: loss function to consider when compiled by model. If None ( default) it will be retrieved
        accordingly to the problem type (if is binary, multiclass...)
        :param activation_fun: activation function to consider in classification layer. If None ( default) it will be retrieved
        accordingly to the problem type (if is binary, multiclass...)
        :param cv: number of folds to consider when doing cross validation or hyperparameter optimization. If None it will
        generate the trainig f a single model.
        :param optType: optimization type to perform. Whether 'gridSearch' or 'randomizedSearch'. If None no optimization is done.
        :param param_grid: param grid to use when optType is not none. If none, param grid by default are retrieved.
        :param n_iter_search: number of iterations in randomizedSearch
        :param n_jobs: 1 by default
        :param scoring: {metrics instance} scoring function built from make_scorer() or a predefined value in string form
            `sklearn.model_evaluation.scoring-paramete
        :return: dnn trained. This dnn trained may be from a single process or the best dnn in the grid space.
        In the case of cross validation, will also retrieve dataframe with scores
        """
        func_name = self.run_dnn_simple.__name__
        if loss_fun is None: loss_fun = self.loss
        if activation_fun is None: activation_fun = self.activation
        saved_args = locals()
        print('dnn model simple')
        model = KerasClassifier(build_fn=create_dnn_simple, number_classes=self.final_units,
                                input_dim=input_dim,
                                optimizer=optimizer,
                                hidden_layers=hidden_layers,
                                dropout_rate=dropout_rate,
                                batchnormalization=batchnormalization,
                                l1=l1, l2=l2,
                                initial_dropout_value=initial_dropout_value,
                                loss_fun = loss_fun, activation_fun = activation_fun,
                                nb_epoch=self.epochs, batch_size=self.batch_size,
                                verbose=self.verbose)

        if cv is not None and optType is None: # will do cross validation of the model
            if self.report_name is not None:
                self._report(['===Train basic models: \n', self.run_dnn_simple.__name__, saved_args])
            scores = self.train_model_cv(self.x_train, self.y_train, cv=cv, model=model)
            dnn_simple = self.run_model(model)
            return scores

        elif optType is not None: # will do param_optimization
            # retrieve default grids
            if self.report_name is not None:
                self._report(['===train models with {} param optimization===\n'.format(optType),
                              func_name, saved_args])
            dnn_simple = self._get_opt_params(param_grid, func_name, model, optType, cv, n_iter_search, n_jobs, scoring)

        else:
            if self.report_name is not None:
                self._report(['===Train basic models: \n', func_name, saved_args])
            dnn_simple = self.run_model(model)

        return dnn_simple

    # @timer
    def run_dnn_embedding(self, input_dim,
                          optimizer='Adam',
                          input_dim_emb=21, output_dim=256, input_length=1000, mask_zero=True,
                          hidden_layers=(128, 64),
                          dropout_rate=(0.3,),
                          batchnormalization=(True,),
                          l1=1e-5, l2=1e-4,
                          loss_fun = None, activation_fun = None,
                          cv=None, optType=None, param_grid=None, n_iter_search=15, n_jobs=1,
                          scoring=make_scorer(matthews_corrcoef)):

        func_name = self.run_dnn_embedding.__name__
        if loss_fun is None: loss_fun = self.loss
        if activation_fun is None: activation_fun = self.activation
        saved_args = locals()
        print('dnn model embedding')
        model = KerasClassifier(build_fn=create_dnn_embedding, number_classes=self.final_units,
                                input_dim=input_dim,
                                optimizer=optimizer,
                                input_dim_emb=input_dim_emb, output_dim=output_dim, input_length=input_length,
                                mask_zero=mask_zero,
                                hidden_layers=hidden_layers,
                                dropout_rate=dropout_rate,
                                batchnormalization=batchnormalization,
                                l1=l1, l2=l2,
                                loss_fun = loss_fun, activation_fun = activation_fun,
                                nb_epoch=self.epochs, batch_size=self.batch_size,
                                verbose=self.verbose)

        if cv is not None and optType is None: # will do cross validation of the model
            dnn_embedding = self.train_model_cv(self.x_train, self.y_train, cv=cv, model=model)
        elif optType is not None: # will do param_optimization
            if self.report_name is not None:
                self._report(['===train models with {} param optimization===\n'.format(optType),
                              func_name, saved_args])
            dnn_embedding = self._get_opt_params(param_grid, func_name, model, optType, cv, n_iter_search, n_jobs, scoring)

        else:
            if self.report_name is not None:
                self._report(['===Train basic models: \n', func_name, saved_args])
            dnn_embedding = self.run_model(model)

        return dnn_embedding

    def run_lstm_simple(self, input_dim,
                        optimizer='Adam',
                        bilstm=True,
                        lstm_layers=(128, 64),
                        dense_layers=(64,),
                        activation='tanh',
                        recurrent_activation='sigmoid',
                        dense_activation="relu",
                        l1=1e-5, l2=1e-4,
                        dropout_rate=(0.3,), recurrent_dropout_rate=(0.3,),
                        dropout_rate_dense=(0.3,),
                        batchnormalization = (True,),
                        loss_fun = None, activation_fun = None,
                        cv=None, optType=None, param_grid=None, n_iter_search=15, n_jobs=1,
                        scoring=make_scorer(matthews_corrcoef)):

        func_name = self.run_lstm_simple.__name__
        if loss_fun is None: loss = self.loss
        if activation_fun is None: activation_fun = self.activation
        saved_args = locals()
        print('lstm simple model')

        model = KerasClassifier(build_fn=create_lstm_bilstm_simple,
                                number_classes=self.final_units,
                                input_dim=input_dim,
                                optimizer=optimizer,
                                bilstm=bilstm,
                                lstm_layers=lstm_layers,
                                dense_layers=dense_layers,
                                activation=activation,
                                recurrent_activation=recurrent_activation,
                                dense_activation=dense_activation,
                                dropout_rate=dropout_rate, recurrent_dropout_rate=recurrent_dropout_rate,
                                dropout_rate_dense=dropout_rate_dense,
                                batchnormalization = batchnormalization,
                                l1=l1, l2=l2,
                                loss_fun = loss_fun, activation_fun = activation_fun,
                                nb_epoch=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

        if cv is not None and optType is None: # will do cross validation of the model
            lstm_simple = self.train_model_cv(self.x_train, self.y_train, cv=cv, model=model)

        elif optType is not None: # will do param_optimization
            if self.report_name is not None:
                self._report(['===train models with {} param optimization===\n'.format(optType),
                              func_name, saved_args])
            lstm_simple = self._get_opt_params(param_grid, func_name, model, optType, cv, n_iter_search, n_jobs, scoring)

        else:
            if self.report_name is not None:
                self._report(['===Train basic models: \n', func_name, saved_args])
            lstm_simple = self.run_model(model)

        return lstm_simple

    def run_lstm_embedding(self,
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
                           loss_fun = None, activation_fun = None,
                           cv=None, optType=None, param_grid=None, n_iter_search=15, n_jobs=1,
                           scoring=make_scorer(matthews_corrcoef)):


        func_name = self.run_lstm_embedding.__name__
        if loss_fun is None: loss_fun = self.loss
        if activation_fun is None: activation_fun = self.activation
        saved_args = locals()
        print('lstm embedding model')
        model = KerasClassifier(build_fn=create_lstm_embedding, number_classes=self.final_units,
                                optimizer=optimizer,
                                input_dim_emb=input_dim_emb, output_dim=output_dim, input_length=input_length,
                                mask_zero=mask_zero,
                                bilstm=bilstm,
                                lstm_layers=lstm_layers,
                                dense_layers=dense_layers,
                                activation=activation,
                                recurrent_activation=recurrent_activation,
                                dense_activation=dense_activation,
                                dropout_rate=dropout_rate, recurrent_dropout_rate=recurrent_dropout_rate,
                                l1=l1, l2=l2,
                                dropout_rate_dense = dropout_rate_dense,
                                batchnormalization = batchnormalization,
                                loss_fun = loss_fun, activation_fun = activation_fun,
                                nb_epoch=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

        if cv is not None and optType is None: # will do cross validation of the model
            lstm_embedding = self.train_model_cv(self.x_train, self.y_train, cv=cv, model=model)

        elif optType is not None: # will do param_optimization
            if self.report_name is not None:
                self._report(['===train models with {} param optimization===\n'.format(optType),
                              func_name, saved_args])
            lstm_embedding = self._get_opt_params(param_grid, func_name, model, optType, cv, n_iter_search, n_jobs, scoring)

        else:
            if self.report_name is not None:
                self._report(['===Train basic models: \n', func_name, saved_args])
            lstm_embedding = self.run_model(model)

        return lstm_embedding

    def run_cnn_lstm(self,
                     input_dim,
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
                     loss_fun = None, activation_fun = None,
                     cv=None, optType=None, param_grid=None, n_iter_search=15, n_jobs=1,
                     scoring=make_scorer(matthews_corrcoef)):

        func_name = self.run_cnn_lstm.__name__
        if loss_fun is None: loss_fun = self.loss
        if activation_fun is None: activation_fun = self.activation
        saved_args = locals()

        print('cnn lstm model')
        model = KerasClassifier(build_fn=create_cnn_lstm,
                                number_classes=self.final_units,
                                input_dim=input_dim,
                                optimizer=optimizer,
                                filter_count=filter_count,
                                padding=padding,
                                strides=strides,
                                kernel_size=kernel_size,
                                cnn_activation=cnn_activation,
                                kernel_initializer=kernel_initializer,
                                dropout_cnn=dropout_cnn,
                                max_pooling=max_pooling,
                                pool_size=pool_size, strides_pool=strides_pool,
                                data_format_pool=data_format_pool,
                                bilstm=bilstm,
                                lstm_layers=lstm_layers,
                                dropout_rate=dropout_rate, recurrent_dropout_rate=recurrent_dropout_rate,
                                activation=activation, recurrent_activation=recurrent_activation,
                                dense_layers=dense_layers,
                                dense_activation=dense_activation,
                                l1=l1, l2=l2,
                                dropout_rate_dense=dropout_rate_dense,
                                batchnormalization=batchnormalization,
                                loss_fun = loss_fun, activation_fun = activation_fun,
                                nb_epoch=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

        if cv is not None and optType is None: # will do cross validation of the model
            cnn_lstm = self.train_model_cv(self.x_train, self.y_train, cv=cv, model=model)

        elif optType is not None: # will do param_optimization
            if self.report_name is not None:
                self._report(['===train models with {} param optimization===\n'.format(optType),
                              func_name, saved_args])
            cnn_lstm = self._get_opt_params(param_grid, func_name, model, optType, cv, n_iter_search, n_jobs, scoring)

        else:
            if self.report_name is not None:
                self._report(['===Train basic models: \n', func_name, saved_args])

            cnn_lstm = self.run_model(model)
        return cnn_lstm

    @timer
    def run_cnn_1D(self, input_dim,
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
                   loss_fun = None, activation_fun = None,
                   cv=None, optType=None, param_grid=None, n_iter_search=15, n_jobs=1,
                   scoring=make_scorer(matthews_corrcoef)):
        func_name = self.run_cnn_1D.__name__
        if loss_fun is None: loss_fun = self.loss
        if activation_fun is None: activation_fun = self.activation
        saved_args = locals()

        print('cnn 1D model')
        model = KerasClassifier(build_fn=create_cnn_1D, input_dim=input_dim,
                                number_classes=self.final_units,
                                optimizer=optimizer,
                                filter_count=filter_count,  # define number layers
                                padding=padding,
                                strides=strides,
                                kernel_size=kernel_size,
                                # list of kernel sizes per layer. if number will be the same in all numbers
                                cnn_activation=cnn_activation,
                                kernel_initializer=kernel_initializer,
                                dropout_cnn=dropout_cnn,
                                # list of dropout per cnn layer. if number will be the same in all numbers
                                max_pooling=max_pooling,
                                pool_size=pool_size, strides_pool=strides_pool,
                                data_format_pool=data_format_pool,
                                dense_layers=dense_layers,
                                dense_activation=dense_activation,
                                dropout_rate=dropout_rate,
                                l1=l1, l2=l2,
                                loss_fun = loss_fun, activation_fun = activation_fun,
                                nb_epoch=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

        if cv is not None and optType is None: # will do cross validation of the model
            cnn_1d = self.train_model_cv(self.x_train, self.y_train, cv=cv, model=model)
        elif optType is not None: # will do param_optimization
            if self.report_name is not None:
                self._report(['===train models with {} param optimization===\n'.format(optType),
                              func_name, saved_args])
            cnn_1d = self._get_opt_params(param_grid, func_name, model, optType, cv, n_iter_search, n_jobs, scoring)

        else:
            if self.report_name is not None:
                self._report(['===Train basic models: \n', func_name, saved_args])

            cnn_1d = self.run_model(model)

        return cnn_1d

    @timer
    def run_cnn_2D(self, input_dim,
                   optimizer='Adam',
                   filter_count=(32, 64, 128),  # define number layers
                   padding='same',
                   strides=1,
                   kernel_size=((3,3),),  # list of kernel sizes per layer. if number will be the same in all numbers
                   cnn_activation='relu',
                   kernel_initializer='glorot_uniform',
                   dropout_cnn=(0.0, 0.2, 0.2),
                   # list of dropout per cnn layer. if number will be the same in all numbers
                   max_pooling=(True,),
                   pool_size=((2,2),), strides_pool=1,
                   data_format_pool='channels_first',
                   dense_layers=(64, 32),
                   dense_activation="relu",
                   dropout_rate=(0.3,),
                   l1=1e-5, l2=1e-4,
                   loss_fun = None, activation_fun = None,
                   cv=None, optType=None, param_grid=None, n_iter_search=15, n_jobs=1,
                   scoring=make_scorer(matthews_corrcoef)):
        func_name = self.run_cnn_2D.__name__
        if loss_fun is None: loss_fun = self.loss
        if activation_fun is None: activation_fun = self.activation
        saved_args = locals()

        print('cnn 2D model')
        model = KerasClassifier(build_fn=create_cnn_2D, input_dim=input_dim, number_classes=self.final_units,
                                optimizer=optimizer,
                                filter_count=filter_count,  # define number layers
                                padding=padding,
                                strides=strides,
                                kernel_size=kernel_size,
                                # list of kernel sizes per layer. if number will be the same in all numbers
                                cnn_activation=cnn_activation,
                                kernel_initializer=kernel_initializer,
                                dropout_cnn=dropout_cnn,
                                # list of dropout per cnn layer. if number will be the same in all numbers
                                max_pooling=max_pooling,
                                pool_size=pool_size, strides_pool=strides_pool,
                                data_format_pool=data_format_pool,
                                dense_layers=dense_layers,
                                dense_activation=dense_activation,
                                dropout_rate=dropout_rate,
                                l1=l1, l2=l2, loss_fun = loss_fun, activation_fun = activation_fun,
                                nb_epoch=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

        if cv is not None and optType is None: # will do cross validation of the model
            cnn_2d = self.train_model_cv(self.x_train, self.y_train, cv=cv, model=model)

        elif optType is not None: # will do param_optimization
            if self.report_name is not None:
                self._report(['===train models with {} param optimization===\n'.format(optType),
                              func_name, saved_args])
            cnn_2d = self._get_opt_params(param_grid, func_name, model, optType, cv, n_iter_search, n_jobs, scoring)

        else:
            if self.report_name is not None:
                self._report(['===Train basic models: \n', func_name, saved_args])

            cnn_2d = self.run_model(model)

        return cnn_2d

    ####################################################################################################################
    # Hyperparameter optimization
    ####################################################################################################################
    # todo
    # using other tools other than grid/randomized search for both machine learning shallow and deep
    # hyperas , tune
    # https://docs.ray.io/en/master/tune/
    # https://machinelearningmastery.com/hyperopt-for-automated-machine-learning-with-scikit-learn/
    # https://pypi.org/project/hyperas/
    # https://docs.ray.io/en/master/tune/
    # https://keras-team.github.io/keras-tuner/
    # https://github.com/hyperopt/hyperopt
    # https://github.com/optuna/optuna
    # https://scikit-optimize.github.io/stable/
    # https://www.automl.org/book/





    ####################################################################################################################
    # INTERPRETABILITY
    ####################################################################################################################
    #todo
    # important book
    # https://christophm.github.io/interpretable-ml-book/cnn-features.html
    # Alguns packages para interpretabilidade:
    # https://github.com/slundberg/shap
    # https://github.com/kundajelab/deeplift
    # https://github.com/marcotcr/lime
    # https://github.com/sicara/tf-explain
    # https://captum.ai/

    # tensorboard visualization
    # the embeddings using in the google paper

    # shap values
    # https://medium.com/@gabrieltseng/interpreting-complex-models-with-shap-values-1c187db6ec83
    # https://towardsdatascience.com/explain-any-models-with-the-shap-values-use-the-kernelexplainer-79de9464897a
    # https://www.kaggle.com/dansbecker/advanced-uses-of-shap-values
    # @timer
    # def shap(self, model, data):
    #     # Create object that can calculate shap values
    #     explainer = shap.DeepExplainer(model,data)
    #
    #     # Calculate Shap values
    #     shap_values = explainer.shap_values(data)
    #     print(shap_values)
    #     shap.force_plot(explainer.expected_value[1], shap_values[1], data)
    #     shap.summary_plot(shap_values[1], data) # When plotting, we call shap_values[1]. For classification problems, there is a separate array of SHAP values for each possible outcome. In this case, we index in to get the SHAP values for the prediction of "True".
    #
    #     shap.dependence_plot('Ball Possession %', shap_values[1], X, interaction_index="Goal Scored")


    # CNN
    # https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2
    # Feature maps
    # Convnet filters
    # Class output
    # https://raghakot.github.io/keras-vis/vis.visualization/
    # https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
    # https://christophm.github.io/interpretable-ml-book/cnn-features.html

####################################################################################################################
# AUTO ML
####################################################################################################################
    #todo
    # https://www.automl.org/book/





# todo por aqui os pesos do melhor modelo os parametros

# todo por a guardar so o melhor modelo e n de epoch em epoch https://www.tensorflow.org/tutorials/keras/save_and_load


# In the neural network terminology:
# one epoch = one forward pass and one backward pass of all the training examples
# batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
# number of iterations = number of passes, each pass using [batch size] number of examples. To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).
# Example: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.
#

# https://colah.github.io/posts/2015-08-Understanding-LSTMs/
# https://developers.google.com/machine-learning/crash-course/ml-intro
# https://www.deeplearning.ai/deep-learning-specialization/
# https://www.tensorflow.org/tutorials/keras/save_and_load







#
# if __name__ == '__main__':
#     from src.mlmodels.run_deep_model import prepare_hotencoded
#
#     ecpred_uniref_90 = '/home/amsequeira/deepbio/datasets/ecpred/ecpred_uniprot_uniref_90.csv'
#     alphabet = "ARNDCEQGHILKMFPSTWYV"
#
#     # # prepare data
#     # x_train, x_test, x_dval, y_train, y_test, y_dval, input_dim, fps_x, fps_y = \
#     #     prepare_hotencoded(ecpred_uniref_90, sequence=True, alphabet=alphabet, seq_len=100,
#     #                        parameter='Cross-reference (SUPFAM)', column_label_name='EC number',
#     #                        multiclass=True, multilabel=False,
#     #                        count_negative_as_class=False,
#     #                        model='dnn')
#
#     # # open class
#     # train = ModelTrainEval(x_train, y_train, x_test, y_test, x_dval, y_dval, validation_split=0.25,
#     #                        epochs=500, callbacks=None, reduce_lr=True, early_stopping=True, checkpoint=True,
#     #                        early_stopping_patience=int(30),
#     #                        reduce_lr_patience=50, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
#     #                        path='/home/amsequeira/deepbio/savedModels/weights/',
#     #                        save=False,
#     #                        plot_model_file='/home/amsequeira/deepbio/savedModels/plot_modelmodel_plot.png',
#     #                        verbose=2, batch_size=256)
#
#     # # train dnn
#     # dnn = train.run_dnn_simple(input_dim,
#     #                            optimizer='Adam',
#     #                            hidden_layers=(128, 64),
#     #                            dropout_rate=0.3,
#     #                            batchnormalization=True,
#     #                            l1=1e-5, l2=1e-4,
#     #                            final_dropout_value=0.0,
#     #                            initial_dropout_value=0.0)
#
#     # lstm = train.run_lstm_simple(input_dim,
#     #                              optimizer='Adam',
#     #                              bilstm=True,
#     #                              lstm_layers=(128, 64),
#     #                              dense_layers=(64,),
#     #                              activation='tanh',
#     #                              recurrent_activation='sigmoid',
#     #                              dense_activation="relu",
#     #                              l1=1e-5, l2=1e-4,
#     #                              initial_dropout_value=0.0,
#     #                              dropout_rate=0.3, recurrent_dropout_rate=0.3,
#     #                              final_dropout_value=0.0)
#
#     # train.model_simple_evaluate()
#     # scores = train.model_complete_evaluate(plots=True)
#     # print(pd.Series(scores).to_frame('scores'))
#     # train.print_evaluation(x_test, y_test)
#     #
#     # predict = train.model_predict(x_test)
#     # print(predict)
#
#     from src.mlmodels.run_deep_model import get_hot_encoded_sequence, get_final_fps_x_y, \
#         get_parameter_hot_encoded
#     from src.mlmodels.run_deep_model import list_sequences_padded_fps_y
#     from src.mlmodels.utils_run_models import divide_dataset, binarize_labels
#     from src.mlmodels.run_deep_model import normalization, divide_train_test
#
#
#     def prepare_hotencoded_fps(file, sequence=True, alphabet='ARNDCEQGHILKMFPSTWYV', seq_len=1000,
#                                parameter='Cross-reference (Pfam)', column_label_name='ec_number',
#                                multiclass=False, multilabel=False,
#                                count_negative_as_class=True):
#         if sequence:
#             fps_x, fps_y_bin, fps_y, ec_number = get_hot_encoded_sequence(file, alphabet, seq_len)
#         else:
#             pfam_binary, fps_y_bin, fps_y, ec_number = get_parameter_hot_encoded(file, parameter, column_label_name)
#             fps_x = pfam_binary
#
#         # get fps_x and fps_y accordingly of being multiclass or multilabel and number of classes
#         fps_x, fps_y = get_final_fps_x_y(fps_x, fps_y_bin, ec_number, multiclass, multilabel, count_negative_as_class)
#
#         input_dim = fps_x.shape[1]
#         return fps_x, fps_y, input_dim
#
#
#     # prepare data for hot encoded (sequence or pfam domains)
#     fps_x, fps_y, input_dim = prepare_hotencoded_fps(ecpred_uniref_90, sequence=True, alphabet='ARNDCEQGHILKMFPSTWYV',
#                                                      seq_len=100,
#                                                      parameter='Cross-reference (Pfam)', column_label_name='ec_number',
#                                                      multiclass=False, multilabel=False,
#                                                      count_negative_as_class=True)
#
#     # open class
#     train = ModelTrainEval(fps_x, fps_y, validation_split=0.25,
#                            epochs=500, callbacks=None, reduce_lr=True, early_stopping=True, checkpoint=True,
#                            early_stopping_patience=int(30),
#                            reduce_lr_patience=50, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
#                            path='/home/amsequeira/deepbio/savedModels/weights/',
#                            save=False,
#                            plot_model_file='/home/amsequeira/deepbio/savedModels/plot_modelmodel_plot.png',
#                            verbose=2, batch_size=256)
#
#     dnn_cv = train.run_dnn_simple_cv(input_dim,
#                                      optimizer='Adam',
#                                      hidden_layers=(128, 64),
#                                      dropout_rate=0.3,
#                                      batchnormalization=True,
#                                      l1=1e-5, l2=1e-4,
#                                      final_dropout_value=0.3,
#                                      initial_dropout_value=0.2,
#                                      cv=3)
