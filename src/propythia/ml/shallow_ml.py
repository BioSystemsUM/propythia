# -*- coding: utf-8 -*-
"""
##############################################################################

File containing a class intend to facilitate machine learning with peptides or other.
The functions are based on the package scikit learn.
Model available: 'svm', 'linear_svm', 'knn', 'sgd', 'lr','rf', 'gnb', 'nn','gboosting'

Authors: Ana Marta Sequeira

Date:06/2019 altered 01/2021

Email:

##############################################################################
"""

import pandas as pd
import time
import seaborn as sns
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from numpy import interp
from itertools import cycle
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegressionCV, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVR, LinearSVR

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import TweedieRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import validation_curve
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, \
    mean_squared_log_error, median_absolute_error, r2_score, mean_poisson_deviance, mean_gamma_deviance
from .utils import timer
from .parameters_shallow import param_shallow
from .param_optimizer import ParamOptimizer

warnings.filterwarnings("ignore")
# Seed for randomization. Set to some definite integer for debugging and set to None for production
seed = None


class ShallowML:
    """
    The MachineLearning class aims to process different models of machine learning with peptides.
    Model available: 'svm', 'linear_svm', 'knn', 'sgd', 'lr','rf', 'gnb', 'nn','gboosting'
    Based on scikit learn.
    """

    # todo acrescentei problem_type
    def __init__(self, x_train, x_test, y_train, y_test, report_name=None, columns_names=None,
                 problem_type='src', classifier = None):
        """
        init function. When the class is called a dataset containing the features values and a target column must
        be provided.
        :param x_train: dataset with features or encodings for training
        :param x_test: dataset with features or encodings for evaluation
        :param y_train: class labels for training
        :param y_test: class labels for testing
        :param report_name: str. If not none it will generate a report txt with the name given) with results by functions
        called within class. Default None
        :param columns_names: Names of columns. important if features importance want to be analysed. None by default.
        """
        # self.X_data = pd.DataFrame(X_data) # take X_data off
        # self.Y_data = Y_data
        self.x_train = x_train
        self.x_test = x_test
        # self.x_dval = X_dval
        self.y_train = y_train
        self.y_test = y_test
        # self.y_dval = y_dval.ravel()
        # y = column_or_1d(y, warn=True)

        # ver se vale a pena aceitar de fora
        self.model_name = None
        self.feat_table = None
        self.classifier = classifier

        if columns_names is not None:
            self.columns = columns_names
        else:
            try:
                self.columns = self.x_train.columns
            except Exception as e:
                print(e)
                print('no features names listed')

        self.problem_type = problem_type
        if self.problem_type == 'src':
            self.main_score = make_scorer(matthews_corrcoef)
            self.final_units = len(np.unique(y_train))
        elif self.problem_type == 'regression':
            self.main_score = 'neg_root_mean_squared_error'
        self.report_name = report_name
        if self.report_name:
            self._report(str(self.report_name))

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
    def get_model(self):
        return self.classifier

    def train_best_model(self, model, scaler=None,
                         score=None,
                         cv=10, optType='gridSearch', param_grid=None,
                         n_jobs=5,
                         random_state=1, n_iter=15, refit=True, **params):
        # todo put the name here oh the siglas. lr - logistic regression ....

        """
        This function performs a parameter grid search or randomizedsearch on a selected classifier model and training data set.
        It returns a scikit-learn pipeline that performs standard scaling (if not None) and contains the best model found by the
        grid search according to the Matthews correlation coefficient or other given metric.
        :param model_name: {str} model to train. Choose between 'svm', 'linear_svm', 'knn', 'sgd', 'lr','rf', 'gnb', 'nn','gboosting'
        :param model: scikit learn model
        :param scaler: {scaler} scaler to use in the pipe to scale data prior to training (integrated  in pipeline)
         Choose from
            ``sklearn.preprocessing``, e.g. 'StandardScaler()', 'MinMaxScaler()', 'Normalizer()' or None.None by default.
        :param score: {metrics instance} scoring function built from make_scorer() or a predefined value in string form
            (choose from the scikit-learn`scoring-parameters <http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_).
        :param optType: type of hyperparameter optimization to do . 'gridSearch' or 'randomizedSearch'
        :param param_grid: {dict} parameter grid for the grid or randomized search
        (see`sklearn.grid_search <http://scikit-learn.org/stable/modules/model_evaluation.html>`_).
        :param cv: {int} number of folds for cross-validation.
        :param n_iter: number of iterations when using randomizedSearch optimizaton. default 15.
        :param n_jobs: {int} number of parallel jobs to use for calculation. if ``-1``, all available cores are used.
        10 by default.
        :param refit: wether to refit the model accordingly to the best model . Default True.
        :param random_state: random_state 1 by default.
        :param **params: params to integrate in scikit models
        :return: best classifier fitted to training data.
        """
        start = time.perf_counter()
        print("performing {}...".format(optType))

        saved_args = locals()
        if model is str:
            model = model.lower()
            self.model_name = model_name

        # todo change the SVM to SVC   linear_svc
        if model == 'svc':
            pipe_model = Pipeline([('scl', scaler),
                                   ('clf', SVC(random_state=random_state, **params))])

        elif model == 'linear_svc':  # scale better to large number of samples
            pipe_model = Pipeline([('scl', scaler),
                                   ('clf', LinearSVC(random_state=random_state, **params))])

        elif model == 'rf':
            pipe_model = Pipeline([('scl', scaler),
                                   ('clf', RandomForestClassifier(random_state=random_state, **params))])

        elif model == 'gboosting':
            pipe_model = Pipeline([('scl', scaler),
                                   ('clf', GradientBoostingClassifier(random_state=random_state, **params))])

        elif model == 'knn':
            pipe_model = Pipeline([('scl', scaler),
                                   ('clf', KNeighborsClassifier(**params))])

        elif model == 'sgd':
            pipe_model = Pipeline([('scl', scaler),
                                   ('clf', SGDClassifier(random_state=random_state, **params))])

        elif model == 'lr':
            pipe_model = Pipeline([('scl', scaler),
                                   ('clf', LogisticRegression(random_state=random_state, **params))])

        elif model == 'gnb':
            pipe_model = Pipeline([('scl', scaler),
                                   ('clf', GaussianNB(**params))])

        elif model == 'nn':
            pipe_model = Pipeline([('scl', scaler),
                                   ('clf', MLPClassifier(**params))])

        # regression models
        elif model == 'svr':
            pipe_model = Pipeline([('scl', scaler),
                                   ('clf', SVR(**params))])

        elif model == 'linear_svr':
            pipe_model = Pipeline([('scl', scaler),
                                   ('clf', LinearSVR(random_state=random_state, **params))])

        elif model == 'rfr':
            pipe_model = Pipeline([('scl', scaler),
                                   ('clf', RandomForestRegressor(random_state=random_state, **params))])

        elif model == 'gboostingr':
            pipe_model = Pipeline([('scl', scaler),
                                   ('clf', GradientBoostingRegressor(random_state=random_state, **params))])

        elif model == 'histgboostingr':
            pipe_model = Pipeline([('scl', scaler),
                                   ('clf', HistGradientBoostingRegressor(random_state=random_state, **params))])

        elif model == 'adaboostr':
            pipe_model = Pipeline([('scl', scaler),
                                   ('clf', AdaBoostRegressor(random_state=random_state, **params))])

        elif model == 'tweedier':
            pipe_model = Pipeline([('scl', scaler),
                                   ('clf', TweedieRegressor(**params))])

        elif model =='knr':
            pipe_model = Pipeline([('scl', scaler),
                                   ('clf', KNeighborsRegressor(**params))])

        elif model == 'gpr':
            pipe_model = Pipeline([('scl', scaler),
                                   ('clf', GaussianProcessRegressor(**params))])

        elif model == 'sgdr':
            pipe_model = Pipeline([('scl', scaler),
                                   ('clf', SGDRegressor(random_state=random_state,**params))])
        # todo add light GBM ? alterar nomes do Gboosting para GBc e GBr ? add XGboost? adicionar para classificacao tb

        # todo alter
        elif model is str:
            # keras classifier
            print("Model not supported, please choose between "
                  "src models: 'svc', 'knn', 'sgd', 'rf', 'gnb', 'nn', 'gboosting'"
                  "or regression models:  ")
            return

        else:
            # todo confirm if is working
            pipe_model = Pipeline([('scl', scaler),
                                   ('clf', model)])

        # define scores
        # todo change this. TEST it. make a self.score ?
        # if model in ['svc', 'linear_svc', 'rf', 'gboosting', 'knn', 'sgd', 'lr', 'gnb', 'nn'] and score is None:
        #     score = make_scorer(matthews_corrcoef)
        # elif model in ['svr', 'linear_svr']:
        #     score = r2_score
        # if score is None:  # did not change with models (because is a different one) and did not specify score
        #     print("Please specify a metric score (available on scikit learn)"
        #           "for example accuraccy or make_scorer(matthews_corrcoef) for src and r2_score for "
        #           "regression problems. Check https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics"
        #           "for more info")
        #     return

        if score is None:
            score = self.main_score  # accordingly with src or regression
        # or ifself.classification_type == ... score ==
        # if self.src is None: parte dos model in

        # retrieve default grids
        if param_grid is None:
            param = param_shallow()
            if optType == 'gridSearch':
                param_grid = param[model]['param_grid']
            else:
                param_grid = param[model]['distribution']

        po = ParamOptimizer(optType=optType, estimator=pipe_model, paramDic=param_grid,
                            dataX=self.x_train, datay=self.y_train,
                            cv=cv, n_iter_search=n_iter, n_jobs=n_jobs, scoring=score, model_name=self.model_name,
                            refit=refit)
        gs = po.get_opt_params()

        # # summaries
        list_to_write_top_3_models = po.report_top_models(gs)

        s1, s2, s3, df = po.report_all_models(gs)  # metrics for the best model. dataframe with all
        # except:
        #     print('not print all models')
        # Set the best parameters to the best estimator
        # todo take out of this function and put otside. a function to calculate a single model. put self.best calssfier and a function to get it
        best_classifier = gs.best_estimator_
        best_classifier_fit = best_classifier.fit(self.x_train, self.y_train)
        self.classifier = best_classifier_fit
        final = time.perf_counter()
        run_time = final - start

        # write report
        if self.report_name is not None:
            self._report(['===TRAIN MODELS===\n', self.train_best_model.__name__, saved_args])
            self._report(list_to_write_top_3_models)
            self._report([s1, s2, s3, f"Finished {self.train_best_model.__name__} in {run_time:.4f} secs\n\n"])
            self._report(df, dataframe=True, float_format='%.3f')

        return best_classifier_fit

    def cross_val_score_model(self, model,
                              score=None,
                              cv=10,
                              n_jobs=10,
                              random_state=1, **params):

        """
        This function performs cross validations core on a selected classifier model and training data set.
        It returns the scores across the different folds, means and standard deviations of these scores.
        :param model_name: {str} model to train. Choose between 'svm', 'knn', 'sgd', 'lr','rf', 'gnb', 'nn','gboosting'
        :param model: scikit learn machine learning model object
        :param score: {metrics instance} scoring function built from make_scorer() or a predefined value in string form
            (choose from the scikit-learn`scoring-parameters <http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_).
        :param cv: {int} number of folds for cross-validation score.
        :param n_jobs: {int} number of parallel jobs to use for calculation. if ``-1``, all available cores are used.
        10 by default.
        :param random_state: random_state 1 by default.
        :param **params: params to integrate in scikit models
        :return: dataframe with scores per fold, mean and std of these scores.
        """
        start = time.perf_counter()
        print("performing cross val score with {} folds".format(cv))

        saved_args = locals()

        if score is None:
            score = self.main_score
        if model == 'svc':
            clf = SVC(random_state=random_state, **params)

        elif model == 'linear_svc':  # scale better to large number of samples
            clf = LinearSVC(random_state=random_state, **params)

        elif model == 'rf':
            clf = RandomForestClassifier(random_state=random_state, **params)

        elif model == 'gboosting':
            clf = GradientBoostingClassifier(random_state=random_state, **params)

        elif model == 'knn':
            clf = KNeighborsClassifier(**params)

        elif model == 'sgd':
            clf = SGDClassifier(random_state=random_state, **params)

        elif model == 'lr':
            clf = LogisticRegression(random_state=random_state, **params)

        elif model == 'gnb':
            clf = GaussianNB(**params)

        elif model == 'nn':
            clf = MLPClassifier(**params)

        # todo add regression models
        elif model == 'gpr':
           clf = GaussianProcessRegressor(**params)

        elif model is str:
            print("Model not supported, please choose between 'svc', 'knn', 'sgd', 'rf', 'gnb', 'nn', 'gboosting', 'gpr' ")
            return
        else:
            clf = model
        # retrieve default grids
        scores = cross_validate(clf, self.x_train, self.y_train, cv=cv, scoring=score)
        # print(scores)
        scores = pd.DataFrame(scores)
        scores.loc['mean'] = scores.mean()
        scores.loc['std'] = scores.std()

        # try:
        #     scores['test_mean_score'] = np.mean(scores['test_score'])
        #     scores['test_std_score'] = np.std(scores['test_score'])
        #
        # except:
        #     for scorer in score:
        #         scores['test_mean_%s' % (scorer)] = np.mean(scores['test_%s' % (scorer)])
        #         scores['test_std_%s' % (scorer)] = np.std(scores['test_%s' % (scorer)])

        final = time.perf_counter()
        run_time = final - start

        # write report
        if self.report_name is not None:
            self._report(['===TRAIN MODELS===\n', self.cross_val_score_model.__name__, saved_args])
            self._report(scores)
            self._report(f"Finished {self.cross_val_score_model.__name__} in {run_time:.4f} secs\n\n")

        return scores

    ####################################################################################################################
    # EVALUATE
    ####################################################################################################################
    # todo join this to use with deep
    def conf_matrix_seaborn_table(self, conf_matrix=None, classifier=None, path_save='', show=True,
                                  square=True, annot=True, fmt='d', cbar=False, **params):
        plt.clf()
        if conf_matrix is None:
            y_pred = classifier.predict(self.x_test)
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

    def score_testset_classification(self, classifier=None):
        """
        Returns the tests set scores for the specified scoring metrics in a ``pandas.DataFrame``. The calculated metrics
        are Matthews correlation coefficient, accuracy, precision, recall, f1 and area under the Receiver-Operator Curve
        (roc_auc). See `sklearn.metrics <http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics>`_
        for more information.
        :param classifier: {classifier instance} pre-trained classifier used for predictions. If None, will use the
        one trained in the class.
        :return: ``pandas.DataFrame`` containing the cross validation scores for the specified metrics.
        """
        if classifier is None:  # priority for the classifier in function
            classifier = self.classifier
        saved_args = locals()
        cm2 = []
        scores = {}
        y_pred = classifier.predict(self.x_test)
        try:
            y_prob = classifier.predict_proba(self.x_test)
        except:
            y_prob = None
        scores['Accuracy'] = accuracy_score(self.y_test, y_pred)
        scores['MCC'] = matthews_corrcoef(self.y_test, y_pred)
        if y_prob is not None:
            scores['log_loss'] = log_loss(self.y_test, y_prob)

        if self.final_units > 2:
            # multiclass
            scores['f1 score weighted'] = f1_score(self.y_test, y_pred, average='weighted')
            scores['f1 score macro'] = f1_score(self.y_test, y_pred, average='macro')
            scores['f1 score micro'] = f1_score(self.y_test, y_pred, average='micro')
            if y_prob is not None:
                scores['roc_auc ovr'] = roc_auc_score(self.y_test, y_prob, average='weighted', multi_class='ovr')
                y_test = self.y_test
                # y_test = y_test.reshape(y_test.shape[0])  # roc auc ovo was giving error
                scores['roc_auc ovo'] = roc_auc_score(y_test, y_prob, average='weighted', multi_class='ovo')
            scores['precision'] = precision_score(self.y_test, y_pred, average='weighted')
            scores['recall'] = recall_score(self.y_test, y_pred, average='weighted')
            cm2 = multilabel_confusion_matrix(self.y_test, y_pred)

        else:
            # binary
            scores['f1 score'] = f1_score(self.y_test, y_pred)
            scores['roc_auc'] = roc_auc_score(self.y_test, y_pred)
            precision, recall, thresholds = precision_recall_curve(self.y_test, y_pred)
            scores['Precision'] = precision
            scores['Recall'] = recall
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
            scores['fdr'] = float(fp) / (tp + fp)
            scores['sn'] = float(tp) / (tp + fn)
            scores['sp'] = float(tn) / (tn + fp)

        report = classification_report(self.y_test, y_pred, output_dict=False)
        cm = confusion_matrix(self.y_test, y_pred)
        cm2 = None

        if self.report_name is not None:
            self._report(['===SCORING TEST SET ===\n', self.score_testset.__name__, saved_args, 'report\n', report,
                          '\nconfusion_matrix\n', cm, '\nmultilabel confusion matrix\n', cm2, '\nscores report\n'])
            scores_df = pd.DataFrame(scores.keys(), columns=['metrics'])
            scores_df['scores'] = scores.values()
            self._report(scores_df, dataframe=True, float_format='%.4f', index=False)

            self.conf_matrix_seaborn_table(conf_matrix=cm,
                                           path_save=str(self.report_name[-4:] + 'confusion_matrix.png'),
                                           show=False)

        return scores, report, cm, cm2
    # todo add this fucntion?
    # see if it is worth it. put it with not priniting . giving something closer to the src report scikit learn
    def regression_report(self,y_true, y_pred):

        error = y_true - y_pred
        percentil = [5, 25, 50, 75, 95]
        percentil_value = np.percentile(error, percentil)

        metrics = [
            ('mean absolute error', mean_absolute_error(y_true, y_pred)),
            ('median absolute error', median_absolute_error(y_true, y_pred)),
            ('mean squared error', mean_squared_error(y_true, y_pred)),
            ('max error', max_error(y_true, y_pred)),
            ('r2 score', r2_score(y_true, y_pred)),
            ('explained variance score', explained_variance_score(y_true, y_pred))
        ]

        print('Metrics for regression:')
        for metric_name, metric_value in metrics:
            print(f'{metric_name:>25s}: {metric_value: >20.3f}')

        print('\nPercentiles:')
        for p, pv in zip(percentil, percentil_value):
            print(f'{p: 25d}: {pv:>20.3f}')
        return None

    # todo add this function !!!!
    # https://medium.com/analytics-vidhya/mae-mse-rmse-coefficient-of-determination-adjusted-r-squared-which-metric-is-better-cd0326a5697e
    def score_testset_regression(self, classifier=None):
        """
        Returns the tests set scores for the specified scoring metrics in a ``pandas.DataFrame``. The calculated metrics
        are Matthews correlation coefficient, accuracy, precision, recall, f1 and area under the Receiver-Operator Curve
        (roc_auc). See `sklearn.metrics <http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics>`_
        for more information.
        :param classifier: {classifier instance} pre-trained classifier used for predictions. If None, will use the
        one trained in the class.
        :return: ``pandas.DataFrame`` containing the cross validation scores for the specified metrics.
        """
        if classifier is None:  # priority for the classifier in function
            classifier = self.classifier
        saved_args = locals()
        cm2 = []
        scores = {}
        y_pred = classifier.predict(self.x_test)
        # todo put abs in negative values? they give negative values so the model try to get the lower score rather than the higher.
        # https://stackoverflow.com/questions/21443865/scikit-learn-cross-validation-negative-values-with-mean-squared-error
        scores['neg_root_mean_squared_error - RMSE'] = abs(mean_squared_error(self.y_test, y_pred, squared=False))
        scores['neg_mean_squared_error - MSE'] = abs(mean_squared_error(self.y_test, y_pred, squared=True))
        scores['neg_mean_absolute_error - MAE'] = abs(mean_absolute_error(self.y_test, y_pred))
        scores['r2'] = r2_score(self.y_test, y_pred) # It represents the proportion of variance (of y) that has been explained
        # by the independent variables in the model. It provides an indication of goodness of fit and therefore a measure
        # of how well unseen samples are likely to be predicted by the model, through the proportion of explained variance.
        # As such variance is dataset dependent, R² may not be meaningfully comparable across different datasets.
        scores['explained_variance'] = explained_variance_score(self.y_test, y_pred) # The best possible score is 1.0, lower values are worse
        scores['max_error'] = max_error(self.y_test, y_pred) # calculates the maximum residual error. aptures the worst case error between the predicted value and the true value

        try:
            scores['neg_mean_squared_log_error - MSLE'] = abs(mean_squared_log_error(self.y_test, y_pred)) # expected value of the squared logarithmic (quadratic) error or loss.

        except Exception as e:
            print('neg_mean_squared_log_error could not be calculated')
            print(str(e))

        try:
            scores['neg_median_absolute_error'] = abs(median_absolute_error(self.y_test, y_pred)) # robust to outliers. The loss is calculated by taking the median of all absolute differences between the target and the prediction. Do not support multioutput
        except Exception as e:
            print('neg_median_absolute_error could not be calculated')
            print(str(e))

        # todo only for positive values? some of them ? check
        try:
            scores['neg_mean_poisson_deviance'] = abs(mean_poisson_deviance(self.y_test, y_pred))
        except Exception as e:
            print('neg_mean_poisson_deviance could not be calculated')
            print(str(e))
        try:
            scores['neg_mean_gamma_deviance'] = abs(mean_gamma_deviance(self.y_test, y_pred))
        except Exception as e:
            print('neg_mean_gamma_deviance could not be calculated')
            print(str(e))
        try:
            scores['mean_absolute_percentage_error'] = abs(mean_absolute_percentage_error(self.y_test, y_pred)) # sensitive to relative errors. supports multioutput
        except Exception as e:
            print('mean_absolute_percentage_error could not be calculated')
            print(str(e))

        report = self.regression_report(self.y_test, y_pred)
        cm = None
        cm2 = None

        if self.report_name is not None:
            self._report(['===SCORING TEST SET ===\n', self.score_testset.__name__, saved_args, 'report\n', report,
                          '\nscores report\n'])
            scores_df = pd.DataFrame(scores.keys(), columns=['metrics'])
            scores_df['scores'] = scores.values()
            self._report(scores_df, dataframe=True, float_format='%.4f', index=False)

        return scores, report

    # todo add this. Put text explaining return all the same?

    def score_testset(self, classifier=None):
        if self.problem_type == 'src':
            scores, report, cm, cm2 = self.score_testset_classification(classifier=classifier)
            return scores, report, cm, cm2
        elif self.problem_type == 'regression':
            scores, report = self.score_testset_regression(classifier=classifier)
            return scores, report

    @timer
    def plot_roc_curve(self, classifier=None, ylim=(0.0, 1.00), xlim=(0.0, 1.0),
                       title='Receiver operating characteristic (ROC) curve',
                       path_save='plot_roc_curve', show=False):
        """
        Function to plot a ROC curve
        On the y axis, true positive rate and false positive rate on the X axis.
        The top left corner of the plot is the 'ideal' point - a false positive rate of zero, and a true positive rate
        of one, meaning a larger area under the curve (AUC) is usually better.
        :param classifier: {classifier instance} pre-trained classifier used for predictions.
        :param ylim: y-axis limits
        :param xlim: x- axis limits
        :param title: title of plot. 'Receiver operating characteristic (ROC) curve' by default.
        :param path_save: path to save the plot. If None , lot is not saved. 'plot_roc_curve' by default.
        :param show: Whether to display the graphic. False by default.
        :return:
        Needs classifier with probability
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        """
        lw = 2
        if classifier is None:
            classifier = self.classifier
        # binary
        if self.final_units <= 2:
            y_score = classifier.predict(self.x_test)
            # y_score = classifier.predict_proba(X_test)[:,1]
            fpr, tpr, thresholds = roc_curve(self.y_test, y_score)
            roc_auc = auc(fpr, tpr)
            print(roc_auc)
            plt.show()
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right", )


        # multiclass/ multilabel
        elif self.final_units > 2:

            # Binarize the output
            classe = np.unique(self.y_train)
            y_train = label_binarize(self.y_train, classes=classe)
            n_classes = y_train.shape[1]
            y_test = label_binarize(self.y_test, classes=classe)

            # Learn to predict each class against the other
            estimator = OneVsRestClassifier(classifier)
            y_score = estimator.fit(self.x_train, y_train).predict_proba(self.x_test)

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            print(all_fpr)
            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(fpr["micro"], tpr["micro"],
                     label='micro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=4)

            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=4)

            # Get Unique ec
            color_labels = np.unique(self.y_train)

            # List of colors in the color palettes
            rgb_values = sns.color_palette("nipy_spectral", len(color_labels))  # 'set2'
            # Map ec to the colors
            for i, color in zip(range(len(color_labels)), rgb_values):
                plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                               ''.format(i, roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right", fontsize='xx-small')
        if path_save is not None:
            plt.savefig(fname=path_save)
        if show is True:
            plt.show()
        plt.clf()

    ####################################################################################################################
    # PLOTS : Model curves
    ####################################################################################################################

    @timer
    def plot_validation_curve(self, param_name, param_range,
                              classifier=None,
                              cv=5,
                              score=None, title="Validation Curve",
                              xlab="parameter range", ylab="MCC", n_jobs=1, show=False,
                              path_save='plot_validation_curve', **params):

        """
        This function plots a cross-validation curve for the specified classifier on all tested parameters given in the
        option 'param_range'.

        :param param_name: {string} parameter to assess in the validation curve plot. For example,
        For SVM,
            "clf__C" (C parameter), "clf__gamma" (gamma parameter).
        For Random Forest,
            "clf__n_estimators" (number of trees),"clf__max_depth" (max num of branches per tree,
            "clf__min_samples_split" (min number of samples required to split an internal tree node),
            "clf__min_samples_leaf" (min number of samples in newly created leaf).
        :param param_range: {list} parameter range for the validation curve.
        :param classifier: {classifier instance} classifier or validation curve (e.g. sklearn.svm.SVC).If None (default)
        uses the classifier inside the class.
        :param cv: {int} number of folds for cross-validation. 5 by default.
        :param score: {metrics instance} scoring function built from make_scorer() or a predefined value in string form
            `sklearn.model_evaluation.scoring-parameter
            <http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_.
        :param title: {str} graph title
        :param xlab: {str} x axis label.
        :param ylab: {str} y axis label.
        :param n_jobs: {int} number of parallel jobs to use for calculation. if ``-1``, all available cores are used.
        1 by default.
        :param path_save: path to save the plot. If None , lot is not saved. 'plot_validation_curve' by default.
        :param show: Whether to display the graphic. False by default.
        **params: other parameters to integrate in the function of validation curve calculus.
        :return: plot of the validation curve.
        """
        if classifier is None:
            classifier = self.classifier
        if score is None:
            score = self.main_score

        train_scores, test_scores = validation_curve(classifier, self.x_train, self.y_train, param_name, param_range,
                                                     cv=cv, scoring=score, n_jobs=n_jobs, **params)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # plotting
        plt.clf()
        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.ylim(0.0, 1.1)
        plt.semilogx(param_range, train_mean, label="Training score", color="b")
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color="b")
        plt.semilogx(param_range, test_mean, label="Cross-validation score", color="g")
        plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2, color="g")
        plt.legend(loc="best")

        if path_save is not None:
            plt.savefig(fname=path_save)
        if show is True:
            plt.show()
        plt.clf()

    @timer
    def plot_learning_curve(self, classifier=None, title='Learning curve', ylim=None,
                            cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5),
                            path_save='plot_learning_curve', show=False, scalability=True, performance=True):
        """
        Plot a learning curve to determine cross validated training and tests scores for different training set sizes.
        IT retrieves graphic representing learning curves, numbers of trainig examples, scores on training sets,
        and scores on tests set and scalability and performance plots if set to True.
        :param classifier: {classifier instance} classifier or validation curve (e.g. sklearn.svm.SVC).If None (default)
        uses the classifier inside the class.
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
            classifier = self.classifier

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(classifier, self.x_train, self.y_train, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes,
                           return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # learning curve
        plt.clf()
        plt.grid(b=True, which='both')
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
        plt.legend(loc="best")
        if path_save is not None:
            plt.savefig(fname=str(path_save + '_learning_curve.png'))
        if show is True:
            plt.show()
        plt.clf()

        if scalability:
            # Plot n_samples vs fit_times
            plt.grid(b=True, which='both')
            plt.plot(train_sizes, fit_times_mean, 'o-')
            plt.fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
            plt.xlabel("Training examples")
            plt.ylabel("fit_times")
            plt.title("Scalability of the model")
            if path_save is not None:
                plt.savefig(fname=str(path_save + '_scalability_model.png'))
            if show is True:
                plt.show()
            plt.clf()

        if performance:
            # Plot fit_time vs score
            plt.grid(b=True, which='both')
            plt.plot(fit_times_mean, test_scores_mean, 'o-')
            plt.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
            plt.xlabel("fit_times")
            plt.ylabel("Score")
            plt.title("Performance of the model")
            if path_save is not None:
                plt.savefig(fname=str(path_save + '_performance_model.png'))
            if show is True:
                plt.show()
            plt.clf()

        # check if it is better way to do this. option of the plots. arrange the model names. in ones it gets the model names and the others dont
        return

    ####################################################################################################################
    # FEATURE IMPORTANCE
    ####################################################################################################################

    # todo adicionar modelos
    def _get_feat_impo(self, classifier=None, column_to_sort='mean_coef'):
        if classifier is None:
            classifier = self.classifier

        if self.model_name in ('rf', 'gboosting', 'rfr', 'gboostingr'):
            feature_importance = pd.DataFrame(self.classifier.named_steps['clf'].feature_importances_,
                                              index=self.columns,
                                              columns=['importance']).sort_values('importance', ascending=False)
        elif self.model_name in ('svc', 'sgd', 'lr', 'linear_svc', 'tweedier','linear_svr', 'gpr', 'svr'):
            df = pd.DataFrame(self.classifier.named_steps['clf'].coef_.T,
                              index=self.columns)  # columns=['Coefficients'])

            # get a column that is the media of the modules of all classes
            df['mean_coef'] = abs(df).mean(axis=1)

            # sort the df by specific class, by avg of all classes
            feature_importance = df.iloc[(-df[column_to_sort].abs()).argsort()]
            # check if better this way or ordered positive negative

            # other hypothesis to get the avg of abs values per row
            # filter_col = [col for col in df if col.startswith('coef_abs')]
            # df['mean_coef_abs'] = df[filter_col].mean(axis=1)
            # get the module of each class coefficient
            # for column in df:
            #     column_name = 'coef_abs_' + str(column)
            #     df[column_name] = abs(df[column])

        else:
            e = "Model not supported, please choose between " \
                "'svc', 'sgd', 'linear_svc', 'lr', 'rf', 'rfr','gboosting'," \
                "'gboostingr', 'tweedier','linear_svr', 'gpr', 'svr' "
            print(e)
            return e
        return feature_importance

    def features_importances_df(self, classifier=None, model_name=None, top_features=20, column_to_sort='mean_coef'):
        """
        Function that given a classifier retrieves the features importances as a dataframe.
        :param classifier: {classifier instance} classifier or validation curve (e.g. sklearn.svm.SVC).If None (default)
        uses the classifier inside the class.
        :param model_name: model used in classifier. Choose between 'svm', 'sgd', 'lr', 'gboosting' or 'rf'.
        this only matters to determine _coef or _featureimportance
        :param top_features: number of features to display on the dataframe. 20 by default.
        :param column_to_sort: column to sort the dataframe. 'mean_coef' by default. This column represents the average
        scores in multiclass problems. coef for each class are also generated
        :return: table with features names and importance for the model
        """

        saved_args = locals()
        # if self.model_name not in ('rf', 'rfr', 'gboosting', 'svm', 'sgd', 'lr', 'linear_svm', 'tweedier_poisson'):
        #     print("Model not supported, please choose between 'svm', 'sgd', 'linear_svm', 'lr', 'rf', 'rfr', or 'gboosting'")
        #     return
        # do something for gboostingr? histgboostingr? that give something like features seen in
        if model_name is not None:
            self.model_name = model_name
        if classifier is not None:
            self.classifier = classifier

        feat_table = self._get_feat_impo(self.classifier, column_to_sort=column_to_sort)
        self.feat_table = feat_table

        if self.report_name is not None:
            self._report([self.features_importances_df.__name__, saved_args])
            self._report(feat_table[30:], dataframe=True, header=True, float_format='%.3f')
            with open(str(self.report_name[:-4] + 'fi.txt'), 'a+') as f:
                feat_table.to_csv(f, sep='\t', mode='a')
        return feat_table

    def features_importances_plot(self, classifier=None, top_features=20, model_name=None,
                                  column_to_plot=None,
                                  show=False, path_save='feat_impo.png',
                                  title=None,
                                  kind='barh', figsize=(9, 7), color='r', edgecolor='black', **params):
        """
        Function that given a classifier retrieves the features importances represented as barplot.
        :param classifier: {classifier instance} classifier or validation curve (e.g. sklearn.svm.SVC).If None (default)
        uses the classifier inside the class.
        :param model_name: model used in classifier. Choose between 'svm', 'sgd', 'lr', 'gboosting' or 'rf'
        :param top_features: number of features to display on plot. 20by default.
        :param column_to_plot: None by default. Can be given a specific class label. If None, will retrieve the
        average column of scores.  in case of models like svm, sgd, the coefs are retrieved from each class.
        if no class is given it gives a plot with average of absolute values and the top features considering that
        if a column is given, it re arranges and plot top features for that specific class.
        :param path_save: path to save the plot. If None , plot is not saved. 'feat_impo.png' by default.
        :param show: Whether to display the graphic. False by default.
        :param title: title of the plot. None by default. If  none title is 'Feature Importance Plot for {}' model name
        :param color: color of plot bard
        :param kind: bar parameter. horizontal bars per default 'barh'. other hypothesis include 'bar', 'box'
        :param figsize: size of the fig. (9,7) by default.
        :param edgecolor: 'black' by default
        :return:
        https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.plot.html
        """

        saved_args = locals()
        if model_name is None:
            model_name = self.model_name
        else:
            self.model_name = model_name
        if classifier is None:
            classifier = self.classifier
        if self.feat_table is None:
            self.feat_table = self._get_feat_impo(classifier, column_to_plot)

        if title is None:
            title = 'Feature Importance Plot for {}'.format(model_name)

        color = color
        linewidth = 0.5
        label = 'Features Coefficients'

        if self.model_name.lower() in ('rf', 'gboosting', 'rfr', 'gboostingr'):
            label = 'Features Importance'
            table = self.feat_table[:top_features]

        # por isto no init ? o in in?
        elif self.model_name.lower() in ('svc', 'sgd', 'lr', 'linear_svc', 'svr', 'linear_svr', 'sgdr', 'tweedier'):
            label = 'Features Coefficients'
            if column_to_plot is None:  # all classes joined
                table = self.feat_table[:top_features]

                color_labels = np.unique(self.y_train)
                # List of colors in the color palettes
                rgb_values = sns.color_palette("nipy_spectral", len(color_labels) + 1)  # 'set2'
                color = rgb_values
                linewidth = 0.0

            else:  # graphic per class
                table = self.feat_table[column_to_plot][:top_features]
        else: # todo fix if it is other model rather the ones in the package por um try except?
            print("Model not supported, please choose between"
                  "'svc', 'sgd', 'linear_svc', 'lr', 'rf', 'rfr','gboosting',"
                  "'gboostingr', 'tweedier','linear_svr', 'gpr', 'svr'")
            return
        plt.clf()
        plt.title(title)
        plt.axvline(x=0, color='.5')
        plt.subplots_adjust(left=.3)
        table.plot(kind=kind, figsize=figsize, color=color, edgecolor=edgecolor, linewidth=linewidth, **params)

        plt.xlabel(label)
        plt.ylabel('Features')
        if path_save is not None:
            plt.savefig(fname=path_save)
        if show:
            plt.show()
        plt.clf()
        return

    ####################################################################################################################
    # PREDICTIONS
    ####################################################################################################################

    @timer
    def predict(self, x, seqs=None, classifier=None, names=None, true_y=None):

        """This function can be used to predict novel data with a trained classifier model. The function returns a
        'pandas.DataFrame' with predictions using the specified estimator and tests data.

        :param classifier: {classifier instance} classifier or validation curve (e.g. sklearn.svm.SVC).If None (default)
        uses the classifier inside the class.
        :param x: {array} descriptor values of the peptides to be predicted.
        :param seqs: {list} sequences of the peptides in ``x``. None by default
        :param names: {list} (optional) names of the peptides in ``x``. None by default.
        :param y: {array} (optional) true (known) classes of the peptides. None by default.
        :return: ``pandas.DataFrame`` containing predictions for ``x``. ``P_class0`` and ``P_class1``
            are the predicted probability of the peptide belonging to class 0 and class 1, respectively.

        Based on a function from moodlamp
        Müller A. T. et al. (2017) modlAMP: Python for anitmicrobial peptides, Bioinformatics 33, (17), 2753-2755,
        DOI:10.1093/bioinformatics/btx285
       """
        saved_args = locals()
        if classifier is None:
            classifier = self.classifier

        predict = classifier.predict(x)

        predict_df = pd.DataFrame(columns=['names', 'sequence', 'class predicted', 'True classes'])
        predict_df['names'] = names
        predict_df['True classes'] = true_y
        predict_df['sequence'] = seqs
        predict_df['class predicted'] = predict
        # remove columns with Nans all (cols that were None in entry
        predict_df = predict_df.dropna(axis=1)
        predict_df = predict_df.round(4)

        try:
            preds = classifier.predict_proba(x)
            cols_prob = ['prob_class_{}'.format(x) for x in range(self.final_units)]
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

# todo PREDICT WINDOW needs to do also regression
# # alternatively the user can use the function GetSubSeq(self,protein_sequence,ToAA='S',window=3)
# # from the readsequence module
# # that Get all 2*window+1 sub-sequences whose center is ToAA (a specific aminoacid) in a protein
# # giving a ToAA and a window
#
# def add_features(self,list_of_sequences,list_functions):
#     """
#     Calculate all features available in package or lis of descriptors given
#     :param list_functions: list of features to be calculated with function adaptable from module Descriptors
#     :param list_of_sequences: list of sequences to calculate features
#     :return: dataframe with sequences and its features
#     """
#
#     features_list=[] #creating an empty list of dataset rows
#     for seq in list_of_sequences:
#         res={'sequence':seq}
#         sequence=ReadSequence() #creating sequence object
#         ps=sequence.read_protein_sequence(seq)
#         protein = Descriptor(ps) # creating object to calculate descriptors)
#         if len(list_functions)==0:
#             feature=protein.get_all(tricomp=True, bin_aa=False, bin_prop=False)
#         else:
#             feature=protein.adaptable(list_functions)
#         res.update(feature)
#         features_list.append(res)
#
#     df = pd.DataFrame(features_list)
#     #df.set_index(['sequence'],inplace=True)
#     return df
# #
# def predict_window(self,classifier, seq,x=None, window_size=20,gap=1,features=[], features_names=None,
#                    features_dataframe=None, names=None, y=None,filename=None):
#     """Scan a protein in a sliding window approach to predict novel peptides with a trained classifier model.
#     The function returns a
#     'pandas.DataFrame' with predictions using the specified estimator and tests data. If true class is provided,
#     it returns the scoring value for the tests data.
#     The user can provide a features_dataframe if does not want to use the descriptors supported. Otherwise, the user
#     can also provide a list with the numbers of the descriptors that want to be calculated (descriptors module adaptable())
#     if none is provided, the function will calculate all the descriptors available.
#
#     :param classifier: {classifier instance} classifier used for predictions.
#     :param x: {array} descriptor values of the peptides to be predicted.
#     :param seq:  sequence of the peptides in ``x``.
#     :param window_size: number of aminoacids to considerer in each seq to check. for default 20
#     :param gap: gap size of the search of windows in sequence. default 1
#     :param features: list containing the list features to be calculated under de adaptable function.
#             if list is empty will calculate all the descriptors available in the package
#     :param features_dataframe: dataframe with sequences and its features
#     :param features_names:names of features. If none will match the features with the ones with the model given
#     :param names: {list} (optional) names of the peptides in ``x``.
#     :param y: {array} (optional) true (known) classes of the peptides.
#     :param filename: {string} (optional) output filename to store the predictions to (``.csv`` format); if ``None``:
#         not saved.
#     :return: ``pandas.DataFrame`` containing predictions for subsequences generated, ``P_class0`` and ``P_class1``
#         are the predicted probability of the peptide belonging to class 0 and class 1, respectively. The previsions
#         are divided in probability classes <0.99, >0.95, >0.9, <0.8, >0.7, >0.6 and 0
#    """
#
#     # generate final list of sequences/split sequences
#     list_of_sequences,indices=sub_seq_sliding_window(seq,window_size,gap,index=True)
#
#
#     # calculate features for sequences
#
#     #if a dataframe with features calculated that will be considered
#     #if not features will be calculated.
#     # if features is none will calculate all the features according to the package and choose the features for the classifier
#     # if features is list with numbers will call the adaptable function
#
#     if features_dataframe != None: featuresDF=features_dataframe
#     else: featuresDF = self.add_features(list_of_sequences,features)
#
#     #select the features used for the model construction
#     if features_names==None:
#         features_to_select=self.X_data.columns
#     else:
#         features_to_select= features_names
#
#     x_predict_data=featuresDF[features_to_select]
#
#     # raw proability predictions of belonging or not
#     preds = classifier.predict_proba(x_predict_data)
#     #0 or 1 if belong or not
#     predict=classifier.predict(x_predict_data)
#
#     # dataframe with probabilities
#     df_pred= self.predict(classifier, x_predict_data, list_of_sequences, names=names, y=y, filename=filename)
#     df_pred['prevision']= predict
#     df_pred['pos_0']=[i[0] for i in indices]
#     df_pred['pos_-1']=[i[1] for i in indices]
#
#     # create a dataframe specifying really the sequences, indices, 1 and 0 and probability
#     # (can put scale >0.99, 095, 0.90, 0.8, 0.7, <0.7
#     column=['sequence', 'prevision','probability','scale_probability', 'pos_0', 'pos_-1']
#     sequence=''
#     pos_0=int
#     pos_1=int
#     rows = []
#
#     for index, row in df_pred.iterrows():
#         value=row['P_class1']
#         if value>=0.99:
#             pos_0=row['pos_0']
#             pos_1=row['pos_-1']
#             rows.append({'sequence': index, 'prevision': 1, 'probability': value, 'scale_probability':5,
#                          'pos_0':pos_0, 'pos_-1':pos_1 })
#
#         if value>=0.95 and value<0.99:
#             pos_0=row['pos_0']
#             pos_1=row['pos_-1']
#             rows.append({'sequence': index, 'prevision': 1, 'probability': value, 'scale_probability':4,
#                          'pos_0':pos_0, 'pos_-1':pos_1 })
#
#         if value>=0.90 and value<0.95:
#             pos_0=row['pos_0']
#             pos_1=row['pos_-1']
#             rows.append( {'sequence': index, 'prevision': 1, 'probability': value, 'scale_probability':3,
#                           'pos_0':pos_0, 'pos_-1':pos_1 })
#
#         if value>=0.80 and value<0.90:
#             pos_0=row['pos_0']
#             pos_1=row['pos_-1']
#             rows.append( {'sequence': index, 'prevision': 1, 'probability':value, 'scale_probability':2,
#                           'pos_0':pos_0, 'pos_-1':pos_1 })
#
#         if value>=0.70 and value<0.80:
#             pos_0=row['pos_0']
#             pos_1=row['pos_-1']
#             rows.append( {'sequence': index, 'prevision': 1, 'probability': value, 'scale_probability':1,
#                           'pos_0':pos_0, 'pos_-1':pos_1 })
#
#         if value>=0.60 and value<0.70:
#             pos_0=row['pos_0']
#             pos_1=row['pos_-1']
#             rows.append({'sequence': index, 'prevision': 1, 'probability': value, 'scale_probability':0,
#                          'pos_0':pos_0, 'pos_-1':pos_1 })
#
#         if value<0.6:
#             pos_0=row['pos_0']
#             pos_1=row['pos_-1']
#             rows.append({'sequence': index, 'prevision': 0, 'probability': value, 'scale_probability':0,
#                          'pos_0':pos_0, 'pos_-1':pos_1 })
#
#     # create a new dataframe where consecutive rows in the same probability scale are joined and positions updated
#     df = pd.DataFrame(rows) # dataframe with scale probabilities
#
#     df['key'] = (df['scale_probability'] != df['scale_probability'].shift(1)).astype(int).cumsum()
#     # add a sentinel column that tracks which group of consecutive data each row applies to
#
#     df_new=df.drop(columns=['sequence', 'prevision'])
#     x=df_new.values.tolist() #not in pandas dataframe
#     remove_list=[]
#
#     for row in range(len(x)-1):
#         pos_fin=x[row][3]
#         pos_ini=x[row][2]
#         prob=round(x[row][0],3)
#         x[row][0] = round(x[row][0], 4)
#         scale_prob=x[row][1]
#         key=x[row][4]
#
#         if x[row][4]==x[row+1][4]:
#             remove_list.append(row+1) #select rows to delete
#
#     # update the positions. is not made in the same loop because in more than two consecutive rows it will
#     # update a row that will be deleted
#     for index in reversed(remove_list):
#         x[index-1][0]=x[index][0]
#
#     for index in sorted(remove_list, reverse=True): # remove rows that were joined
#         del x[index]
#
#     # add sequence
#     for row in range(len(x)):
#         seqs=seq[int(x[row][2]):int(x[row][3])]
#         x[row].append(seqs)
#
#     final_df=pd.DataFrame(x, columns=['probability','scale_probability','pos_0','pos_-1','key','sequence'])
#     final_df=final_df.drop(columns=['key'])
#     final_df = final_df[['pos_0','pos_-1','probability','scale_probability','sequence']]
#     return final_df
