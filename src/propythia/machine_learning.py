# -*- coding: utf-8 -*-
"""
##############################################################################

File containing a class intend to facilitate machine learning with peptides.
The functions are based on the package scikit learn.
The organization of the code is based on the mddlamp package.

Authors: Ana Marta Sequeira

Date:06/2019

Email:

##############################################################################
"""
import pandas as pd
import numpy as np
import time
from propythia.sequence import ReadSequence
from propythia.descriptors import Descriptor
from propythia.adjuv_functions.sequence.get_sub_seq import sub_seq_sliding_window
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn import metrics as mets
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit, train_test_split

# import scikitplot as skplt
# from more_itertools import unique_everseen

import warnings
warnings.filterwarnings("ignore")


class MachineLearning:
    """
    The MachineLearning class aims to process different models of machine learning with peptides.
    Based on scikit learn.
    """
    def _load_data(self, sklearn_load,target,test_size,classes):
        """
        load the data. the inputs are inherited from the init function when the class is called.
        :return:
        """
        data = sklearn_load
        X_data = pd.DataFrame(data)
        self.X_data=X_data
        self.Y_data=target

        self.labels=np.ravel(label_binarize(self.Y_data, classes=classes)) # ham will be 0 and spam will be 1

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_data, self.labels, test_size=test_size, random_state=42)
        # self.y_train_encoded=np.ravel(label_binarize(self.y_train, classes=['fps','tmds'])) # ham will be 0 and spam will be 1
        # self.y_test_encoded=np.ravel(label_binarize(self.y_test, classes=['fps','tmds'])) # ham will be 0 and spam will be 1

    def __init__(self, sklearn_load,target,test_size=0.3,classes=['pos','neg']):
        """
        init function. When the class is called a dataset containing the features values and a target column must be provided.
        Test size is by default 0.3 but can be altered by user.
        :param sklearn_load: dataset X_data
        :param target: column with class labels
        :param test_size: size for division of the dataset in train and tests
        """
        self._load_data(sklearn_load,target,test_size,classes)

    def train_best_model(self,model,sample_weights=None, scaler=StandardScaler(),
                     score=make_scorer(matthews_corrcoef), param_grid=None, n_jobs=-1, cv=10):
        """
        This function performs a parameter grid search on a selected classifier model and peptide training data set.
        It returns a scikit-learn pipeline that performs standard scaling and contains the best model found by the
        grid search according to the Matthews correlation coefficient.
        :param model: {str} model to train. Choose between 'svm', 'knn', 'sgd', 'lr','rf', 'gnb', 'nn','gboosting'
        :param x_train: {array} descriptor values for training data.
        :param y_train: {array} class values for training data.
        :param sample_weights: {array} sample weights for training data.
        :param scaler: {scaler} scaler to use in the pipe to scale data prior to training. Choose from
            ``sklearn.preprocessing``, e.g. 'StandardScaler()', 'MinMaxScaler()', 'Normalizer()'.
        :param score: {metrics instance} scoring function built from make_scorer() or a predefined value in string form
            (choose from the scikit-learn`scoring-parameters <http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_).
        :param param_grid: {dict} parameter grid for the gridsearch (see`sklearn.grid_search <http://scikit-learn.org/stable/modules/model_evaluation.html>`_).
        :param n_jobs: {int} number of parallel jobs to use for calculation. if ``-1``, all available cores are used.
        :param cv: {int} number of folds for cross-validation.
        :return: best estimator pipeline.

        **Default parameter grids:**

        =================                   ==============================================================================
        Model                                Parameter grid
        =================                   ==============================================================================
        SVM                                     param_grid = [{'clf__C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                                                        'clf__kernel': ['linear'],
                                                        'clf__gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}]

        Random Forest (RF)                      param_grid = [{'clf__n_estimators': [10, 100, 500],
                                                        'clf__max_features': ['sqrt', 'log2'],
                                                        'clf__bootstrap': [True],
                                                        'clf__criterion': ["gini"]}]

        k Nearest Neighbours (KNN)                param_grid = [{'clf__n_neighbors': [2, 5, 10, 15],
                                                            'clf__weights': ['uniform', 'distance'],
                                                            'clf__leaf_size':[15, 30, 60]}]

        Stochastic Gradient Descent (SGD)   param_grid = [{'clf__loss': ['hinge', 'log', 'modified_huber', 'perceptron'],
                                                            'clf__penalty': ['l2', 'l1','elasticnet'],
                                                            'clf__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],}]

        Logistic Regression CV (LR)        param_grid = [{'clf__Cs': param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100, 1000],
                                       '                   clf__solver': ['newton-cg', 'lbfgs', 'sag']}]

        Gaussian Naive Bayes (GNB)                param_grid = [{'clf__var_smoothing': [1e-12, 1e-9, 1e-4]}]

        Neural network  (NN)                     param_grid = [{'clf__activation': ['identity', 'logistic', 'tanh', 'relu'],
                                                            #'clf__solver': ['lbfgs', 'sgd', 'adam'],
                                                            #'clf__learning_rate': [ 'constant', 'invscaling', 'adaptive'],
                                                            'clf__batch_size': [0,5,10]}]

        Gradient Boosting (gboosting)       param_grid = [{'clf__loss': ['deviance', 'exponential'],
                                                            'clf__n_estimators': [10, 100, 500],
                                                            'clf__max_depth': [1,3,5,10]}]
        ====================              ==============================================================================
        Based on a function from moodlamp
        Müller A. T. et al. (2017) modlAMP: Python for anitmicrobial peptides, Bioinformatics 33, (17), 2753-2755,
        DOI:10.1093/bioinformatics/btx285
        """

        print("performing grid search...")

        if model.lower() == 'svm':
            pipe_svc = Pipeline([('scl', scaler),
                                 ('clf', SVC(class_weight='balanced', random_state=1, probability=True))])

            if param_grid is None:
                param_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
                param_grid = [{'clf__C': param_range,
                               'clf__kernel': ['linear'],
                               'clf__gamma': param_range
                               #'clf__kernel': ['rbf'] does not allow to retrieve feature importances with rbf kernel
                               }]

            gs = GridSearchCV(estimator=pipe_svc,
                              param_grid=param_grid,
                              # fit_params={'clf__sample_weight': sample_weights},
                              scoring=score,
                              cv=cv,
                              n_jobs=n_jobs)

            gs.fit(self.X_train, self.y_train)
            print("Best score %s (scorer: %s) and parameters from a %d-fold cross validation:" % (model.lower(),score, cv))
            print("MCC score:\t%.3f" % gs.best_score_)
            print("Parameters:\t%s" % gs.best_params_)

            # Set the best parameters to the best estimator
            best_classifier = gs.best_estimator_
            best_classifier_fit=best_classifier.fit(self.X_train, self.y_train)
            return best_classifier_fit

        elif model.lower() == 'rf':
            pipe_rf = Pipeline([('scl', scaler),
                                ('clf', RandomForestClassifier(random_state=1, class_weight='balanced'))])

            if param_grid is None:
                param_grid = [{'clf__n_estimators': [10, 100, 500],
                               'clf__max_features': ['sqrt', 'log2'],
                               'clf__bootstrap': [True],
                               'clf__criterion': ["gini"]}]

            gs = GridSearchCV(estimator=pipe_rf,
                              param_grid=param_grid,
                              scoring=score,
                              cv=cv,
                              n_jobs=n_jobs)

            gs.fit(self.X_train, self.y_train)
            print("Best score %s (scorer: %s) and parameters from a %d-fold cross validation:" % (model.lower(),score, cv))
            print("MCC score:\t%.3f" % gs.best_score_)
            print("Parameters:\t%s" % gs.best_params_)

            # Set the best parameters to the best estimator
            best_classifier = gs.best_estimator_
            best_classifier_fit=best_classifier.fit(self.X_train, self.y_train)
            return best_classifier_fit

        elif model.lower() == 'gboosting':
            pipe_rf = Pipeline([('scl', scaler),
                                ('clf', GradientBoostingClassifier(random_state=1))])

            if param_grid is None:
                param_grid = [{'clf__loss': ['deviance', 'exponential'],
                                'clf__n_estimators': [10, 100, 500],
                               'clf__max_depth': [1,3,5,10]}]

            gs = GridSearchCV(estimator=pipe_rf,
                              param_grid=param_grid,
                              scoring=score,
                              cv=cv,
                              n_jobs=n_jobs)

            gs.fit(self.X_train, self.y_train)
            print("Best score %s (scorer: %s) and parameters from a %d-fold cross validation:" % (model.lower(),score, cv))
            print("MCC score:\t%.3f" % gs.best_score_)
            print("Parameters:\t%s" % gs.best_params_)

            # Set the best parameters to the best estimator
            best_classifier = gs.best_estimator_
            best_classifier_fit=best_classifier.fit(self.X_train, self.y_train)
            return best_classifier_fit

        elif model.lower() == 'knn':
            pipe_rf = Pipeline([('scl', scaler),
                                ('clf', KNeighborsClassifier())])

            if param_grid is None:
                param_grid = [{'clf__n_neighbors': [2, 5, 10, 15],
                               'clf__weights': ['uniform', 'distance'],
                               'clf__leaf_size':[15, 30, 60]}]

            gs = GridSearchCV(estimator=pipe_rf,
                              param_grid=param_grid,
                              scoring=score,
                              cv=cv,
                              n_jobs=n_jobs)

            gs.fit(self.X_train, self.y_train)
            print("Best score %s (scorer: %s) and parameters from a %d-fold cross validation:" % (model.lower(),score, cv))
            print("MCC score:\t%.3f" % gs.best_score_)
            print("Parameters:\t%s" % gs.best_params_)

            # Set the best parameters to the best estimator
            best_classifier = gs.best_estimator_
            return best_classifier.fit(self.X_train, self.y_train)

        elif model.lower() == 'sgd':
            pipe_rf = Pipeline([('scl', scaler),
                                ('clf', SGDClassifier(class_weight='balanced', random_state=1))])

            if param_grid is None:
                param_grid = [{'clf__loss': ['hinge', 'log', 'modified_huber', 'perceptron'],
                               'clf__penalty': [None,'l2', 'l1','elasticnet'],
                                'clf__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}]

            gs = GridSearchCV(estimator=pipe_rf,
                              param_grid=param_grid,
                              scoring=score,
                              cv=cv,
                              n_jobs=n_jobs)

            gs.fit(self.X_train, self.y_train)
            print("Best score %s (scorer: %s) and parameters from a %d-fold cross validation:" % (model.lower(),score, cv))
            print("MCC score:\t%.3f" % gs.best_score_)
            print("Parameters:\t%s" % gs.best_params_)

            # Set the best parameters to the best estimator
            best_classifier = gs.best_estimator_
            return best_classifier.fit(self.X_train, self.y_train)

        elif model.lower() == 'lr':
            pipe_lr = Pipeline([('scl', scaler),
                                 ('clf', LogisticRegressionCV(class_weight='balanced', random_state=1))])

            if param_grid is None:
                param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100, 1000] # np.logspace(-4, 4, 10)
                param_grid = [{'clf__Cs': param_range,
                               'clf__solver': ['newton-cg', 'lbfgs', 'sag'], #solvers with l1 penalization
                               }]
            gs = GridSearchCV(estimator=pipe_lr,
                              param_grid=param_grid,
                              # fit_params={'clf__sample_weight': sample_weights},
                              scoring=score,
                              cv=cv,
                              n_jobs=n_jobs)

            gs.fit(self.X_train, self.y_train)
            print("Best score %s (scorer: %s) and parameters from a %d-fold cross validation:" % (model.lower(),score, cv))
            print("MCC score:\t%.3f" % gs.best_score_)
            print("Parameters:\t%s" % gs.best_params_)

            # Set the best parameters to the best estimator
            best_classifier = gs.best_estimator_
            best_classifier_fit=best_classifier.fit(self.X_train, self.y_train)
            return best_classifier_fit

        elif model.lower() == 'gnb':
            pipe_rf = Pipeline([('scl', scaler),
                                ('clf', GaussianNB(priors=None, var_smoothing=1e-09))])

            if param_grid is None:
                param_grid = [{'clf__var_smoothing': [1e-12, 1e-9, 1e-4]}]

            gs = GridSearchCV(estimator=pipe_rf,
                              param_grid=param_grid,
                              scoring=score,
                              cv=cv,
                              n_jobs=n_jobs)

            gs.fit(self.X_train, self.y_train)
            print("Best score %s (scorer: %s) and parameters from a %d-fold cross validation:" % (model.lower(),score, cv))
            print("MCC score:\t%.3f" % gs.best_score_)
            print("Parameters:\t%s" % gs.best_params_)

            # Set the best parameters to the best estimator
            best_classifier = gs.best_estimator_
            return best_classifier.fit(self.X_train, self.y_train)

        elif model.lower() == 'nn':
            pipe_rf = Pipeline([('scl', scaler),
                                ('clf', MLPClassifier())])

            if param_grid is None:
                param_grid = [{'clf__activation': ['identity', 'logistic', 'tanh', 'relu'],
                               #'clf__solver': ['lbfgs', 'sgd', 'adam'],
                                #'clf__learning_rate': [ 'constant', 'invscaling', 'adaptive'],
                               'clf__batch_size': [0,5,10]}]

            gs = GridSearchCV(estimator=pipe_rf,
                              param_grid=param_grid,
                              scoring=score,
                              cv=cv,
                              n_jobs=n_jobs)

            gs.fit(self.X_train, self.y_train)
            print("Best score %s (scorer: %s) and parameters from a %d-fold cross validation:" % (model.lower(),score, cv))
            print("MCC score:\t%.3f" % gs.best_score_)
            print("Parameters:\t%s" % gs.best_params_)

            # Set the best parameters to the best estimator
            best_classifier = gs.best_estimator_
            best_classifier_fit=best_classifier.fit(self.X_train, self.y_train)
            return best_classifier_fit

        else:
            print("Model not supported, please choose between 'svm', 'knn', 'sgd', 'rf', 'gnb', 'nn', 'gboosting' ")

    def plot_validation_curve(self,classifier, param_name, param_range,
                              cv=ShuffleSplit(n_splits=100, test_size=0.3, random_state=42),
                              score=make_scorer(matthews_corrcoef), title="Validation Curve",
                              xlab="parameter range", ylab="MCC", n_jobs=-1, filename=None):

        """This function plots a cross-validation curve for the specified classifier on all tested parameters given in the
        option 'param_range'.

        :param classifier: {classifier instance} classifier or validation curve (e.g. sklearn.svm.SVC).
        :param x_train: {array} descriptor values for training data.
        :param y_train: {array} class values for training data.
        :param param_name: {string} parameter to assess in the validation curve plot. For example,
        For SVM,
            "clf__C" (C parameter), "clf__gamma" (gamma parameter).
        For Random Forest,
            "clf__n_estimators" (number of trees),"clf__max_depth" (max num of branches per tree, "clf__min_samples_split" (min number of samples required to
            split an internal tree node), "clf__min_samples_leaf" (min number of samples in newly created leaf).
        :param param_range: {list} parameter range for the validation curve.
        :param cv: {int} number of folds for cross-validation.
        :param score: {metrics instance} scoring function built from make_scorer() or a predefined value in string form
            `sklearn.model_evaluation.scoring-parameter
            <http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_.
        :param title: {str} graph title
        :param xlab: {str} x axis label.
        :param ylab: {str} y axis label.
        :param n_jobs: {int} number of parallel jobs to use for calculation. if ``-1``, all available cores are used.
        :param filename: {str} if filename given the figure is stored in the specified path.
        :return: plot of the validation curve.

        Based on a function from moodlamp
        Müller A. T. et al. (2017) modlAMP: Python for anitmicrobial peptides, Bioinformatics 33, (17), 2753-2755,
        DOI:10.1093/bioinformatics/btx285
        """

        train_scores, test_scores = validation_curve(classifier, self.X_train, self.y_train, param_name, param_range,
                                                     cv=cv, scoring=score, n_jobs=n_jobs)
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

        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    def score_testset(self,classifier, sample_weights=None):
        """ Returns the tests set scores for the specified scoring metrics in a ``pandas.DataFrame``. The calculated metrics
        are Matthews correlation coefficient, accuracy, precision, recall, f1 and area under the Receiver-Operator Curve
        (roc_auc). See `sklearn.metrics <http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics>`_
        for more information.

        :param classifier: {classifier instance} pre-trained classifier used for predictions.
        :param x_test: {array} descriptor values of the tests data.
        :param y_test: {array} true class values of the tests data.
        :param sample_weights: {array} weights for the tests data.
        :return: ``pandas.DataFrame`` containing the cross validation scores for the specified metrics.

        """

        scores = []
        metrics = ['MCC', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
                   'TN', 'FP', 'FN', 'TP', 'FDR', 'sensitivity', 'specificity']
        funcs = ['matthews_corrcoef', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score', 'roc_auc_score',
                 'confusion_matrix']

        for f in funcs:
            # fore every metric, calculate the scores
            scores.append(getattr(mets, f)(self.y_test, classifier.predict(self.X_test), sample_weight=sample_weights))

        tn, fp, fn, tp = scores.pop().ravel()
        scores = scores + [tn, fp, fn, tp]
        fdr = float(fp) / (tp + fp)
        scores.append(fdr)
        sn = float(tp) / (tp + fn)
        scores.append(sn)
        sp = float(tn) / (tn + fp)
        scores.append(sp)
        df_scores = pd.DataFrame({'Scores': scores}, index=metrics)

        return df_scores.round(2)

    def plot_roc_curve(self,classifier, ylim=[0.0, 1.00], xlim=[0.0, 1.0],title='Receiver operating characteristic'): #for binary class
        """
        Function to plot a ROC curve
         On the y axis, true positive rate and false positive rate on the X axis.
         The top left corner of the plot is the 'ideal' point - a false positive rate of zero, and a true positive rate of one,
         meaning a larger area under the curve (AUC) is usually better.
        :param classifier: {classifier instance} pre-trained classifier used for predictions.
        :param x_test: {array} descriptor values of the tests data.
        :param y_test:{array} true class values of the tests data.
        :param ylim: y-axis limits
        :param xlim: x- axis limits
        :param title: title of plot
        :return: plot ROC curve
        """

        y_score = classifier.predict(self.X_test)
        #y_score = classifier.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(self.y_test, y_score)
        roc_auc = auc(fpr, tpr)
        print(roc_auc)
        plt.show()
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()

    def features_importances(self,classifier,model_name,top_features=20,color='b'):
        """
        Function that given a classifier retrieves the features importances as a dataset and represent as barplot.
        :param classifier: classifier
        :param model_name: model used in classifier. Choose between 'svm', 'sgd', 'lr', 'gboosting' or 'rf'
        :param x_train: descriptor values for training data.
        :param top_features: number of features to display on plot
        :param color: color of plot bard
        :return: bar plot of features and table with features names and importance for the model
        """

        feature_names=self.X_train.columns

        if model_name.lower() in ('rf', 'gboosting'):

            feature_importance=classifier.named_steps['clf'].feature_importances_
            feat_table=pd.DataFrame(feature_importance,index = self.X_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)

            #creating plot
            ind=feature_importance.ravel()
            indices = np.argsort(ind)[-top_features:] #top ranking features
            plt.figure(1)
            plt.title('Feature Importances')
            plt.barh(range(len(indices)), ind[indices], color=color, align='center')
            plt.yticks(range(len(indices)), feature_names[indices])
            plt.xlabel('Relative Importance')
            plt.show()

            print(feat_table)
            return feat_table

        elif model_name.lower() in ('svm' , 'sgd', 'lr'):

            feature_importance=classifier.named_steps['clf'].coef_
            coef = feature_importance.ravel()
            top_features=top_features//2 #to consider posiive and negative

            if (len(coef))<top_features: top_features=len(coef)

            top_positive_coefficients = np.argsort(coef)[-top_features:]
            top_negative_coefficients = np.argsort(coef)[:top_features]
            top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
            unique = []
            [unique.append(item) for item in top_coefficients if item not in unique] #remove duplicates
            top_coefficients=unique

            feat_table=pd.DataFrame(feature_importance[0],index = self.X_train.columns,columns=['importance']).sort_values('importance', ascending=False)
            print(feat_table)

            # create plot
            plt.figure(figsize=(15, 5))
            colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]

            plt.bar(np.arange(len(top_coefficients)), coef[top_coefficients], color=colors)
            feature_names = np.array(feature_names)
            plt.xticks(np.arange(0, 1+len(top_coefficients)), feature_names[top_coefficients], rotation=60, ha='right')
            plt.show()

            return feat_table

        else:
            print("Model not supported, please choose between 'svm', 'sgd', 'lr', 'rf' or 'gnb'")

    def plot_learning_curve(self,estimator, title='Learning curve {}', ylim=None,
                            cv=ShuffleSplit(n_splits=100, test_size=0.3, random_state=42),
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Plot a learning curve to determine cross validated training and tests scores for different training set sizes
        :param estimator: classifier/ model to use
        :param title: title of the plot
        :param ylim:
        :param cv: cross validation to use
        :param n_jobs:  number of parallel jobs to use for calculation. if ``-1``, all available cores are used.
        :param train_sizes: train sizes to tests
        :return: graphic representing learning curves, numbers of trainig examples, scores on training sets, and scores on tests set
        """

        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, self.X_data, self.Y_data, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

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
        plt.show()
        return train_sizes, train_scores, test_scores,plt

    def predict(self,classifier, x, seqs, names=None, y=None, filename=None):

        """This function can be used to predict novel peptides with a trained classifier model. The function returns a
        'pandas.DataFrame' with predictions using the specified estimator and tests data. If true class is provided,
        it returns the scoring value for the tests data.

        :param classifier: {classifier instance} classifier used for predictions.
        :param x: {array} descriptor values of the peptides to be predicted.
        :param seqs: {list} sequences of the peptides in ``x``.
        :param names: {list} (optional) names of the peptides in ``x``.
        :param y: {array} (optional) true (known) classes of the peptides.
        :param filename: {string} (optional) output filename to store the predictions to (``.csv`` format); if ``None``:
            not saved.
        :return: ``pandas.DataFrame`` containing predictions for ``x``. ``P_class0`` and ``P_class1``
            are the predicted probability of the peptide belonging to class 0 and class 1, respectively.

        Based on a function from moodlamp
        Müller A. T. et al. (2017) modlAMP: Python for anitmicrobial peptides, Bioinformatics 33, (17), 2753-2755,
        DOI:10.1093/bioinformatics/btx285
       """
        preds = classifier.predict_proba(x)

        predict=classifier.predict(x)

        if not (y and names):
            d_pred = {'P_class0': preds[:, 0], 'P_class1': preds[:, 1]}
            df_pred = pd.DataFrame(d_pred, index=seqs)

        elif not y:
            d_pred = {'Name': names, 'P_class0': preds[:, 0], 'P_class1': preds[:, 1]}
            df_pred = pd.DataFrame(d_pred, index=seqs)

        elif not names:
            d_pred = {'P_class0': preds[:, 0], 'P_class1': preds[:, 1], 'True_class': y}
            df_pred = pd.DataFrame(d_pred, index=seqs)

        else:
            d_pred = {'Name': names, 'P_class0': preds[:, 0], 'P_class1': preds[:, 1], 'True_class': y}
            df_pred = pd.DataFrame(d_pred, index=seqs)

        if filename:
            df_pred.to_csv(filename + time.strftime("-%Y%m%desc-%H%M%S.csv"))

        return df_pred

    # alternatively the user can use the function GetSubSeq(self,protein_sequence,ToAA='S',window=3)
    # from the readsequence module
    # that Get all 2*window+1 sub-sequences whose center is ToAA (a specific aminoacid) in a protein
    # giving a ToAA and a window

    def add_features(self,list_of_sequences,list_functions):
        """
        Calculate all features available in package or lis of descriptors given
        :param list_functions: list of features to be calculated with function adaptable from module Descriptors
        :param list_of_sequences: list of sequences to calculate features
        :return: dataframe with sequences and its features
        """

        features_list=[] #creating an empty list of dataset rows
        for seq in list_of_sequences:
            res={'sequence':seq}
            sequence=ReadSequence() #creating sequence object
            ps=sequence.read_protein_sequence(seq)
            protein = Descriptor(ps) # creating object to calculate descriptors)
            if len(list_functions)==0:
                feature=protein.get_all(tricomp=True, bin_aa=False, bin_prop=False)
            else:
                feature=protein.adaptable(list_functions)
            res.update(feature)
            features_list.append(res)

        df = pd.DataFrame(features_list)
        #df.set_index(['sequence'],inplace=True)
        return df

    def predict_window(self,classifier, seq,x=None, window_size=20,gap=1,features=[], features_names=None,
                       features_dataframe=None, names=None, y=None,filename=None):
        """Scan a protein in a sliding window approach to predict novel peptides with a trained classifier model.
        The function returns a
        'pandas.DataFrame' with predictions using the specified estimator and tests data. If true class is provided,
        it returns the scoring value for the tests data.
        The user can provide a features_dataframe if does not want to use the descriptors supported. Otherwise, the user
        can also provide a list with the numbers of the descriptors that want to be calculated (descriptors module adaptable())
        if none is provided, the function will calculate all the descriptors available.

        :param classifier: {classifier instance} classifier used for predictions.
        :param x: {array} descriptor values of the peptides to be predicted.
        :param seq:  sequence of the peptides in ``x``.
        :param window_size: number of aminoacids to considerer in each seq to check. for default 20
        :param gap: gap size of the search of windows in sequence. default 1
        :param features: list containing the list features to be calculated under de adaptable function.
                if list is empty will calculate all the descriptors available in the package
        :param features_dataframe: dataframe with sequences and its features
        :param features_names:names of features. If none will match the features with the ones with the model given
        :param names: {list} (optional) names of the peptides in ``x``.
        :param y: {array} (optional) true (known) classes of the peptides.
        :param filename: {string} (optional) output filename to store the predictions to (``.csv`` format); if ``None``:
            not saved.
        :return: ``pandas.DataFrame`` containing predictions for subsequences generated, ``P_class0`` and ``P_class1``
            are the predicted probability of the peptide belonging to class 0 and class 1, respectively. The previsions
            are divided in probability classes <0.99, >0.95, >0.9, <0.8, >0.7, >0.6 and 0
       """

        #generate final list of sequences/split sequences
        list_of_sequences,indices=sub_seq_sliding_window(seq,window_size,gap,index=True)


        #calculate features for sequences

        #if a dataframe with features calculated that will be considered
        #if not features will be calculated.
            # if features is none will calculate all the features according to the package and choose the features for the classifier
            # if features is list with numbers will call the adaptable function

        if features_dataframe != None: featuresDF=features_dataframe
        else: featuresDF = self.add_features(list_of_sequences,features)

        #select the features used for the model construction
        if features_names==None:
            features_to_select=self.X_data.columns
        else:
            features_to_select= features_names

        x_predict_data=featuresDF[features_to_select]

        # raw proability predictions of belonging or not
        preds = classifier.predict_proba(x_predict_data)
        #0 or 1 if belong or not
        predict=classifier.predict(x_predict_data)

        # dataframe with probabilities
        df_pred= self.predict(classifier, x_predict_data, list_of_sequences, names=names, y=y, filename=filename)
        df_pred['prevision']= predict
        df_pred['pos_0']=[i[0] for i in indices]
        df_pred['pos_-1']=[i[1] for i in indices]

        # create a dataframe specifying really the sequences, indices, 1 and 0 and probability
        # (can put scale >0.99, 095, 0.90, 0.8, 0.7, <0.7
        column=['sequence', 'prevision','probability','scale_probability', 'pos_0', 'pos_-1']
        sequence=''
        pos_0=int
        pos_1=int
        rows = []

        for index, row in df_pred.iterrows():
            value=row['P_class1']
            if value>=0.99:
                pos_0=row['pos_0']
                pos_1=row['pos_-1']
                rows.append({'sequence': index, 'prevision': 1, 'probability': value, 'scale_probability':5,
                             'pos_0':pos_0, 'pos_-1':pos_1 })

            if value>=0.95 and value<0.99:
                pos_0=row['pos_0']
                pos_1=row['pos_-1']
                rows.append({'sequence': index, 'prevision': 1, 'probability': value, 'scale_probability':4,
                             'pos_0':pos_0, 'pos_-1':pos_1 })

            if value>=0.90 and value<0.95:
                pos_0=row['pos_0']
                pos_1=row['pos_-1']
                rows.append( {'sequence': index, 'prevision': 1, 'probability': value, 'scale_probability':3,
                              'pos_0':pos_0, 'pos_-1':pos_1 })

            if value>=0.80 and value<0.90:
                pos_0=row['pos_0']
                pos_1=row['pos_-1']
                rows.append( {'sequence': index, 'prevision': 1, 'probability':value, 'scale_probability':2,
                              'pos_0':pos_0, 'pos_-1':pos_1 })

            if value>=0.70 and value<0.80:
                pos_0=row['pos_0']
                pos_1=row['pos_-1']
                rows.append( {'sequence': index, 'prevision': 1, 'probability': value, 'scale_probability':1,
                              'pos_0':pos_0, 'pos_-1':pos_1 })

            if value>=0.60 and value<0.70:
                pos_0=row['pos_0']
                pos_1=row['pos_-1']
                rows.append({'sequence': index, 'prevision': 1, 'probability': value, 'scale_probability':0,
                             'pos_0':pos_0, 'pos_-1':pos_1 })

            if value<0.6:
                pos_0=row['pos_0']
                pos_1=row['pos_-1']
                rows.append({'sequence': index, 'prevision': 0, 'probability': value, 'scale_probability':0,
                             'pos_0':pos_0, 'pos_-1':pos_1 })

        # create a new dataframe where consecutive rows in the same probability scale are joined and positions updated
        df = pd.DataFrame(rows) # dataframe with scale probabilities

        df['key'] = (df['scale_probability'] != df['scale_probability'].shift(1)).astype(int).cumsum()
        # add a sentinel column that tracks which group of consecutive data each row applies to

        df_new=df.drop(columns=['sequence', 'prevision'])
        x=df_new.values.tolist() #not in pandas dataframe
        remove_list=[]

        for row in range(len(x)-1):
            pos_fin=x[row][3]
            pos_ini=x[row][2]
            prob=round(x[row][0],3)
            x[row][0] = round(x[row][0], 4)
            scale_prob=x[row][1]
            key=x[row][4]

            if x[row][4]==x[row+1][4]:
                remove_list.append(row+1) #select rows to delete

        # update the positions. is not made in the same loop because in more than two consecutive rows it will
        # update a row that will be deleted
        for index in reversed(remove_list):
            x[index-1][0]=x[index][0]

        for index in sorted(remove_list, reverse=True): # remove rows that were joined
            del x[index]

        # add sequence
        for row in range(len(x)):
            seqs=seq[int(x[row][2]):int(x[row][3])]
            x[row].append(seqs)

        final_df=pd.DataFrame(x, columns=['probability','scale_probability','pos_0','pos_-1','key','sequence'])
        final_df=final_df.drop(columns=['key'])
        final_df = final_df[['pos_0','pos_-1','probability','scale_probability','sequence']]
        return final_df

