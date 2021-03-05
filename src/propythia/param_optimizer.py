#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##############################################################################

File containing a class intend to facilitate parameter optimization. This class is not supposed to be ran outside
the Shallow learning and Deep learning classes.

The functions are based on the package scikit learn

Authors: Ana Marta Sequeira

Date: 12/2020

Email:

##############################################################################
"""
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef


# returns the best model fit but dont know if it uses the callbacks and stuff for deep learning models
# https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
class ParamOptimizer():
    """
    Class to optimize model parameters
    """

    def __init__(self, estimator, optType, paramDic, dataX, datay, cv=5, n_iter_search=15, n_jobs=1,
                 scoring=make_scorer(matthews_corrcoef), model_name=None, refit=True):

        self.optType = optType
        self.paramDic = paramDic
        self.dataX = dataX
        self.datay = datay
        self.model = estimator
        self.cv = cv
        self.n_iter_search = n_iter_search
        self.n_jobs = n_jobs
        self.score = scoring
        self.model_name = model_name
        self.final_units = len(np.unique(list(datay)))
        self.gs = None
        self.refit = refit

    # Utility function to report best scores
    def report_top_models(self, gs=None, n_top=3):
        """
        Utility function to report the top models
        :param gs:
        :param n_top:
        :return:
        """

        if gs is None:
            gs = self.gs
        print(gs)
        results = gs.cv_results_
        list_to_write = []
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                s1 = "Model with rank: {0}\n".format(i)
                s2 = "Mean validation score: {0:.3f} (std: {1:.3f})\n".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate])
                s3 = ("Parameters: {0:}\n".format(results['params'][candidate]))  # todo this values with 4 decimasl
                s4 = "\n"
                print(s1, s2, s3, s4)
                list_to_write.append([s1, s2, s3, s4])
        return list_to_write

    def report_all_models(self, gs=None):
        """
        Utility function to report all models tested
        :param gs:
        :return:
        """
        if gs is None:
            gs = self.gs
        print(self.score)
        print(self.cv)
        s1 = "Best score (scorer: %s) and parameters from a %d-fold cross validation:\n" % \
             (self.score, self.cv)
        s2 = "MCC score:\t%.3f\n" % gs.best_score_
        s3 = "Parameters:\t%s\n" % gs.best_params_
        print(s1, s2, s3)
        # results = gs.cv_results_
        # means = []
        # stds = []
        # for scorer in self.score:
        #     mean = {}
        #     std = {}
        #     mean[scorer] = results['mean_test_%s' % (scorer)]
        #     std[scorer] = results['std_test_%s' % (scorer)]
        #     means.append(mean)
        #     stds.append(std)

        means = gs.cv_results_['mean_test_score']
        stds = gs.cv_results_['std_test_score']
        params = gs.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        s4 = pd.DataFrame(params)
        s4['means'] = means
        s4['stds'] = stds
        print(s4)
        s4 = s4.sort_values(by=['means'], ascending=False)
        cols = ['means', 'stds'] + [col for col in s4 if col not in ('means', 'stds')]
        df = s4[cols]
        return s1, s2, s3, df

    def _randomized_search(self):
        # run randomized search
        start = time()
        gs = RandomizedSearchCV(estimator=self.model,
                                param_distributions=self.paramDic,
                                scoring=self.score,
                                cv=self.cv,
                                n_jobs=self.n_jobs,
                                n_iter=self.n_iter_search, refit=self.refit)
        gs = gs.fit(self.dataX, self.datay)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), self.n_iter_search))
        self.gs = gs
        return gs

    def _grid_search(self):
        # run grid search
        gs = GridSearchCV(estimator=self.model,
                          param_grid=self.paramDic,
                          scoring=self.score,
                          cv=self.cv,
                          n_jobs=self.n_jobs, refit=self.refit)
        start = time()
        gs.fit(self.dataX, self.datay)
        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(gs.cv_results_['params'])))

        return gs

    def get_opt_params(self):
        """
        Function to do gridSearch or randomizedSearch optimization
        :return: gridsearch object with metrics for the models and best model.  best model is accessed using
        gs.best_estimator_
        """
        if self.optType == 'gridSearch':
            gs = self._grid_search()
            return gs
        elif self.optType == 'randomizedSearch':
            gs = self._randomized_search()
            return gs
        else:
            print('Invalid optimization type!')
            return None
