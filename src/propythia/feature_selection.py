# -*- coding: utf-8 -*-
"""
############################################################################
File containing a class used for feature selection. The FeatureSelection class aims to select features based on supervised algorithms in order to improve
estimators’ accuracy scores or to boost their performance on very high-dimensional datasets.

Authors: Ana Marta Sequeira

Date:06/2019 altered 12/2020

Email:

##############################################################################
"""

import pandas as pd
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
# from sklearn.feature_selection import f_regression, mutual_info_regression #for regression problems
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
# from scores import score_methods
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from propythia.adjuv_functions.ml_deep.utils import timer
import numpy as np


class FeatureSelection:
    """
     The FeatureSelection class aims to select features to improve estimators’ accuracy scores or to boost
     their performance on very high-dimensional datasets.
     It implements sklearn functions
     """

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

    # todo pass the variance threshold to here
    # fix the preprocess to integratae in propythia
    def __init__(self, x_original, target, columns_names, dataset=None, report_name=''):
        """
        init function. When the class is called a dataset containing the features values and a target column must be provided.
        Test size is by default 0.3 but can be altered by user.
        :param dataset:
        :param x_original: dataset X_data to sklearn_load
        :param target: column with class labels
        :param test_size: column with class labels
        """
        self.dataset = dataset
        if columns_names is None:
            if dataset is not None:
                self.columns_names = dataset.columns
            else:
                self.columns_names = list(range(len(self.X_data)))
        else:
            self.columns_names = columns_names

        self.x_original = x_original
        self.X_data = x_original
        self.Y_data = target
        self.column_selected = None
        self.report_name = report_name
        if self.report_name:
            self._report(str(self.report_name))

    def get_x_data(self):
        return self.X_data

    def get_transformed_dataset(self):
        df = pd.DataFrame(self.X_data, columns=self.column_selected)
        df['label'] = self.Y_data
        return df

    @timer
    def run_univariate(self, score_func=mutual_info_classif, mode='percentile', param=50, **params):
        """
        Univariate feature selector, it selects works the best features based on univariate statistical tests.
        It can select the k highest scoring features or a user specified percentage of features.
        Scoring functions for classification problems can be chi2, f_classif or mutual_info_classif

        :param scaler: scaler function to use to datasets before apply univariate tests.
        It can be None or any function supported by SKlearn like StandardScaler()
        :param score_func: function that returns univariate scores and p-values (or only scores for SelectKBest and SelectPercentile)
                    ( for classification: chi2, f_classif, mutual_info_classif
        :param mode: feature selection mode (‘percentile’, ‘k_best’, ‘fpr’, ‘fdr’, ‘fwe’)
        :param param:parameter of corresponding mode
        :return: univariate scores and p-values
        """
        saved_args = locals()

        transformer = GenericUnivariateSelect(score_func, mode=mode, param=param, **params)
        x_fit_univariate = transformer.fit(self.X_data, self.Y_data)
        x_transf_univariate = transformer.transform(self.X_data)
        scores = x_fit_univariate.scores_  # scores of features
        p_values = x_fit_univariate.pvalues_  # p features

        # get columns selected
        column_selected = transformer.get_support(indices=True)  # indexes of selected columns
        self.column_selected = column_selected

        scores_df = self.scores_ranking(scores=scores, df_column_name='scores_ranking', all=False)

        s1 = 'original X dataset shape: {}'.format(self.X_data.shape)
        s2 = 'New X dataset shape: {}'.format(x_transf_univariate.shape)
        s3 = 'number of column selected: {}'.format(column_selected.shape)
        s4 = 'scores: {}'.format(scores_df)
        s5 = 'column selected: {}'.format(column_selected)
        print('{}\n{}\n{}\n{}\n{}'.format(s1, s2, s3, s4, s5))
        # assign the new X
        self.X_data = x_transf_univariate

        if self.report_name:
            self._report([self.run_univariate.__name__, saved_args, s1, s2, s3, s4, s5])
            self._report(scores_df[:100], dataframe=True)

        return transformer, x_fit_univariate, x_transf_univariate, column_selected, scores, scores_df

    # todo retrieves a lot. it gets cofusing and lots of work to do fit in one dataset and transform in other. check.
    #  put the other funtions retrieving in the same way

    @timer
    def run_recursive_feature_elimination(self, cv=None, estimator=SVC(kernel="linear"), n_jobs=None,
                                          step=1, **params):
        """
        Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), recursive feature elimination (RFE)
        is to select features by recursively considering smaller and smaller sets of features.
        First, the estimator is trained on the initial set of features and the importance of each feature is obtained either
        through a coef_ attribute or through a feature_importances_ attribute.
        Then, the least important features are pruned from current set of features.That procedure is recursively repeated on the pruned set until
        the desired number of features to select is eventually reached.

        RFECV performs RFE in a cross-validation loop to find the optimal number of features.

        :param scaler: scaler function to use to datasets before apply univariate tests.
        It can be None or any function supported by SKlearn like StandardScaler()
        :param cross_validation: if yes: RFECV . if not: RFE
        :param estimator: estimator that assign wights to features
        :param n_features_to_select: to RFE
        :param min_features_to_select: to RFECV
        :param cv: number of folds in cross validation
        :param scoring: for RFECV
        :param n_jobs:
        :param step:If greater than or equal to 1, then step corresponds to the (integer) number of features to remove
        at each iteration. If within (0.0, 1.0), then step corresponds to the percentage (rounded down) of features to
        remove at each iteration
        PUT HERE what are the params that can be added !!!!
        :param verbose:
        :return: rfe fit, rfe transformed, original dataset with features selected, columns names and the features ranking
        """
        # todo check if it is ok with small dataset

        saved_args = locals()
        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE
        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV

        if cv == 1:
            transformer = RFE(estimator, step, **params)
        else:
            transformer = RFECV(estimator, step, cv, n_jobs, **params)

        x_fit_rfe = transformer.fit(self.X_data, self.Y_data)
        x_transf_rfe = transformer.transform(self.X_data)
        # ranking
        features_ranking = transformer.ranking_

        # get columns selected
        column_selected = transformer.get_support(indices=True)  # indexes of selected columns
        self.column_selected = column_selected
        feat_impo_df = self.scores_ranking(scores=features_ranking, df_column_name='features_ranking', all=False)

        s1 = 'original X dataset shape: {}'.format(self.X_data.shape)
        s2 = 'New X dataset shape: {}'.format(x_transf_rfe.shape)
        s3 = 'number of column selected: {}'.format(column_selected.shape)
        s4 = 'features ranking: {}'.format(feat_impo_df)
        s5 = 'column selected: {}'.format(column_selected)
        print('{}\n{}\n{}\n{}\n{}'.format(s1, s2, s3, s4, s5))

        # assign the new X
        self.X_data = x_transf_rfe
        if self.report_name:
            self._report([self.run_recursive_feature_elimination.__name__, saved_args, s1, s2, s3, s4, s5])
            self._report(feat_impo_df[:30], dataframe=True)
        return x_fit_rfe, column_selected, feat_impo_df

    @timer
    def run_from_model(self, model=LinearSVC(C=0.1, penalty="l1", dual=False), **params):
        """
        SelectFromModel is a meta-transformer that can be used along with any estimator that has a coef_ or feature_importances_ attribute
        after fitting.
        The features are considered unimportant and removed, if the corresponding coef_ or feature_importances_ values are
        below the provided threshold parameter.

        :param scaler: scaler function to use to datasets before apply univariate tests.
        It can be None or any function supported by SKlearn like StandardScaler()
        :param data: dataset to perform the feature selection
        :param model:
                examples:
                ExtraTreesClassifier(n_estimators=50)
                LinearSVC(C=0.01, penalty="l2", dual=False)
                LogisticRegression(C=0.1, penalty="l1", dual=False)

        :return:
        """
        saved_args = locals()

        transformer = model.fit(self.X_data, self.Y_data)
        select_model = SelectFromModel(transformer, prefit=True, **params)
        column_selected = select_model.get_support(indices=True)  # indexes of selected columns
        self.column_selected = column_selected
        x_transf_model = select_model.transform(self.X_data)

        if 'Tree' in str(model):
            feat_impo = transformer.feature_importances_
        else:
            feat_impo = transformer.coef_[0]

        feat_impo_df = self.scores_ranking(scores=feat_impo, df_column_name='features importance', all=False)

        s1 = 'original X dataset shape: {}'.format(self.X_data.shape)
        s2 = 'New X dataset shape: {}'.format(x_transf_model.shape)
        s3 = 'number of column selected: {}'.format(column_selected.shape)
        s4 = 'features importance: {}'.format(feat_impo_df)
        s5 = 'column selected: {}'.format(column_selected)
        print('{}\n{}\n{}\n{}\n{}'.format(s1, s2, s3, s4, s5))

        # assign the new X
        self.X_data = x_transf_model
        if self.report_name:
            self._report([self.run_from_model.__name__, saved_args, s1, s2, s3, s4, s5])
            self._report(feat_impo_df[:30], dataframe=True)
        return select_model, x_transf_model, column_selected, feat_impo, feat_impo_df

    def scores_ranking(self, scores, df_column_name='features ranking', all=False):
        """
        Retrieves a dataframe with features names and scores of importance
        :param scores: list of scores of the features (can be obtained by the function univariate)
        :param column_selected: list containing the indexes of the selected columns (can be obtained by the function univariate)
        :param all: to return all the features and scores or only the selected ones (by default)
        :return: a dataframe containing the names of features and the scores of univariate tests by descending importance
        """
        if all:
            return pd.DataFrame(scores, index=self.columns_names,
                                columns=[df_column_name]).sort_values(by=[df_column_name], ascending=False)
        else:
            # get scores of column selected
            score = []
            for index in self.column_selected:
                score.append(scores[index])
            columns_names_new = np.asarray(self.columns_names)[self.column_selected]

            return pd.DataFrame(score, index=columns_names_new,
                                columns=[df_column_name]).sort_values(by=[df_column_name], ascending=False)
