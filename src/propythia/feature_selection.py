# -*- coding: utf-8 -*-
"""
##############################################################################

File containing a class used for feature selection. The FeatureSelection class aims to select features based on supervised algorithms in order to improve
estimators’ accuracy scores or to boost their performance on very high-dimensional datasets.

Authors: Ana Marta Sequeira

Date:06/2019

Email:

##############################################################################
"""

import pandas as pd
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import f_classif, mutual_info_classif
#from sklearn.feature_selection import f_regression, mutual_info_regression #for regression problems
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
#from scores import score_methods
from sklearn.svm import SVC
from sklearn.feature_selection import RFE,RFECV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class FeatureSelection:
    """
     The FeatureSelection class aims to select features to improve estimators’ accuracy scores or to boost
     their performance on very high-dimensional datasets.
     It implements sklearn functions
     """
    def _load_data(self, dataset, x_original,target,test_size):
        """
        load the data. the inputs are inherited from the init function when the class is called.
        :param dataset:
        :param x_original:
        :param target:
        :param test_size:
        :return:
        """

        self.dataset = pd.DataFrame(dataset)
        self.X_data=x_original
        self.Y_data=target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_data, self.Y_data, test_size=test_size, random_state=42)
        self.test_size=test_size

    def __init__(self, dataset, x_original, target, test_size=0.3):
        """
        init function. When the class is called a dataset containing the features values and a target column must be provided.
        Test size is by default 0.3 but can be altered by user.
        :param dataset:
        :param x_original: dataset X_data to sklearn_load
        :param target: column with class labels
        :param test_size: column with class labels
        """

        self._load_data(dataset, x_original,target,test_size)

    def univariate(self,score_func=mutual_info_classif, mode='percentile', param=50,scaler = StandardScaler()):
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

        if scaler: self.X_data = scaler.fit_transform(self.X_data)

        transformer = GenericUnivariateSelect(score_func, mode, param)
        x_fit_univariate = transformer.fit(self.X_data, self.Y_data)
        x_transf_univariate = transformer.transform(self.X_data)

        scores = x_fit_univariate.scores_ #scores of features
        #original dataset with columns selected (get feature names back)
        column_selected = transformer.get_support(indices=True) #indexes of selected columns
        dataset_features = self.dataset.iloc[:, column_selected] #dataset with features names select
        dataset_features = dataset_features.assign(labels= self.Y_data) #put labels back
        self.dataset = dataset_features
        self.X_data = self.X_data[:, column_selected] #transform the X_data shape
        return x_fit_univariate, x_transf_univariate,column_selected,scores,dataset_features

    def features_scores(self, x_original,scores, column_selected, all=False):
        """
        Retrieves a dataframe with features names and scores of importance resulting of the univariate tests
        :param dataset: original dataset with columns names
        :param scores: list of scores of the features (can be obtained by the function univariate)
        :param column_selected: list containing the indexes of the selected columns (can be obtained by the function univariate)
        :param all: to return all the features and scores or only the selected ones (by default)
        :return: a dataframe containing the names of features and the scores of univariate tests by descending importance
        """

        if all:
            return pd.DataFrame(scores,index=x_original.columns,columns=['scores']).sort_values(by=['scores'],ascending=False)
        else:
            # get scores of column selected
            score=[]
            for index in column_selected:
                score.append(scores[index])
            columns_names=x_original.columns[column_selected]

            return pd.DataFrame(score,index=columns_names,columns=['scores']).sort_values(by=['scores'],ascending=False)

    def recursive_feature_elimination(self,cross_validation=False,estimator=SVC(kernel="linear"),
                                      n_features_to_select=None,
                                      min_features_to_select=1,cv=None,scoring=None,n_jobs=None,
                                      step=1, verbose=0,scaler = StandardScaler()):
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
        :param verbose:
        :return: rfe fit, rfe transformed, original dataset with features selected, columns names and the features ranking
        """

        if scaler: self.X_data= scaler.fit_transform(self.X_data)
        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE
        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV

        if cross_validation:
            transformer=RFECV(estimator, step, min_features_to_select, cv, scoring, verbose, n_jobs)

        else:
            transformer=RFE(estimator, n_features_to_select, step, verbose)

        x_fit_rfe=transformer.fit(self.X_data, self.Y_data)
        x_transf_rfe=transformer.transform(self.X_data)

        #original dataset with columns selected (get feature names back)
        column_selected=transformer.get_support(indices=True) #indexes of selected columns
        dataset_features=self.dataset.iloc[:, column_selected]#dataset with features names select
        dataset_features=dataset_features.assign(labels= self.Y_data) #put labels back
        self.dataset=dataset_features
        self.X_data=self.X_data[:, column_selected] #transform the X_data shape
        #ranking
        features_ranking=transformer.ranking_

        return x_fit_rfe, x_transf_rfe,column_selected,features_ranking,dataset_features

    def rfe_ranking(self,features_ranking):
        """
        Retrieves a dataframe with features names and its ranking position ordered. Positions 1 are the selected ones.
        :param dataset: dataset used to performed te fre
        :param features_ranking: array containing the features ranking obtained with rfe
        :return: dataframe containing the features names and its position ranking ordered by ranking
        """
        return pd.DataFrame(features_ranking,index=self.dataset.columns,columns=['features ranking']).sort_values(by=['features ranking'],ascending=True)

    def select_from_model_feature_elimination(self,model=LinearSVC(C=0.1, penalty="l1", dual=False), scaler = StandardScaler()):
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
                LinearSVC(C=0.01, penalty="l1", dual=False)
                LogisticRegression(C=0.1, penalty="l1", dual=False)

        :return:
        """
        if scaler:
            self.X_data= scaler.fit_transform(self.X_data)
            #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_data, self.Y_data, test_size=self.test_size, random_state=42)
        transformer=model.fit(self.X_data,self.Y_data )
        select_model = SelectFromModel(transformer, prefit=True)
        # features = np.array(data.columnnames())
        # print(features[select_model.get_support()])
        column_selected=select_model.get_support(indices=True) #indexes of selected columns
        dataset_features=self.dataset.iloc[:, column_selected] #dataset with features names select
        dataset_features=dataset_features.assign(labels= self.Y_data) #put labels back

        x_transf_model = select_model.transform(self.X_data)

        if 'Tree' in str(model):
            feature_importances=transformer.feature_importances_
            feature_importances_DF = pd.DataFrame(feature_importances,index = self.dataset.iloc[:,:-1].columns,columns=['importance']).sort_values('importance', ascending=False)

        else:
            feature_importances=transformer.coef_
            feature_importances=feature_importances[0]
            feature_importances_DF = pd.DataFrame(feature_importances,index = self.dataset.iloc[:,:-1].columns,columns=['importance']).sort_values('importance', ascending=False)
        self.dataset=dataset_features
        self.X_data=self.X_data[:, column_selected] #transform the X_data shape
        return transformer, x_transf_model,column_selected,feature_importances,feature_importances_DF,dataset_features


