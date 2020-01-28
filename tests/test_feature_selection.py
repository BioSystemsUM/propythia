
"""
##############################################################################

File containing tests functions to check if all functions from feature_selection module are properly working

Authors: Ana Marta Sequeira

Date: 06/2019

Email:

##############################################################################
"""
from propythia.feature_selection import FeatureSelection
import pandas as pd
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import f_classif, mutual_info_classif
#from sklearn.feature_selection import f_regression, mutual_info_regression #for regression problems
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from propythia.adjuv_functions.scoring.scores import score_methods


def test_feature_selection():
    dataset = pd.read_csv(r'datasets/dataset1_test_clean.csv', delimiter=',', encoding='latin-1')
    x_original=dataset.loc[:, dataset.columns != 'labels']
    labels=dataset['labels']

    fselect=FeatureSelection(dataset, x_original, labels)

    # #**Select KBest**
    # #KBest com *mutual info classif*
    X_fit_univariate, X_transf_univariate,column_selected,scores,dataset_features = \
        fselect.univariate(score_func=mutual_info_classif, mode='k_best', param=1000)

    #**Select Percentile
    #Percentile with *f classif*
    X_fit_univariate, X_transf_univariate,column_selected,scores,dataset_features = \
        fselect.univariate(score_func=f_classif, mode='percentile', param=0.6)

    # Select only the features with p value inferior to 0.015
    X_fit_univariate, X_transf_univariate,column_selected,scores,dataset_features \
        = fselect.univariate(score_func=f_classif, mode='fpr', param=0.05)

    #shape of transformed dataset
    print(X_transf_univariate.shape)
    #columns selected by high score
    print(fselect.features_scores(x_original,scores,column_selected, False))



    #**SRecursive feature elimination
    #estimator=SVC kernel=linear with 5 cross validation
    X_fit_rfe, X_transf_rfe,column_selected,ranking,dataset_features= \
        fselect.recursive_feature_elimination(cross_validation=True,cv=5)

    #shape of transformed dataset
    print(X_transf_rfe.shape)
    #columns selected names
    print(dataset.columns[column_selected])
    #scores
    score_methods(x_original,X_transf_rfe,labels)
    #
    # Select from model

    # L1-based feature selection   f linear_model.LogisticRegression/svm.LinearSVC for classification
    # With SVMs and logistic-regression, the parameter C controls the sparsity: the smaller C the fewer features selected.
    model_lsvc = LinearSVC(C=0.1, penalty="l1", dual=False)
    model_lr=LogisticRegression(C=0.1, penalty="l2", dual=False)
    model_tree=ExtraTreesClassifier(n_estimators=50)


    # Select from model
    #model= Tree classifier. 50 estiamtors
    X_fit_model, X_transf_model,column_selected,feature_importances,feature_importances_DF,dataset_features= \
        fselect.select_from_model_feature_elimination(model=model_tree)

    #model= logistic regression

    X_fit_model, X_transf_model,column_selected,feature_importances,feature_importances_DF,dataset_features= \
        fselect.select_from_model_feature_elimination( model=model_lr)

    #model= linearsvs
    X_fit_model, X_transf_model,column_selected,feature_importances,feature_importances_DF,dataset_features= \
        fselect.select_from_model_feature_elimination(model=model_lsvc)

    print('original shape', dataset.shape)
    print('reduce shape', fselect.dataset.shape)
    print('dataset reduced with column names\n', fselect.dataset.head(3))
    print('feature importances\n',feature_importances_DF)
    print('scores')
    score_methods(x_original,X_transf_model,labels)
    print(fselect.dataset)


if __name__ == "__main__":
    test_feature_selection()
