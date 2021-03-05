
"""
##############################################################################

File containing tests functions to check if all functions from feature_selection module are properly working

Authors: Ana Marta Sequeira

Date: 06/2019 altered 01/2021

Email:

##############################################################################
"""
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from propythia.feature_selection import FeatureSelection


def test_feature_selection():
    dataset = pd.read_csv(r'datasets/dataset1_test_clean.csv', delimiter=',', encoding='latin-1')
    x_original=dataset.loc[:, dataset.columns != 'labels']
    labels=dataset['labels']


    # UNIVARIATE
    # **Select KBest** KBest with *mutual info classif*
    fs=FeatureSelection(x_original = x_original, target=labels, columns_names=x_original.columns,
                        dataset=None, report_name=None)
    transformer, x_fit_univariate, x_transf_univariate, column_selected, scores, scores_df = \
        fs.run_univariate(score_func=mutual_info_classif, mode='k_best', param=50)

    #**Select Percentile with *f classif*
    fs=FeatureSelection(x_original = x_original, target=labels, columns_names=x_original.columns,
                        dataset=None, report_name=None)
    transformer, x_fit_univariate, x_transf_univariate, column_selected, scores, scores_df = \
        fs.run_univariate(score_func=f_classif, mode='percentile', param=0.5)


    # Select only the features with p value inferior to 0.015
    fs=FeatureSelection(x_original = x_original, target=labels, columns_names=x_original.columns,
                        dataset=None, report_name=None)
    transformer, x_fit_univariate, x_transf_univariate, column_selected, scores, scores_df = \
        fs.run_univariate(score_func=f_classif, mode='fpr', param=0.05)


    # shape of transformed dataset
    print('shape of transformed dataset', x_transf_univariate.shape)
    # columns selected by high score
    scores = fs.scores_ranking(scores=scores, df_column_name='features ranking', all=False)
    print(scores)

    #
    # # Recursive feature elimination
    # #estimator=SVC kernel=linear with 5 cross validation
    # fs=FeatureSelection(x_original = x_original, target=labels, columns_names=x_original.columns,
    #                     dataset=None, report_name=None)
    #
    # transformer, x_fit_rfe, x_transf_rfe, column_selected, feat_impo_df =\
    #     fs.run_recursive_feature_elimination(cv=5, estimator=SVC(kernel="linear"), n_jobs=None,step=1)
    # # shape of transformed dataset
    # print(x_transf_rfe.shape)
    # # columns selected names
    # print(dataset.columns[column_selected])
    # # scores
    # print(feat_impo_df)


    # Select from model

    # L1-based feature selection   f linear_model.LogisticRegression/svm.LinearSVC for classification
    # With SVMs and logistic-regression, the parameter C controls the sparsity: the smaller C the fewer features selected.
    model_lsvc = LinearSVC(C=0.1, penalty="l1", dual=False)
    model_lr=LogisticRegression(C=0.1, penalty="l2", dual=False)
    model_tree=ExtraTreesClassifier(n_estimators=50)

    # model= Tree classifier. 50 estimators
    fs=FeatureSelection(x_original = x_original, target=labels, columns_names=x_original.columns,
                        dataset=None, report_name=None)
    select_model, x_transf_model, column_selected, feat_impo, feat_impo_df = \
        fs.run_from_model(model=model_tree)

    # model LR
    fs=FeatureSelection(x_original = x_original, target=labels, columns_names=x_original.columns,
                        dataset=None, report_name=None)
    select_model, x_transf_model, column_selected, feat_impo, feat_impo_df = \
        fs.run_from_model(model=model_lr)

    # model SVC
    fs=FeatureSelection(x_original = x_original, target=labels, columns_names=x_original.columns,
                        dataset=None, report_name=None)
    select_model, x_transf_model, column_selected, feat_impo, feat_impo_df = \
        fs.run_from_model(model=model_lsvc)

    print(feat_impo_df)

    print('original shape', dataset.shape)
    print('reduce shape', x_transf_model.shape)
    print('dataset reduced with column names\n', fs.get_transformed_dataset().head(3))

    dataset = fs.get_transformed_dataset()

if __name__ == "__main__":
    test_feature_selection()
