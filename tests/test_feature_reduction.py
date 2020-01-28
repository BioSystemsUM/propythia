"""
##############################################################################

File containing tests functions to check if all functions from feature_reduction module are properly working

Authors: Ana Marta Sequeira

Date: 06/2019

Email:

##############################################################################
"""
from propythia.feature_reduction import FeatureReduction
import pandas as pd

def test_feature_reduction():

    # read and define dataset
    dataset = pd.read_csv(r'datasets/dataset1_test_clean.csv', delimiter=',', encoding='latin-1')
    labels = dataset['labels']
    dataset = dataset.loc[:, dataset.columns != 'labels']

    # createObject
    fea_reduced=FeatureReduction()

    # perform pca
    pca,x_pca=fea_reduced.pca(dataset, n_components=50)

    # check the variance ratio of components
    print(fea_reduced.variance_ratio_components(x_pca))

    # table with the contribution of each feature to the pca. does not mean that are the most significant.
    # Unsupervised learning
    print(fea_reduced.contribution_of_features_to_component(dataset, pca, x_pca))

    # GRAPHS
    # bar plot with the contribution of each pca
    fea_reduced.pca_bar_plot(pca)

    # scatter plot of two principal components relative to labels
    fea_reduced.pca_scatter_plot(dataset, pca, x_pca, labels)

    print("Original shape: {}".format(str(dataset.shape)))
    print("Reduced shape: {}".format(str(x_pca.shape)))
    print('Variance explained by PC:', sum(pca.explained_variance_ratio_))
    print("Number of components {}".format(pca.n_components_))


if __name__ == "__main__":
    test_feature_reduction()
