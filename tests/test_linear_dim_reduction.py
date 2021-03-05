"""
##############################################################################

File containing tests functions to check if all functions from feature_reduction module are properly working

Authors: Ana Marta Sequeira

Date: 06/2019 altered 01/2021

Email:

##############################################################################
"""
from propythia.linear_dim_reduction import FeatureDecomposition
import pandas as pd

def test_linear_dim_reduction():

    # read and define dataset
    dataset = pd.read_csv(r'datasets/dataset1_test_clean.csv', delimiter=',', encoding='latin-1')
    labels = dataset['labels']
    dataset = dataset.loc[:, dataset.columns != 'labels']

    # createObject
    fd=FeatureDecomposition(fps_x=dataset, report_name=None, classes=labels)

    # perform pca
    pca,x_pca=fd.run_pca(n_components=50)

    # check the variance ratio of components
    ex_variance_ratio = fd.variance_ratio_components()
    print(ex_variance_ratio)

    # table with the contribution of each feature to the pca. does not mean that are the most significant.
    # result = fd.contribution_of_features_to_component()

    # GRAPHS
    # bar plot with the contribution of each pca
    fd.pca_bar_plot(show=True, path_save=None,
                    title='Percentage of explained variance ratio by PCA',
                    width=1, data=None, color='b', edgecolor='k', linewidth=0,
                    tick_label=None)

    fd.pca_cumulative_explain_ratio(show=True, path_save=None)

    # scatter plot of two principal components relative to labels
    fd.pca_scatter_plot(target=labels, pca1=0, pca2=1, title=None, show=True, path_save='pca_scatter_plot.png')
    # scatter plot of three principal components relative to labels
    fd.pca_scatter_plot3d(target=labels, pca1=0, pca2=1, pca3=2, title=None, show=True,
                   path_save='pca_scatter_plot.png')

    print("Original shape: {}".format(str(dataset.shape)))
    print("Reduced shape: {}".format(str(x_pca.shape)))
    print('Variance explained by PC:', sum(pca.explained_variance_ratio_))
    print("Number of components {}".format(pca.n_components_))

    # other algorithms
    # batch sparse pca
    fd=FeatureDecomposition(fps_x=dataset, report_name=None, classes=labels)
    fd.run_batch_sparse_pca(n_components=50, alpha=1, batch_size=50)

    # truncated svd
    fd=FeatureDecomposition(fps_x=dataset, report_name=None, classes=labels)
    fd.run_truncated_svd(n_components=50, n_iter=5, random_state=42)


if __name__ == "__main__":
    test_feature_reduction()
