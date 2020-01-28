# -*- coding: utf-8 -*-
"""
##############################################################################

File containing a class used for reducing the number of features on a dataset based
on unsupervised techniques.

Authors: Ana Marta Sequeira

Date: 06/2019

Email:

##############################################################################
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#from scores import score_methods


class FeatureReduction:
    """
    The Feature Reduction class aims to Reduce the number of features on a dataset based on unsupervised techniques.
    pca statistical procedure that orthogonally transforms the original n coordinates of a data set into a new set of
    n coordinates called principal components. Principal components are a combination of features that capture well the
    variance of the original features.
    Based on scikit learn
    """

    def __init__(self):
        """	constructor """

    def pca(self, dataset, scaler=StandardScaler(), n_components=None, copy=True, whiten=False, svd_solver='auto',
            tol=0.0, iterated_power='auto', random_state=None):
        """
        Function that realizes the pca analysis
        :param dataset: data on to perform pca
        :param scaler: scaler to scale data. standard scaler by default
        :param n_components: Number of components to keep. if n_components is not set all components are kept
        :param copy:
        :param whiten:
        :param svd_solver:  string {‘auto’, ‘full’, ‘arpack’, ‘randomized’}
        :param tol:
        :param iterated_power:
        :param random_state:
        :return:
        For more information: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        """
        # scaling data using standard scaller. VEr o scale, normalization, log cenas.....tenho de fazer os imports se nao n conhece
        scaler.fit(dataset)
        x_scaled=scaler.transform(dataset)

        # performing pca
        pca=PCA(n_components, copy, whiten, svd_solver, tol, iterated_power, random_state)
        pca.fit(x_scaled)
        x_pca=pca.transform(x_scaled)

        # analysis
        # print("Original shape: {}".format(str(x_scaled.shape)))
        # print("Reduced shape: {}".format(str(x_pca.shape)))
        # print('Variance explained by the PC:', sum(pca.explained_variance_ratio_))
        # print("Number of components {}".format(pca.n_components_))
        # print("pca components by explained variance ratio:\n{}".format(pca.components_))

        return pca, x_pca
        # it is possible to access all the information in scikit learn

    def variance_ratio_components(self,x_pca):
        """
        measures the variance ratio of the principal components
        :param x_pca:
        :return: variance ratio of principal components
        """
        ex_variance=np.var(x_pca,axis=0)
        ex_variance_ratio = ex_variance/np.sum(ex_variance)
        return ex_variance_ratio

    def contribution_of_features_to_component(self, data, pca, x_pca):
        """
        Function that retrieves a dataframe containing the contribution of each feature (rows) for component
        As unsupervised learning does not represent the importance of features but representing the directions
        of maximum variance in the data.
        :param data: dataset as dataframe
        :param pca: dataset fit to pca
        :param x_pca: dataset transformed to pca
        :return: dataframe containing the contribution of each feature (rows) for component
        """

        data=data
        coef = pca.components_.T
        columns=[]
        for x in range(pca.n_components_):
            columns.append(str('PC-'+str(x+1)))
        return pd.DataFrame(coef, columns=columns, index=data.columns)

    def pca_bar_plot(self, pca, height=1, width=1, data=None, color='b', edgecolor='k', linewidth=0, tick_label=None):
        """
        function that derives a bar plot representing the percentage of explained variance ratio by pca
        :param pca: dataset fit to pca
        :param height: scalar or sequence of scalars. The height(s) of the bars.
        :param width:  scalar or array-like, optional. The width(s) of the bars
        :param data:
        :param color: scalar or array-like, optional. The colors of the bar faces.
        :param edgecolor:  scalar or array-like, optional. The colors of the bar edges.
        :param linewidth: scalar or array-like, optional. Width of the bar edge(s). If 0, don't draw edges.
        :param tick_label: string or array-like, optional. The tick labels of the bars. Default: None
        :return: bar plot representing the percentage of explained variance ratio by pca
        For more information please see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.bar.html
        """
        plt.bar(range(pca.n_components_),height=pca.explained_variance_ratio_*100,width=width,data=data,color=color,
                edgecolor=edgecolor,linewidth=linewidth,tick_label=tick_label)
        plt.xlabel('Principal components')
        plt.ylabel('Percentage of explained variance')
        plt.show()

    def pca_scatter_plot(self, data, pca, x_pca, target, pca1=0, pca2=1, title='pca'):
        """
        Scatter plot of the labels based on two components (by default the first ones)
        :param title: string. title of the scatter plot
        :param data: dataset. dataframe
        :param pca: dataset fit to pca
        :param x_pca: dataset transformed to pca
        :param target: labels of dataset
        :param pca1: first pca to be considered. default PCA1
        :param pca2: second pca to be considered. default PCA2
        :return: graph showing the positions of labels according of the two chosen components
        """

        for classe in target.unique():
            sp=data.index[target == classe]
            plt.plot(x_pca[sp,pca1], x_pca[sp,pca2], 'o', label=classe)
        plt.title(title)
        plt.legend(loc='best', shadow=False)
        plt.show()

