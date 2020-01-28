# -*- coding: utf-8 -*-
"""
##############################################################################

File containing a class intend to facilitate clustering analysis.
The functions are based on the package scikit learn.

Authors: Ana Marta Sequeira

Date: 06/2019

Email:

##############################################################################
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class Cluster:
    """
    The cluster class aims to perform and plot clustering analysis
    Based on scikit learn.
    """

    def _load_data(self, sklearn_load,target,test_size):
        """
        load the data. the inputs are inherited from the init function when the class is called.
        :return: selfs
        """
        data = sklearn_load
        X_data = pd.DataFrame(data)
        self.X_data=X_data
        self.Y_data=target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_data, target, test_size=test_size, random_state=42)

    def __init__(self, sklearn_load,target,test_size=0.3):
        """
        init function. When the class is called a dataset containing the features values and a target column must be provided.
        Test size is by default 0.3 but can be altered by user.
        :param sklearn_load: dataset X_data
        :param target: column with class labels
        :param test_size: size for division of the dataset in train and tests
        """
        self._load_data(sklearn_load,target,test_size)

    def kmeans(self, max_iter=300, n_clusters=None):
        """
        Function that performs K means cluster.
        :param max_iter: number of max terations for cluster (300 by default)
        :param n_clusters: if None, it will define the number of clusters as the number of existing labels
        :return: cross table with counts for labels vs classification in clusters
        """

        if not n_clusters: n_clusters = len(np.unique(self.y_train))

        clf = KMeans(n_clusters = n_clusters, max_iter=max_iter,random_state=42,init='k-means++')
        clf.fit(self.X_data)
        y_labels = clf.labels_
        centroids = clf.cluster_centers_

        #table
        table=pd.crosstab(y_labels, columns=self.Y_data, rownames=['clusters'])
        print(table)
        return table

    def hierarchical(self,metric='correlation', method='complete'):
        """
        Perform hierarchical clustering
        :param metric: distance metric to use in the case that y is a collection of observation vectors. eg. 'correlation', 'euclidean'
        see (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist)
        :param method: method to be used. exemples: 'complete', 'single', 'average', 'ward'
        see (https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)
        :return: dendogram of the clustered data
        """

        Z=linkage(self.X_data, metric=metric, method=method)
        plt.figure(figsize=(25,10))
        plt.title('Hierarchical Clustering Dendrogram metric: {}, method: {}'.format(metric,method))
        plt.xlabel('sample index')
        plt.ylabel('distance')
        dendrogram(Z,
                   labels=list(self.Y_data),
                   leaf_rotation=90, #rotate the x axis labels
                   leaf_font_size=8) #font size for the x axis labels
        label_colors={np.unique(self.y_train)[0]:'r', np.unique(self.y_train)[1]:'g'}
        ax=plt.gca()
        xlabels=ax.get_xmajorticklabels()
        for lbl in xlabels: lbl.set_color(label_colors[lbl.get_text()])
        plt.show()

    def classify(self, model=SVC(random_state=42)):
        """
        Function that fits the model in train datasets and predict on the tests dataset, returning the accuracy
        :param model: model to make prediction (SVC by default)
        :return: the accuracy of the prediction
        """

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        print('Accuracy: {}'.format(accuracy_score(self.y_test, y_pred)))

    def kmeans_predict(self, output='add', max_iter=300, n_clusters=None):
        """
        Perform the kmeans to train data and predict the tests set
        :param output: 'add' or 'replace'.
        If add, the labels produced by clustering will be added as features
        If replace, labels produced will be replace the old labels
        :param max_iter: max number of iterations of cluster
        :param n_clusters: if None, it will define the number of clusters as the number of existing labels
        :return: values of X datasets altered (if add) or the Y datasets replaced
        """
        if not n_clusters: n_clusters = len(np.unique(self.y_train))
        clf = KMeans(n_clusters = n_clusters, max_iter=max_iter,random_state=42)
        clf.fit(self.X_train)
        y_labels_train = clf.labels_
        y_labels_test = clf.predict(self.X_test)

        if output == 'add':
            self.X_train.loc[:,'km_clust'] = y_labels_train
            self.X_test.loc[:,'km_clust'] = y_labels_test
        elif output == 'replace':
            self.X_train = y_labels_train.loc[:, np.newaxis]
            self.X_test = y_labels_test.loc[:, np.newaxis]
        else:
            raise ValueError('output should be add or replace')
        return self




