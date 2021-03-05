"""
##############################################################################

File containing tests functions to check if all functions from clustering module are properly working

Authors: Ana Marta Sequeira

Date: 06/2019  ALTERED 01/2021

Email:

##############################################################################
"""
import pandas as pd
import numpy as np
from propythia.clustering import Cluster
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize


def test_clustering():

    dataset = pd.read_csv(r'datasets/dataset1_test_clean_fselection.csv', delimiter=',')

    # separate labels
    x_original=dataset.loc[:, dataset.columns != 'labels']
    labels=dataset.loc[:,'labels']
    lab = label_binarize(labels, ['neg', 'pos'])
    labels =[item for sublist in lab for item in sublist]
    # scale data
    scaler = StandardScaler()
    fps_x = scaler.fit_transform(x_original)

    # create the cluster object
    cl=Cluster(fps_x, target=labels, report_name=None)

    # perform K means
    clf, y_labels, centroids = cl.run_kmeans(max_iter=300, n_clusters=None, init='k-means++', random_state=42)

    # minibatch
    clf, y_labels, centroids = cl.run_minibatch_kmeans(max_iter=300, batch_size=100,
                                                       n_clusters=2, init='k-means++', random_state=42)

    cl=Cluster(fps_x, target=labels, report_name=None)

    # cl.hierarchical_dendogram(metric='correlation', method='complete', path_save=None, show=True,
    #                        truncate_mode='level', p=3)
    #
    # cl.hierarchical_dendogram(metric='correlation', method='average', path_save=None, show=True,
    #                              truncate_mode='level', p=3)
    # cl.hierarchical_dendogram(metric='euclidean', method='ward', path_save=None, show=True,
    #                              truncate_mode='level', p=3)

    hmodel, y_labels, n_leaves = \
        cl.run_hierarchical(n_clusters=None, affinity='euclidean', linkage='ward', y_data=None)
    cl.plot_dendrogram(model=hmodel, title='Hierarchical Clustering Dendrogram', truncate_mode='level', p=3,
                    path_save=None, show=True)
if __name__=="__main__":
    test_clustering()
