"""
##############################################################################

File containing tests functions to check if all functions from clustering module are properly working

Authors: Ana Marta Sequeira

Date: 06/2019  ALTERED 01/2021

Email:

##############################################################################
"""
import pandas as pd

from propythia.clustering import Cluster
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize


def test_clustering():

    dataset = pd.read_csv(r'datasets/dataset1_test_clean_fselection.csv', delimiter=',')

    # separate labels
    x_original=dataset.loc[:, dataset.columns != 'labels']
    labels=dataset.loc[:,'labels']
    labels = label_binarize(labels, ['neg', 'pos'])

    # scale data
    scaler = StandardScaler()
    fps_x = scaler.fit_transform(x_original)

    # create the cluster object
    clust=Cluster(fps_x, target=labels, report_name=r'datasets/clust_report.txt')

    # perform K means
    clust.run_kmeans(max_iter=300, n_clusters=2, init='k-means++', random_state=42)
    # evaluate k means
    clust.clust_evaluation()

    # mini batch k means
    clust.run_minibatch_kmeans(max_iter=300, batch_size=100, n_clusters=None, init='k-means++', random_state=42)

    clust.hierarchical_dendogram(metric='correlation', method='complete', path_save=None, show=True,
                           truncate_mode='level', p=3)

    clust.hierarchical_dendogram(metric='correlation', method='average', path_save=None, show=True,
                                 truncate_mode='level', p=3)
    clust.hierarchical_dendogram(metric='euclidean', method='ward', path_save=None, show=True,
                                 truncate_mode='level', p=3)


if __name__=="__main__":
    test_clustering()
