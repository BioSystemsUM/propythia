"""
##############################################################################

File containing tests functions to check if all functions from clustering module are properly working

Authors: Ana Marta Sequeira

Date: 06/2019

Email:

##############################################################################
"""
from propythia.clustering import Cluster
from sklearn.preprocessing import StandardScaler
import timeit
import pandas as pd
from sklearn.svm import SVC

def test_clustering():

    dataset = pd.read_csv(r'datasets\dataset1_test_clean_fselection.csv', delimiter=',')

    #separate labels
    x_original=dataset.loc[:, dataset.columns != 'labels']
    labels=dataset.loc[:,'labels']

    #scale data
    scaler = StandardScaler()
    scaler.fit_transform(x_original)

    #create the cluster object
    clust=Cluster(x_original,labels)

    # #perform K means
    clust.kmeans_predict().classify(model=SVC())
    clust.kmeans()

    #perform hierarchical clustering
    clust.hierarchical(metric='correlation', method='average')
    clust.hierarchical(metric='euclidean', method='ward')
    clust.hierarchical(metric='correlation', method='complete')
    clust.hierarchical(metric='cityblock', method='average')
    clust.hierarchical(metric='euclidean', method='complete')



if __name__=="__main__":
    test_clustering()
