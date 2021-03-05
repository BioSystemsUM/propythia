"""
##############################################################################

File containing tests functions to check if all functions from manifold module are properly working

Authors: Ana Marta Sequeira

Date: 01/2021

Email:

##############################################################################
"""
from propythia.manifold import Manifold
import pandas as pd

def test_manifold():

    # read and define dataset
    dataset = pd.read_csv(r'datasets/dataset1_test_clean.csv', delimiter=',', encoding='latin-1')
    labels = dataset['labels']
    dataset = dataset.loc[:, dataset.columns != 'labels']

    # createObject
    mf = Manifold(x_data=dataset, classes=labels, X_embedded=None, projected = None)
    # run tsne
    tsne = mf.run_tsne(n_components = 2)
    mf.manifold_scatter_plot(target = labels, dim1=0, dim2=1, title='tsne plot', show=True, path_save=None)

    #run umap
    mf = Manifold(x_data=dataset, classes=labels, X_embedded=None, projected = None)
    mapper, embedding = mf.run_umap(n_neighbors=20, min_dist=0.1, n_components=50,metric='correlation', target=None)
    mf.manifold_scatter_plot(target = labels, dim1=0, dim2=1, title='umap plot', show=True, path_save=None)


if __name__ == "__main__":
    test_manifold()