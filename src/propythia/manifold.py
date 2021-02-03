# -*- coding: utf-8 -*-
"""
##############################################################################

File containing a class to do manifold learning.
Manifold learning is an approach to non-linear dimensionality reduction.
This class performs t-sne and UMAP.
For additional information please read:
    https://umap-learn.readthedocs.io/en/latest/
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
Authors: Ana Marta Sequeira

Date: 12/2020

Email:

##############################################################################
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import umap.plot
from sklearn.manifold import TSNE
from propythia.adjuv_functions.ml_deep.utils import timer
sns.set()


class Manifold:
    """
    Class to run manifold learning methods. Contrary to methods such as PCA, manifold preserve nonlinear relationships
    in the data. This class allows to explore t-sne: tool to visualize high-dimensional data.
    The class opens with x_data, a dataset containing the features or encodings. Classes of this representations may
    be given in the classes argument (optional). If wanted, already porjected/ embedding matrices
    may be given just to run plots.
    """
    def __init__(self, x_data, classes=None, X_embedded=None, projected = None):
        self.fps_x = x_data # array
        self.X_embedded = X_embedded
        self.classes = classes
        self.mapper = None
        self.umapembedding = None
        self.projected = projected
        self.labels = None

    @timer
    def run_tsne(self, n_components = 2, **params):
        """
        t-sne is tool to visualize high-dimensional data. It converts similarities between data points to joint
        probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the
        low-dimensional embedding and the high-dimensional data. t-SNE has a cost function that is not convex,
        i.e. with different initializations we can get different results.
        :param n_components: number components to projct t-sne. by default 2.
        :param **params: paameters for scikit learn t-sne
        :return:
        """
        X_embedded = TSNE(n_components= n_components,**params).fit_transform(self.fps_x)
        self.X_embedded = X_embedded
        self.projected = self.X_embedded
        self.embedding = X_embedded
        self.labels ='TSNE'
        return X_embedded

    @timer
    def run_umap(self,n_neighbors=20, min_dist=0.1, n_components=50,metric='correlation', target=None):
        """
        Function to run UMAP
        :param n_neighbors: how UMAP balances local versus global structure in the data
        low values of n_neighbors will force UMAP to concentrate on very local structure
        (potentially to the detriment of the big picture), while large values will push UMAP to look at larger
        eighborhoods of each point when estimating the manifold structure of the data, losing fine detail structure
        for the sake of getting the broader of the data. 20 by default
        :param min_dist: how tightly UMAP is allowed to pack points together. It, quite literally,
        provides the minimum distance apart that points are allowed to be in the low dimensional representation.
        This means that low values of min_dist will result in clumpier embeddings. This can be useful if you are
        interested in clustering, or in finer topological structure. Larger values of min_dist will prevent UMAP
        from packing points together and will focus on the preservation of the broad topological structure instead.
        0.1 by default
        :param n_components: components to consider. 50 by default.
        :param metric: default 'correlation'
        :param target: optional.
        More information on https://umap-learn.readthedocs.io/en/latest/parameters.html#n-neighbors
        :return:
        """

        reducer = umap.UMAP(n_neighbors = n_neighbors , min_dist=min_dist, n_components=n_components,
                            metric=metric)
        mapper = reducer.fit(self.fps_x)
        embedding = reducer.fit_transform(self.fps_x, y=target)
        self.projected = mapper.embedding_
        self.mapper=mapper
        self.embedding = embedding
        self.labels ='UMAP'
        return mapper, embedding

    def manifold_scatter_plot(self, target, dim1=0, dim2=1, title=None, show=True, path_save='manifold_scatter_plot.png'):
        """
        Function to retrieve scatter plot for 2 dimensions for both t-SNE and UMAP analysis.

        :param target:  list/array of targets of data.
        :param dim1: axis to consider to plot. by default the first: 0
        :param dim2: axis to consider to plot. by default the first: 1
        :param title: title of the generated plot . None by default
        :param show: If t display or not the scatter plot.
        :param path_save: path to save scatter plot. 'manifold_scatter_plot.png' by default
        :return: scatter plot of two components derived from manifold analysis, colored by target.
        """
        plt.clf()

        n_color=len(np.unique(target))
        # Get Unique ec
        color_labels = np.unique(target)

        # List of colors in the color palettes
        rgb_values = sns.color_palette("nipy_spectral", n_color) # 'set2'

        # Map ec to the colors
        color_map = dict(zip(color_labels, rgb_values))

        # Finally use the mapped values
        plt.scatter(self.projected[:, dim1], self.projected[:, dim2],
                    c=target.map(color_map), s=0.3)

        # create legend
        # classes  - colour labels
        # class_colours rgb values
        recs = []
        from matplotlib.lines import Line2D

        for i in range(0, n_color):
            recs.append(Line2D((0,0.75),(0,0), color=rgb_values[i], marker='o', linestyle=''))
            # recs.append(mpatches.Circle((0, 0), 0.5, fc=rgb_values[i]))

        n_col = int(n_color/23)+1

        lgd = plt.legend(recs, color_labels, bbox_to_anchor=(1,1), loc="upper left",
                         fontsize='xx-small', ncol = n_col, shadow=False, title='classes')
        # add labels and title
        plt.xlabel('dim 1')
        plt.ylabel('dim 2')
        if title is None:
            title='{} scatter plot of 2 dimensions'.format(self.labels)
        plt.title(title)

        # save and empty plt
        if path_save is not None:
            plt.savefig(fname=path_save,bbox_extra_artists=(lgd,), bbox_inches='tight', pad_inches=0)
        if show is True:
            plt.show()
        plt.clf()

    # def manifold_scatter_plot2(self, target, title=None, show=True, path_save='manifold_scatter_plot.png'):
    #     fig, ax = plt.subplots(1, figsize=(14, 10))
    #     plt.scatter(self.projected[:, 0], self.projected[:, 1], s=0.3, c=target, cmap='Spectral', alpha=1.0)
    #     plt.setp(ax, xticks=[], yticks=[])
    #     cbar = plt.colorbar(boundaries=np.arange(11)-0.5)
    #     cbar.set_ticks(np.arange(len(np.unique(target))))
    #     cbar.set_ticklabels(np.unique(target))
    #     plt.title('Fashion MNIST Embedded via UMAP')
    #     if title is None:
    #         title='{} scatter plot of 2 dimensions'.format(self.labels)
    #     plt.title(title)
    #
    #     # save and empty plt
    #     if path_save is not None:
    #         plt.savefig(fname=path_save, bbox_inches='tight', pad_inches=0)
    #     if show is True:
    #         plt.show()
    #     plt.clf()






# https://arxiv.org/pdf/1802.03426.pdf
# https://umap-learn.readthedocs.io/en/latest/parameters.html


 # much data perplixity between 5-50 perplixity , very large try 100
# https://blog.clairvoyantsoft.com/mlmuse-visualisation-of-high-dimensional-data-using-t-sne-ac6264316d7f
# https://distill.pub/2016/misread-tsne/
# use pca or svd before of data to high dimensional
# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
# https://stats.stackexchange.com/questions/238538/are-there-cases-where-pca-is-more-suitable-than-t-sne
# plot embeddings / word embeddings
# https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b



# to visualize embeddings
# https://umap-learn.readthedocs.io/en/latest/embedding_space.html in space 2 D or for example i a sphere and can in jupyter interactive maps
# https://www.kaggle.com/colinmorris/visualizing-embeddings-with-t-sne

