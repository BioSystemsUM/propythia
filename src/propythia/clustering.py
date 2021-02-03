"""
##############################################################################

File containing a class intend to facilitate clustering analysis.
The functions are based on the package scikit learn.

Authors: Ana Marta Sequeira

Date: 06/2019 altered 01/2021

Email:

##############################################################################
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from propythia.adjuv_functions.ml_deep.utils import timer


class Cluster:
    """
    The cluster class aims to perform and plot clustering analysis
    Based on scikit learn.
    """

    def __init__(self, fps_x, target=None, report_name=None):
        """
        init function. When the class is called a x dataset must be provided (with features or encodings).
        Target is optional and only needed f you want to validate your cluster.
        If given a report name,  the class will automatically generate report for this analysis.
        :param fps_x: dataset X_data
        :param target: column with class labels. None by default
        :param report_name: str. if given, the class will automatically retrieve a txt report with results from
        the functions called inside class.
        """

        self.X_data = fps_x
        self.Y_data = target
        self.y_kmeans = None
        self.clf = None
        self.y_labels = None
        self.centroids = None
        self.affinity_matrix = None
        self.report_name = report_name
        if self.report_name:
            self._report(str(self.report_name))

    def _report(self, info):
        filename = str(self.report_name)
        with open(filename, 'a+') as file:
            if isinstance(info, str):
                file.writelines(info)
            else:
                for l in info:
                    file.writelines('\n{}'.format(l))

    def _kmeans_table(self, rownames=('clusters',)):
        """
        Function to generate a cross table with results from Kmeans
        :param rownames: row names. ('clusters',) by default.
        :return: table
        """
        table = pd.crosstab(self.y_labels, columns=self.Y_data, rownames=rownames)
        return table

    def _kmeans_seaborn_table(self, path_save='', show=True):
        """
        Function to geenrate a seaborn graphic with table results from kmeans
        :param path_save:  path to save file generated. By default ''
        :param show: If to display or not the graphic. True by default.
        :return: seaborn plot with k means results
        """
        plt.clf()
        mat = confusion_matrix(self.Y_data, self.y_labels)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        # save and empty plt
        if path_save is not None:
            plt.savefig(fname=path_save)
        if show is True:
            plt.show()
        plt.clf()

    def clust_evaluation(self, y_data):
        """
        Function to evaluate clustering results on target data.
        :param y_data:  list of targets correspondent to fps_x. If not given, it will retrieve the target from init.
        :return: accuracy evaluation of the kmeans clustering prediction
        """
        if self.Y_data is None:
            self.Y_data = y_data

        accuracy = accuracy_score(self.Y_data, self.y_kmeans)
        print('accuracy', accuracy)
        # table
        table = self._kmeans_table()
        print(table)

        # fig table
        kmeans_table_path = str(self.report_name.split('.', 1)[0] + 'kmeans_table.png')
        self._kmeans_seaborn_table(path_save=kmeans_table_path, show=True)

        things_to_report = ['accuracy: {}'.format(accuracy), table]
        return things_to_report

    @timer
    def run_kmeans(self, max_iter=300, n_clusters=None, init='k-means++', random_state=42, **params):
        """
        Function that performs K means cluster.
        :param init: init function for kmeans . 'k-means++' by default.
        :param max_iter: number of max terations for cluster (300 by default)
        :param n_clusters: if None, it will define the number of clusters as the number of existing labels
        :return: cross table with counts for labels vs classification in clusters
        """

        if not n_clusters:
            n_clusters = len(np.unique(self.Y_data))

        saved_args = locals()

        self.clf = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, random_state=random_state, **params)
        self.clf.fit(self.X_data)
        self.y_labels = self.clf.labels_
        self.centroids = self.clf.cluster_centers_
        self.y_kmeans = self.clf.predict(self.X_data)

        if self.Y_data is not None:
            things_to_report = self.clust_evaluation(self.Y_data)
        else:
            s = 'no true labels / y is given to evaluate k means'
            things_to_report = [s]
            print(s)

        if self.report_name is not None:
            self._report([self.run_kmeans.__name__, saved_args, 'y_labels_\n{}'.format(self.y_labels),
                          'centroids_\n{}'.format(self.centroids),
                          things_to_report])

        return self.clf, self.y_labels, self.centroids

    @timer
    def run_minibatch_kmeans(self, max_iter=300, batch_size=100, n_clusters=None, init='k-means++', random_state=42,
                             **params):
        """
        Function that performs K means cluster using miini batches.
        :param init: init function for kmeans . 'k-means++' by default.
        :param max_iter: number of max terations for cluster (300 by default)
        :param n_clusters: if None, it will define the number of clusters as the number of existing labels
        :param batch_size: batch size to use in mini batch. 100 by default.
        :return: cross table with counts for labels vs classification in clusters
        """
        if not n_clusters:
            n_clusters = len(np.unique(self.Y_data))
        saved_args = locals()

        self.clf = MiniBatchKMeans(n_clusters=n_clusters, init=init, batch_size=batch_size, max_iter=max_iter,
                                   random_state=random_state, **params)
        self.clf.fit(self.X_data)
        self.y_labels = self.clf.labels_
        self.centroids = self.clf.cluster_centers_
        self.y_kmeans = self.clf.predict(self.X_data)

        if self.Y_data is not None:
            things_to_report = self.clust_evaluation(self.Y_data)
        else:
            s = 'no true labels / y is given to evaluate k means'
            things_to_report = [s]
            print(s)

        if self.report_name:
            self._report([self.run_minibatch_kmeans.__name__, saved_args, 'y_labels_\n{}'.format(self.y_labels),
                          'centroids_\n{}'.format(self.centroids),
                          things_to_report])

        return self.clf, self.y_labels, self.centroids

#######################################################################################################################################
    # multiprocessing
    # accept the other matrix before
    # put the dendogram outside this?
    @timer
    def hierarchical_dendogram(self, metric='correlation', method='complete', path_save='', show=True,
                               truncate_mode='level', p=3, **params):
        """
        Perform hierarchical clustering scipy matrices
        :param metric: distance metric to use in the case that y is a collection of observation vectors. eg. 'correlation', 'euclidean'
        see (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist)
        :param method: method to be used. exemples: 'complete', 'single', 'average', 'ward'
        see (https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)
        test if no labels work
        :param truncate_mode: 'level' by default.
        :return: dendogram of the clustered data The hierarchical clustering encoded as a linkage matrix.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
        https://scipy.github.io/devdocs/generated/scipy.cluster.hierarchy.dendrogram.html
        ** params for dendogram
        """
        saved_args = locals()
        plt.clf()
        Z = linkage(self.X_data, metric=metric, method=method)

        # The hierarchical clustering encoded as a linkage matrix.
        plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram metric: {}, method: {}'.format(metric, method))
        plt.xlabel('sample index')
        plt.ylabel('distance')
        labels = list(range(len(self.X_data)))
        label_colors = None
        if self.Y_data:
            labels = list(self.Y_data)
            color_labels = np.unique(self.Y_data)
            rgb_values = sns.color_palette("nipy_spectral", len(color_labels))  # 'set2'
            label_colors = dict(zip(color_labels, rgb_values))

        dendrogram(Z,
                   labels=labels,
                   truncate_mode=truncate_mode,
                   p=p,
                   leaf_rotation=90,  # rotate the x axis labels
                   leaf_font_size=8, **params)  # font size for the x axis labels
        ax = plt.gca()
        xlabels = ax.get_xmajorticklabels()
        for lbl in xlabels: lbl.set_color(label_colors[lbl.get_text()])
        if path_save is not None:
            plt.savefig(fname=path_save, bbox_inches='tight', pad_inches=0)
        if show is True:
            plt.show()
        plt.clf()
        if self.report_name:
            self._report([self.hierarchical_dendogram.__name__, saved_args,
                          'y_labels_\n{}'.format(self.y_labels)])
        return Z

    # do the same as the scipy but it is taking to long. because te matrix is a little bit different
    # it needs acessory functions to plot the dendogram
    # check if functions are ok and if are worth to keep
    # check if can be combine the fuctions to the graphic from scipy matrix, as it is the same dendogram
    # and use the same function to plot
    @timer
    # check if it can receive a linkage matrix
    def run_hierarchical(self, n_clusters=None, affinity='euclidean', linkage='ward', y_data=None, **params):
        """
        https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering

        :param n_clusters:
        :param affinity:
        :param linkage:
        :param path_save:
        :param show:
        :return:
        """
        # https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/
        if not n_clusters:
            n_clusters = len(np.unique(self.Y_data))
        saved_args = locals()
        cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage, **params)
        hmodel = cluster.fit_predict(self.X_data)
        self.y_labels = hmodel.labels_
        self.n_leaves = hmodel.n_leaves_
        self.n_components = hmodel.n_connected_components_
        self.hmodel = hmodel

        if self.Y_data or y_data is not None:
            things_to_report = self.clust_evaluation(y_data)
        else:
            s = 'no true labels / y is given to evaluate k means'
            things_to_report = [s]
            print(s)

        if self.report_name:
            self._report([self.run_hierarchical.__name__, saved_args, 'y_labels_\n{}'.format(self.y_labels),
                          'n_leaves_\n{}'.format(self.n_leaves),
                          'n_connected_components_\n{}'.format(self.n_components),
                          things_to_report])

        return hmodel, cluster, self.y_labels, self.n_leaves

    def scikit_dendrogram(self, model, **kwargs):
        # code from scikit learn
        # https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
        # model = agglomerative cluster.fit(x

        # Create linkage matrix and then plot the dendrogram
        # create the counts of samples under each node

        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                          counts]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)

    def plot_dendrogram(self, model=None, title='Hierarchical Clustering Dendrogram', truncate_mode='level', p=3,
                        path_save='', show=True):
        if model is None:
            model = self.hmodel
        plt.clf()
        plt.title(title)
        # plot the top three levels of the dendrogram
        self.scikit_dendrogram(model, truncate_mode=truncate_mode, p=p)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        if path_save is not None:
            plt.savefig(fname=path_save)
        if show is True:
            plt.show()
        plt.clf()

    # with parallel_backend('loki')
    # from sklearn.neighbors import kneighbors_graph
    # connectivity = kneighbors_graph(X, n_neighbors=10)
    # get a connectivity matrix before to do the agglomerative clustering first
    # https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html#sklearn.neighbors.kneighbors_graph
    # https://stackoverflow.com/questions/47321133/sklearn-hierarchical-agglomerative-clustering-using-similarity-matrix

    # check the nightly version of scikit learn. useful information all the pipeline. ROC curves automatic...
    # https://scikit-learn.org/dev/common_pitfalls.html
    # https://scikit-learn.org/dev/computing/parallelism.html#parallelism
    # check for parallel runs. understand
    # https://joblib.readthedocs.io/en/latest/parallel.html#joblib.parallel_backend
    # https://scikit-learn.org/dev/computing/parallelism.html
    # check these three clusters. not scalable for large volumes of data
    # @timer nt scalable for many many features but good try / validate with other datasets.
    # https://scikit-learn.org/stable/modules/clustering.html#dbscan
    # def spectral_clustering(self, n_clusters=None, affinity='rbf',
    #                         assign_labels='kmeans', **params): #affinity = rbf   nearest_neighbors
    #     """
    #     allow kmeans to discover non linear boundaries
    #     take off? did notuse because it is super slow or doesn thave memory
    #     :return:
    #     """
    #     if not n_clusters:
    #         n_clusters = len(np.unique(self.Y_data))
    #
    #     self.clf = SpectralClustering(n_clusters=n_clusters, affinity=affinity,
    #                                   assign_labels=assign_labels,**params )
    #     self.clf.fit(self.X_data)
    #     self.y_labels = self.clf.labels_
    #     self.affinity_matrix = self.clf.affinity_matrix_
    #
    #     self.y_kmeans = self.clf.predict(self.X_data)
    #     print('accuracy', accuracy_score(self.Y_data, self.y_kmeans))
    #
    #     # table
    #     table = self.kmeans_table()
    #     print(table)
    #     sb = self.kmeans_seaborn_table()
    #
    # @timer
    # def gaussianmm(self,n_components = None,covariance_type='full', random_state=42, **params):
    #     """
    #     CHECK WHAT MAKES SENSE
    #     think not scalable for large datasets. so test in other
    #     https://scikit-learn.org/stable/modules/mixture.html#mixture
    #     https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
    #     https://scikit-learn.org/stable/modules/mixture.html
    #     https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
    #     :return:
    #     """
    #     if n_components is None:
    #         if self.Y_data:
    #             n_components=len(np.unique(self.Y_data))
    #         else:
    #             n_components = 1
    #     gmm = mixture.GMM(n_components=n_components, covariance_type=covariance_type, random_state=random_state, **params)
    #
    #     # the number of components can be ptimze by decreasing the BIC, see scikit learn and other page. graphics to do the est n_compoennts
    #     gmm_fit = gmm.fit(self.X_data)
    #     labels = gmm.predict(self.X_data)
    #     probs = gmm.predict_proba(self.X_data)
    #     print(probs[:5].round(3))
    #
    # @timer
    # def bayesiangaussianmm(self,n_components = None,covariance_type='full', random_state=42, **params):
    #     """
    #     CHECK WHAT MAKES SENSE
    #     ssame. not scalable I think
    #     https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
    #     https://scikit-learn.org/stable/modules/mixture.html
    #     https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
    #     https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture
    #     this one can have less than n_compoennts. it makes aditional operations. so n_components is not the real number of components but the maximum
    #     :return:
    #     """
    #     if n_components is None:
    #         if self.Y_data:
    #             n_components=len(np.unique(self.Y_data))
    #         else:
    #             n_components = 1
    #     gmm = mixture.BayesianGaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state, **params)
    #     gmm_fit = gmm.fit(self.X_data)
    #     labels = gmm.predict(self.X_data)
    #     probs = gmm.predict_proba(self.X_data)
    #     print(probs[:5].round(3))
    # def dbscan(self):
    #     # The DBSCAN algorithm views clusters as areas of high density separated by areas of low density. Due to this
    #     # rather generic view, clusters found by DBSCAN can be any shape, as opposed to k-means which assumes that clusters
    #     # are convex shaped. The central component to the DBSCAN is the concept of core samples, which are samples that are
    #     # in areas of high density.
    #     clustering = OPTICS(min_samples=2,cluster_method='dbscan').fit(self.X_data)
    #     self.labels = clustering.labels_
    #     hierarchy = clustering.cluster_hierarchy_
    #     return clustering, self.labels

    # def kmeans_predict(self, x_test, output='add', max_iter=300, n_clusters=None):
    #     """
    #     Perform the kmeans to train data and predict the tests set
    #     :param output: 'add' or 'replace'.
    #     If add, the labels produced by clustering will be added as features
    #     If replace, labels produced will be replace the old labels
    #     :param max_iter: max number of iterations of cluster
    #     :param n_clusters: if None, it will define the number of clusters as the number of existing labels
    #     :return: values of X datasets altered (if add) or the Y datasets replaced
    #     TEST IT BECAUSE I MAKE CHANGES
    #     """
    #     if not n_clusters: n_clusters = len(np.unique(self.Y_data))
        # check if model built or no
        # if no
        #     clf = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
        #     clf.fit(self.X_data)
        #
        # if yes
        #     get the model
        #     predict the model to x_test
        #     y_labels_test = clf.predict(x_test)
        # get the labels of model (trained here or in class)
        # y_labels_train = clf.labels_
        # define what happens to outputs
        # if output == 'add':
        #     self.X_train.loc[:, 'km_clust'] = y_labels_train
        #     self.X_test.loc[:, 'km_clust'] = y_labels_test
        # elif output == 'replace':
        #     self.X_train = y_labels_train.loc[:, np.newaxis]
        #     self.X_test = y_labels_test.loc[:, np.newaxis]
        # else:
        #     raise ValueError('output should be add or replace')
        # return self

    # def classify(self, model=SVC(random_state=42)):
    #     """
    #     Function that fits the model in train datasets and predict on the tests dataset, returning the accuracy
    #     :param model: model to make prediction (SVC by default)
    #     :return: the accuracy of the prediction
    #     """
    #
    #     model.fit(self.X_train, self.y_train)
    #     y_pred = model.predict(self.X_test)
    #     print('Accuracy: {}'.format(accuracy_score(self.y_test, y_pred)))
    # take off dont know the sense of this here

# https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
# implement with tensorflow for GPU
# https://www.altoros.com/blog/using-k-means-clustering-in-tensorflow/
# https://www.commencis.com/thoughts/comparison-of-clustering-performance-for-both-cpu-and-gpu/ KMEANS TF
