"""
##############################################################################

File with useful functions to plot curves to better understanding of ML and DL models (udes in these classes)
Authors: Ana Marta Sequeira

Date:12/2020

Email:

##############################################################################
"""

import sys

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
from numpy import interp
from itertools import cycle
from sklearn.metrics import *
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.model_selection import validation_curve
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier



warnings.filterwarnings("ignore")
# Seed for randomization. Set to some definite integer for debugging and set to None for production
seed = None


def plot_summary_accuracy(model, path_save=None, show=True):
    """
    Function to plot training and validation accuracy
    :param model:
    :param path_save:
    :param show:
    :return:
    """
    dnn = model
    plt.clf()
    # summarize history for accuracy
    plt.plot(dnn.history['accuracy'])
    plt.plot(dnn.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if path_save is not None:
        plt.savefig(fname=path_save)
    if show is True:
        plt.show(block=True)
    plt.clf()


def plot_summary_loss(model, path_save=None, show=True):
    """
    Function to plot validationa nd training loss
    :param model:
    :param path_save:
    :param show:
    :return:
    """
    dnn = model
    # summarize history for loss
    plt.clf()
    plt.plot(dnn.history['loss'])
    plt.plot(dnn.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if path_save is not None:
        plt.savefig(fname=path_save)
    if show is True:
        plt.show(block=True)
    plt.clf()

def plot_roc_curve(classifier, final_units, x_test, y_test, x_train, y_train, ylim=(0.0, 1.00), xlim=(0.0, 1.0),
                   title='Receiver operating characteristic (ROC) curve',
                   path_save='plot_roc_curve', show=False, batch_size=None):
    """
    Function to plot a ROC curve
     On the y axis, true positive rate and false positive rate on the X axis.
     The top left corner of the plot is the 'ideal' point - a false positive rate of zero, and a true positive rate of one,
     meaning a larger area under the curve (AUC) is usually better.
    :param classifier: {classifier instance} pre-trained classifier used for predictions.
    :param x_test: {array} descriptor values of the tests data.
    :param y_test:{array} true class values of the tests data.
    :param ylim: y-axis limits
    :param xlim: x- axis limits
    :param title: title of plot
    :return: plot ROC curve
    Needs classifier with probability
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    """
    lw = 2
    # binary
    if final_units < 2:
        y_score = classifier.predict(x_test, batch_size=batch_size)
        # y_score = classifier.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        print(roc_auc)
        plt.show()
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right", )
        if path_save is not None:
            plt.savefig(fname=path_save)
        if show is True:
            plt.show()
        plt.clf()

    # multiclass/ multilabel
    elif final_units > 2:
        # Binarize the output
        classe = np.unique(y_train)
        y_train = label_binarize(y_train, classes=classe)
        n_classes = y_train.shape[1]
        y_test = label_binarize(y_test, classes=classe)

        # Learn to predict each class against the other
        estimator = OneVsRestClassifier(classifier)
        y_score = estimator.fit(x_train, y_train).predict_proba(x_test, batch_size=batch_size)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        # Get Unique ec
        color_labels = np.unique(y_train)

        # List of colors in the color palettes
        rgb_values = sns.color_palette("nipy_spectral", final_units)  # 'set2'
        # Map ec to the colors
        for i, color in zip(range(final_units), rgb_values):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right", fontsize='xx-small')
        if path_save is not None:
            plt.savefig(fname=path_save)
        if show is True:
            plt.show()
        plt.clf()


def plot_precision_recall_curve(y, y_pred, n_classes, show=False, path_save=''):
    """
    Performs plot precision-recall curve
    :param y: Series or Dataframe of true labels
    :param y_pred: Series or Dataframe of predicted labels
    :return: None
    """

    if n_classes < 2:
        plt.clf()
        plt.figure()
        precision, recall, thresholds = precision_recall_curve(y, y_pred)
        plt.plot(recall, precision)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")

    else:
        # For each class
        # tem q estar ONE vs ONE classifier o y_pred tem de estar com os para shallow machine learning
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y[:, i], y_pred[:, i])
            average_precision[i] = average_precision_score(y[:, i], y_pred[:, i])
        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(y.ravel(),
                                                                        y_pred.ravel())
        average_precision["micro"] = average_precision_score(y, y_pred, average="micro")
        print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

        # setup plot details
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
        plt.figure(figsize=(7, 8))
        f_scores = np.linspace(0.2, 0.8, num=int(n_classes + 1))
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

        lines.append(l)
        labels.append('iso-f1 curves')
        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('micro-average Precision-recall (area = {0:0.2f})'
                      ''.format(average_precision["micro"]))

        for i, color in zip(range(n_classes), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                          ''.format(i, average_precision[i]))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

    if path_save is not None:
        plt.savefig(fname=path_save)
    if show is True:
        plt.show()
    plt.clf()


def plot_validation_curve(classifier, x_train, y_train, param_name, param_range,
                          cv=5,
                          score=make_scorer(matthews_corrcoef), title="Validation Curve",
                          xlab="parameter range", ylab="MCC", n_jobs=1, show=False, path_save='plot_validation_curve',
                          **params):
    """This function plots a cross-validation curve for the specified classifier on all tested parameters given in the
    option 'param_range'.

    :param classifier: {classifier instance} classifier or validation curve (e.g. sklearn.svm.SVC).
    :param x_train: {array} descriptor values for training data.
    :param y_train: {array} class values for training data.
    :param param_name: {string} parameter to assess in the validation curve plot. For example,
    For SVM,
        "clf__C" (C parameter), "clf__gamma" (gamma parameter).
    For Random Forest,
        "clf__n_estimators" (number of trees),"clf__max_depth" (max num of branches per tree, "clf__min_samples_split" (min number of samples required to
        split an internal tree node), "clf__min_samples_leaf" (min number of samples in newly created leaf).
    :param param_range: {list} parameter range for the validation curve.
    :param cv: {int} number of folds for cross-validation.
    :param score: {metrics instance} scoring function built from make_scorer() or a predefined value in string form
        `sklearn.model_evaluation.scoring-parameter
        <http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_.
    :param title: {str} graph title
    :param xlab: {str} x axis label.
    :param ylab: {str} y axis label.
    :param n_jobs: {int} number of parallel jobs to use for calculation. if ``-1``, all available cores are used.
    :param filename: {str} if filename given the figure is stored in the specified path.
    :return: plot of the validation curve.

    Based on a function from moodlamp
    Müller A. T. et al. (2017) modlAMP: Python for anitmicrobial peptides, Bioinformatics 33, (17), 2753-2755,
    DOI:10.1093/bioinformatics/btx285
    """
    # just goes with numeric parameters not with string or tuple ones. problem in the axis the plotting semilogx and fill between, but actually does notmake sense plot not numeric values
    train_scores, test_scores = validation_curve(classifier, x_train, y_train, param_name, param_range,
                                                 cv=cv, scoring=score, n_jobs=n_jobs, **params)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # plotting
    plt.clf()
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.ylim(0.0, 1.1)

    plt.semilogx(param_range, train_mean, label="Training score", color="b")
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color="b")
    plt.semilogx(param_range, test_mean, label="Cross-validation score", color="g")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    if path_save is not None:
        plt.savefig(fname=path_save)
    if show is True:
        plt.show()
    plt.clf()


def plot_learning_curve(classifier, x_train, y_train, title='Learning curve', ylim=None,
                        cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5),
                        path_save='plot_learning_curve', show=False, scalability=True, performance=True):
    """
    Plot a learning curve to determine cross validated training and tests scores for different training set sizes
    :param estimator: classifier/ model to use
    :param title: title of the plot
    :param ylim:
    :param cv: cross validation to use
    :param n_jobs:  number of parallel jobs to use for calculation. if ``-1``, all available cores are used.
    :param train_sizes: train sizes to tests
    :return: graphic representing learning curves, numbers of trainig examples, scores on training sets, and scores on tests set
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    """

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(classifier, x_train, y_train, cv=cv,
                                                                          n_jobs=n_jobs, train_sizes=train_sizes,
                                                                          return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # learning curve
    plt.clf()
    plt.figure()
    plt.grid(b=True, which='both')
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    if path_save is not None:
        plt.savefig(fname=str(path_save + 'learning_curve.png'))
    if show is True:
        plt.show()
    plt.clf()

    if scalability:
        # Plot n_samples vs fit_times
        plt.grid(b=True, which='both')
        plt.plot(train_sizes, fit_times_mean, 'o-')
        plt.fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
        plt.xlabel("Training examples")
        plt.ylabel("fit_times")
        plt.title("Scalability of the model")
        if path_save is not None:
            plt.savefig(fname=str(path_save + 'scalability_model.png'))
        if show is True:
            plt.show()
        plt.clf()

    if performance:
        # Plot fit_time vs score
        plt.grid(b=True, which='both')
        plt.plot(fit_times_mean, test_scores_mean, 'o-')
        plt.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
        plt.xlabel("fit_times")
        plt.ylabel("Score")
        plt.title("Performance of the model")
        if path_save is not None:
            plt.savefig(fname=str(path_save + 'performance_model.png'))
        if show is True:
            plt.show()
        plt.clf()

    # todo check if it is better way to do this. option of the plots. arrange the model names. in ones it gets the model names and the others dont
    return

    #
    #
    #
    # @staticmethod
    # def precision_recall_curve_binary(y, y_pred):
    #     """
    #     Performs plot precision-recall curve
    #     :param y: Series or Dataframe of true labels
    #     :param y_pred: Series or Dataframe of predicted labels
    #     :return: None
    #     """
    #     plt.figure()
    #     precision, recall, thresholds = precision_recall_curve(y, y_pred)
    #     plt.plot(recall, precision)
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.0])
    #     plt.xlabel("Recall")
    #     plt.ylabel("Precision")
    #     plt.title("Precision-Recall Curve")
    #     plt.show()
    #     plt.close()
    #
    # @staticmethod
    # def precision_recall_curve_multiclass(y, y_pred, n_classes):
    #     # For each class
    #     # tem q estar ONE vs ONE classifier o y_pred tem de estar com os paara shallow machine learning
    #     precision = dict()
    #     recall = dict()
    #     average_precision = dict()
    #     for i in range(n_classes):
    #         precision[i], recall[i], _ = precision_recall_curve(y[:, i], y_pred[:, i])
    #         average_precision[i] = average_precision_score(y[:, i], y_pred[:, i])
    #     # A "micro-average": quantifying score on all classes jointly
    #     precision["micro"], recall["micro"], _ = precision_recall_curve(y.ravel(),
    #                                                                     y_pred.ravel())
    #     average_precision["micro"] = average_precision_score(y, y_pred, average="micro")
    #     print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
    #
    #     from itertools import cycle
    #     # setup plot details
    #     colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    #     plt.figure(figsize=(7, 8))
    #     f_scores = np.linspace(0.2, 0.8, num=int(n_classes + 1))
    #     lines = []
    #     labels = []
    #     for f_score in f_scores:
    #         x = np.linspace(0.01, 1)
    #         y = f_score * x / (2 * x - f_score)
    #         l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    #         plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    #
    #     lines.append(l)
    #     labels.append('iso-f1 curves')
    #     l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    #     lines.append(l)
    #     labels.append('micro-average Precision-recall (area = {0:0.2f})'
    #                   ''.format(average_precision["micro"]))
    #
    #     for i, color in zip(range(n_classes), colors):
    #         l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    #         lines.append(l)
    #         labels.append('Precision-recall for class {0} (area = {1:0.2f})'
    #                       ''.format(i, average_precision[i]))
    #
    #     fig = plt.gcf()
    #     fig.subplots_adjust(bottom=0.25)
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.title('Extension of Precision-Recall curve to multi-class')
    #     plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    #     plt.show()
