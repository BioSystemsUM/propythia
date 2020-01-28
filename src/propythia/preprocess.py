# -*- coding: utf-8 -*-
"""
##############################################################################

A class used for utility functions and transformer classes to change raw feature
vectors into a representation that is more suitable for the downstream estimators
All the functions are imported from sklearn.preprocessing

Authors: Ana Marta Sequeira

Date: 05/2019

Email:

##############################################################################
"""
from sklearn.preprocessing import *
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np


class Preprocess:
    """
    The Preprocess class aims to transform the feature vectors into a representation suitable for downstream estimators.
    Clean the dataset from features redundants and deal with Nans.
    All the functions are imported from the module sklearn.preprocessing
    """

    def __init__(self):
        """	constructor """

    def missing_data(self, data):
        """
        check nans. if yes: returns a error message
        :param data: dataframe
        :return: error message or 0 nans error
        """

        if len(data.columns[data.isnull().any()]) != 0:
            print('warning, miss values. should drop columns or fill this values')
        else:
            print('0 nans')
        # print(dataset.columns[dataset.isnull().any()])
        # print(dataset.isnull().values.any())

    def remove_columns_all_zeros(self,data, columns_names=False):
        """
        Removes columns that have all values as zero.
        :param data: dataframe
        :param columns_names: if True retrieves the names of columns with only zeros
        :return: dataset without columns with all values=zero
        """
        if columns_names:
            return data.loc[:, (~(data == 0).all())], data.columns[(data == 0).all()]
        else:
            return data.loc[:, (~(data == 0).all())],None

    def remove_duplicate_columns(self, data, columns_names=False):
        """
        Removes columns duplicated.
        :param data: dataframe
        :param columns_names: if True retrieves the names of columns duplicates
        :return: dataset without duplicated columns
        """

        if columns_names:
            return data.loc[:, ~data.T.duplicated(keep='first')], data.columns[
                (data.T.duplicated(keep='first'))]
        else:
            return data.loc[:, ~data.T.duplicated(keep='first')], None
        # return dataset.T.drop_duplicates().T

    def remove_low_variance(self, data, threshold=0, standard=True, columns_names=False):
        """
        Based on scikit learn
        VarianceThreshold is a simple baseline approach to feature selection.
        It removes all features whose variance doesn’t meet some threshold
        :param data: dataframe
        :param threshold: value. The threshold of variance to drop columns (eg 0.8)
        :param standard: if , in the case of threshold >0, the user wants to standardize features before apply variance
        threshold. minmaxscaler will be applied
        :param columns_names:
        :return: dataset without low variance columns (not scaled)
        """

        data = data
        if standard:
            scaler = MinMaxScaler()
            scaler.fit(data)
            scaler.transform(data)
            sel = VarianceThreshold(threshold)
            transf = sel.fit_transform(data)

            # original dataset without columns
            column_selected = sel.get_support(indices=True)

            data = data.iloc[:, column_selected]
            columns_excluded = list(set(data.columns) - set(data.columns))
            # print(data.head)
            # print(data.describe)

            if columns_names:
                return data, columns_excluded
            else:
                return data

        else:
            sel = VarianceThreshold(threshold)
            transf = sel.fit_transform(data)
            # original dataset without columns
            column_selected = sel.get_support(indices=True)
            data = data.iloc[:, column_selected]
            columns_excluded = list(set(data.columns) - set(data.columns))

            if columns_names:
                return transf, columns_excluded
            else:
                return transf

    def preprocess(self, data, columns_names=True, threshold=0, standard=True):
        """
        Removes columns that have all values as zero, duplicated and low variance columns

        :param data: dataset of input
        :param columns_names: if True retrieves the names of columns deleted
        :param threshold: the threshold of variance to drop columns
        :param standard: if true minmaxscaler will be applied
        :return: dataset original without deleted columns
        """

        dataset_zero, column_zero = self.remove_columns_all_zeros(data, columns_names=columns_names)
        dataset_without_duplicate, column_duplicated = self.remove_duplicate_columns(dataset_zero, columns_names=columns_names)
        dataset_clean, column_not_variable = self.remove_low_variance(dataset_without_duplicate, threshold=threshold,
                                                                      standard=standard, columns_names=columns_names)
        columns_deleted = (column_zero.append(column_duplicated)).append(column_not_variable)
        if columns_names:
            return dataset_clean, columns_deleted
        else:
            return dataset_clean

