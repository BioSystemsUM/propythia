"""
##############################################################################

File containing tests functions to check if all functions from preprocess module are properly working

Authors: Ana Marta Sequeira

Date: 06/2019

Email:

##############################################################################
"""
from propythia.preprocess import Preprocess
import pandas as pd


def test_preprocess():
    dataset = pd.read_csv(r'datasets/dataset_test.csv', delimiter=',', encoding='latin-1')
    # print(dataset.describe())
    # print(dataset.shape)

    # separate labels
    labels = dataset['labels']
    dataset = dataset.loc[:, dataset.columns != 'labels']

    # Create Preprocess object
    prepro = Preprocess()

    # CHECK IF NAN
    prepro.missing_data(dataset)

    dataset_zero, colum_Zero = prepro.remove_columns_all_zeros(dataset, True)  # remove zero columns
    # print(colum_Zero)
    # print(dataset_zero.shape)
    dataset_without_duplicate, column_duplicated = prepro.remove_duplicate_columns(dataset_zero,
                                                                                   True)  # DUPLICATED COLUMNS
    # print(column_duplicated)
    # print(dataset_without_duplicate.shape)
    # REMOVE ZERO VARIANCE COLUMNS
    dataset_clean, column_not_variable = prepro.remove_low_variance(dataset_without_duplicate, standard=True,
                                                                    columns_names=True)
    # print(column_not_variable)
    # print(dataset_clean.shape)

    ######OR

    dataset_clean, columns_deleted = prepro.preprocess(dataset, columns_names=True, threshold=0, standard=True)

    # put labels back
    dataset_clean['labels'] = labels
    print(dataset_clean.shape)


if __name__ == "__main__":
    test_preprocess()