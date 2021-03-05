# -*- coding: utf-8 -*-
"""
##############################################################################

File containing tests functions to check if all functions are properly working

Authors: Ana Marta Sequeira

Date: 06/2019

Email:

##############################################################################
"""
import timeit

from tests.test_sequence import test_sequence
from tests.test_descriptors import test_descriptors
from tests.test_preprocess import test_preprocess
from tests.test_clustering import test_clustering
from tests.test_feature_selection import test_feature_selection
from tests.test_manifold import test_manifold
from tests.test_linear_dim_reduction import test_linear_dim_reduction
from test_shallow_ml import test_shallow_ml
from test_deep_ml import test_deep_ml

if __name__ == "__main__":
    start = timeit.default_timer()
    test_sequence()
    test_descriptors()
    test_preprocess()
    test_linear_dim_reduction()
    test_manifold()
    test_feature_selection()
    test_clustering()
    test_shallow_ml()
    test_deep_ml()
    stop = timeit.default_timer()
    print('Time: ', stop - start, ((stop - start) / 60))