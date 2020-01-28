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
from tests.test_feature_reduction import test_feature_reduction
from tests.test_machine_learning import test_machine_learning

if __name__ == "__main__":
    start = timeit.default_timer()
    test_sequence()
    test_descriptors()
    test_preprocess()
    test_feature_reduction()
    test_feature_selection()
    test_clustering()
    test_machine_learning()
    stop = timeit.default_timer()
    print('Time: ', stop - start, ((stop - start) / 60))