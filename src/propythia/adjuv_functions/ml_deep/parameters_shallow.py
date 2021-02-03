# -*- coding: utf-8 -*-
"""
##############################################################################

Default parameters grid for GridSearch (param_grid) and RandomizedSearch (distribution) using shallow models
Model available: 'svm', 'knn', 'sgd', 'lr','rf', 'gnb', 'nn','gboosting'
The values for grid are stored in dictionary and can be accessed using
param = param_shallow()
param_grid = param[model.lower()]['param_grid']
param_grid = param[model.lower()]['distribution']
Authors: Ana Marta Sequeira

Date:12/2020

Email:

##############################################################################
"""
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform, truncnorm, randint


# https://blog.usejournal.com/a-comparison-of-grid-search-and-randomized-search-using-scikit-learn-29823179bc85
# https://blog.usejournal.com/a-comparison-of-grid-search-and-randomized-search-using-scikit-learn-29823179bc85
# https://scikit-learn.org/stable/modules/grid_search.html#grid-search
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py

# In contrast to GridSearchCV, not all parameter values are tried out, but rather a fixed number of parameter settings
# is sampled from the specified distributions. The number of parameter settings that are tried is given by n_iter.
# If all parameters are presented as a list, sampling without replacement is performed. If at least one parameter
# is given as a distribution, sampling with replacement is used.
# It is highly recommended to use continuous distributions for continuous parameters

# For continuous parameters, such as C above, it is important to specify a continuous distribution to take full
# advantage of the randomization. This way, increasing n_iter will always lead to a finer search./

# The randomized search and the grid search explore exactly the same space of parameters.
# The result in parameter settings is quite similar, while the run time for randomized search is drastically lower.

# Tuning the hyper-parameters of an estimator
# A search consists of:
# an estimator (regressor or classifier such as sklearn.svm.SVC()),
# a parameter space;
# a method for searching or sampling candidates;
# a cross-validation scheme; and
# a score function.

# GridSearchCV exhaustively considers all parameter combinations
# RandomizedSearchCV can sample a given number of candidates from a parameter space with a specified distribution.

# put early stopping in grid searchs? sgd, Gboosting, nn, RF


def param_shallow():
    param = {'svm':
                 {'param_grid':
                      [{'clf__C': [0.01, 0.1, 1.0, 10],
                        'clf__kernel': ['linear']},
                       {'clf__C': [0.01, 0.1, 1.0, 10],
                        'clf__kernel': ['rbf'],
                        'clf__gamma': ['scale', 0.001, 0.0001]}],
                  'distribution':
                      [{'clf__C': loguniform(1e-2, 1e1),
                        'clf__kernel': ['linear']},
                       {'clf__C': loguniform(1e-2, 1e1),
                        'clf__gamma': loguniform(1e-4, 1e1),  # np.power(10, np.arange(-4, 1, dtype=float)),
                        'clf__kernel': ['rbf']}]
                  },

             'linear_svm':
                 {'param_grid':
                      [{'clf__C': [0.01, 0.1, 1.0, 10]}],
                  'distribution':
                      [{'clf__C': loguniform(1e-4, 1e1)}]
                  },

             'rf':
                 {'param_grid':
                      [{'clf__n_estimators': [10, 100, 500],
                        'clf__max_features': ['sqrt', 'log2'],
                        'clf__bootstrap': [True],
                        'clf__criterion': ["gini"]}],
                  'distribution':
                      [{'clf__n_estimators': randint(1e1, 1e3),
                        'clf__max_features': ['sqrt', 'log2'],
                        'clf__bootstrap': [True],
                        'clf__criterion': ["gini"]}]
                  },

             'gboosting':
                 {'param_grid':
                      [{'clf__n_estimators': [10, 100, 500],
                        'clf__max_depth': [1, 3, 5, 10],
                        'clf__max_features': [0.6, 0.9]}],
                  'distribution':
                      [{'clf__learning_rate': loguniform(1e-3, 1e0),
                        'clf__n_estimators': randint(1e1, 1e3),
                        'clf__max_depth': randint(1e0, 2e1),
                        'clf__max_features': loguniform(5e-1, 1)}]
                  },

             'knn':
                 {'param_grid': [{'clf__n_neighbors': [2, 5, 10, 15],
                                  'clf__weights': ['uniform', 'distance'],
                                  'clf__leaf_size': [15, 30, 60]}],
                  'distribution':
                      [{'clf__n_neighbors': randint(2e0, 3e1),
                        'clf__weights': ['uniform', 'distance'],
                        'clf__leaf_size': loguniform(1e1, 1e2)}]
                  },

             'sgd':
                 {'param_grid':
                      [{'clf__loss': ['hinge', 'log', 'modified_huber', 'perceptron'],
                        'clf__alpha': [0.00001, 0.0001, 0.001, 0.01],
                        'clf__early_stopping': [True],
                        'clf__validation_fraction': [0.2],
                        'clf__n_iter_no_change': [30]}],
                  'distribution':
                      [{'clf__loss': ['hinge', 'log', 'modified_huber', 'perceptron'],
                        'clf__alpha': loguniform(1e-5, 1e-2),
                        'clf__early_stopping': [True],
                        'clf__validation_fraction': [0.2],
                        'clf__n_iter_no_change': [50]}]
                  },

             'lr':
                 {'param_grid':
                      [{'clf__C': [0.01, 0.1, 1.0, 10.0],
                        'clf__solver': ['liblinear', 'lbfgs', 'sag']}],
                  'distribution':
                      [{'clf__C': loguniform(1e-2, 1e1),
                        'clf__solver': ['liblinear', 'lbfgs', 'sag']}]
                  },

             'gnb':
                 {'param_grid':
                      [{'clf__var_smoothing': [1e-12, 1e-9, 1e-6]}],
                  'distribution':
                      [{'clf__var_smoothing': loguniform(1e-12, 1e-6)}]
                  },

             'nn':
                 {'param_grid':
                      [{'clf__activation': ['logistic', 'tanh', 'relu'],
                        'clf__alpha': [0.00001, 0.0001, 0.001],
                        'clf__learning_rate_init': [0.0001, 0.001, 0.01],
                        'clf__early_stopping': [True],
                        'clf__validation_fraction': [0.2],
                        'clf__n_iter_no_change': [50]}],
                  'distribution':
                      [{'clf__activation': ['logistic', 'tanh', 'relu'],
                        'clf__alpha': loguniform(1e-5, 1e-3),
                        'clf__learning_rate_init': loguniform(1e-4, 1e-2),
                        'clf__early_stopping': [True],
                        'clf__validation_fraction': [0.2],
                        'clf__n_iter_no_change': [50]}]  # put a quarter of the iterations by default
                  },

             }
    return param
