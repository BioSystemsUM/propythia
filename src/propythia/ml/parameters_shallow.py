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
    param = {'svc':
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

             'linear_svc':
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

             # todo have added !!!!!!!
             'svr':
                 {'param_grid':
                      [{'clf__kernel': [ 'poly', 'rbf', 'sigmoid'], #'linear',
                        'clf__C': [ 0.1, 1.0, 10],
                        'clf__epsilon': [0.01, 0.1, 0.2, 1.0, 5.0],  # acceptable error margin(ϵ)
                        }],
                  'distribution':
                      [{'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                        'clf__C': loguniform(1e-2, 1e1),
                        'clf__epsilon': loguniform(1e-2, 1e2)
                        }]
                  },
             # 'linear_svr':
             #     {'param_grid':
             #          [{'clf__C': [0.01, 0.1, 1.0, 10],
             #            'clf__epsilon': [0.01, 0.1, 0.2, 1.0],  # acceptable error margin(ϵ)
             #            'clf__loss':['epsilon_insensitive', 'squared_epsilon_insensitive'],
             #            'clf__intercept_scaling':[0.1,1.0,5.0,10]
             #            }],
             #      'distribution':
             #          [{'clf__C': loguniform(1e-2, 1e1),
             #            'clf__epsilon': loguniform(1e-2, 1e2),
             #            'clf__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
             #            'clf__intercept_scaling':loguniform(1e-1, 1e1),
             #            }]
             #      },
             'linear_svr':
                 {'param_grid':
                      [{'clf__C': [0.01, 0.1, 1.0],
                        'clf__epsilon': [0.01, 0.1, 1.0],  # acceptable error margin(ϵ)
                        'clf__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
                        'clf__intercept_scaling': [0.1, 1.0]
                        }],
                  'distribution':
                      [{'clf__C': loguniform(1e-2, 1e1),
                        'clf__epsilon': loguniform(1e-2, 1e2),
                        'clf__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
                        'clf__intercept_scaling': loguniform(1e-1, 1e1),
                        }]
                  },
             # todo add min samples leaf/ split and like that?

             'rfr':
                 {'param_grid':
                      [{'clf__n_estimators': [10, 100, 300, 500],
                        'clf__max_features': ['auto', 'sqrt', 'log2'],
                        }],
                  'distribution':
                      [{'clf__n_estimators': randint(1e1, 5e3),
                        'clf__max_features': ['auto', 'sqrt', 'log2'],
                        }]
                  },
             'gboostingr':
                 {'param_grid':
                      [{'clf__n_estimators': [100, 400, 300, 500],
                        'clf__learning_rate': [0.05, 0.1, 0.5],
                        'clf__max_features': ['auto', 'sqrt', 'log2'],
                        }],
                  'distribution':
                      [{'clf__n_estimators': randint(1e2, 5e3),
                        'clf__learning_rate': loguniform(0.05, 0.5),
                        'clf__max_features': ['auto', 'sqrt', 'log2'],
                        }]
                  },
             'histgboostingr':
                 {'param_grid':
                      [{'clf__max_iter': [50, 100, 200, 300, 500],
                        'clf__learning_rate': [0.05, 0.1, 0.5],
                        'clf__loss': ['squared_error', 'absolute_error', 'poisson'],
                        }],
                  'distribution':
                      [{'clf__n_estimators': randint(5e1, 5e3),
                        'clf__learning_rate': loguniform(0.05, 0.5),
                        'clf__loss': ['squared_error', 'absolute_error', 'poisson'],
                        }]
                  },

             'adaboostr':
                 {'param_grid':
                      [{'clf__n_estimators': [500, 300, 100],
                        'clf__learning_rate': [0.5, 1, 1.5],
                        }],
                  'distribution':
                      [{'clf__n_estimators': randint(1e1, 5e3),
                        'clf__learning_rate': loguniform(0.05, 2),
                        }]
                  },

             'tweedier':
                 {'param_grid':
                      [{'clf__power': [1,2,3],#, 2, 3], # 0 is normal. other better models and distribution is not above zero 2 and 3 do not include the zero on the interval
                        # underlying target distribution according to the following table 0 normal 1 poisson 2 gamma 3 inverse gaussian
                        'clf__alpha': [0, 0.5, 1],
                        # regularization strength. alpha = 0 is equivalent to unpenalized GLMs
                        }],
                  'distribution':
                      [{'clf__power': [1, 2, 3],
                        'clf__max_features': loguniform(0, 1),
                        }]
                  },
             'knr':
                 {'param_grid':
                      [{'clf__n_neighbors': [2, 5, 10, 15],
                        'clf__weights': ['uniform', 'distance'],
                        'clf__leaf_size': [15, 30, 50],
                        }],
                  'distribution':
                      [{'clf__n_neighbors': randint(2e1, 15),
                        'clf__weights': ['uniform', 'distance'],
                        'clf__leaf_size':randint(15, 50),
                        }]
                  },

    # melhorar isto ou nem vale a pena estar aqui
             'gpr':
                 {'param_grid':
                      [{'clf__alpha': [1e-10,1e-5, 1e-3] # 1e-10 default

                        }],
                  'distribution':
                      [{'clf__alpha': loguniform(1e-11, 1e-5)
                        }]
                  },
             'sgdr':
                 {'param_grid':
                      [{'clf__loss': ['squared_loss', 'huber', 'squared_epsilon_insensitive'], # todo change squared loss to squared_error
                        'clf__alpha': [0.001, 0.0001, 0.00001], #default 0.0001
                        'clf__learning_rate': ['adaptive','optimal','invscaling'],  # acceptable error margin(ϵ)
    # early_stopping True
                        }],
                  'distribution':
                      [{'clf__loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                        'clf___alpha': loguniform(1e-5, 1e-2),
                        'clf__learning_rate':  ['adaptive','optimal','invscaling'],
                        }]
                  },

             }
    return param
