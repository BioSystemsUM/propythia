"""
##############################################################################

Default parameters grid for GridSearch (param_grid) and RandomizedSearch (distribution) using DL models
The values for grid are stored in dictionary and can be accessed using
param = param_shallow()
param_grid = param[model.lower()]['param_grid']
param_grid = param[model.lower()]['distribution']
Authors: Ana Marta Sequeira

Date:12/2020

Email:

##############################################################################
"""
import keras

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


# OPTIMIZERS
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9)
opt1 = keras.optimizers.SGD(learning_rate=lr_schedule)
opt2 = keras.optimizers.SGD(learning_rate=0.001)
opt3 = keras.optimizers.Adam(learning_rate=lr_schedule)
opt4 = keras.optimizers.Adam(learning_rate=0.001)
opt5 = keras.optimizers.Adam(learning_rate=0.01)
opt6 = keras.optimizers.RMSprop(learning_rate=lr_schedule)
opt7 = keras.optimizers.RMSprop(learning_rate=0.001)


def param_deep():
    param = {
        'run_dnn_simple':
            {'param_grid':
                [{
                    # 'optimizer':[opt2, opt4, opt5,opt7],
                    'hidden_layers': [(128, 64), (128, 64, 32), (64, 32)],
                    'dropout_rate': [(0.3,), (0.2,), (0.4,)],
                    'l1': [0, 1e-4, 1e-5],
                    'l2': [0, 1e-4, 1e-5],
                    # 'batch_size': [256, 512,1024],
                }],
                'distribution':
                    [{
                        # 'optimizer': [opt2, opt4,opt5, opt7],
                        'hidden_layers': [(64,), (128, 64), (128, 64, 32), (64, 32)],
                        'dropout_rate': [(0.1,), (0.2,), (0.25,), (0.3,), (0.35,), (0.4,), (0.5,)],
                        'l1': [0, 1e-3, 1e-4, 1e-5],
                        'l2': [0, 1e-3, 1e-4, 1e-5],
                        # 'batch_size': [128, 256, 512,1024],
                    }],
            },
        'run_dnn_embedding':
            {'param_grid':
                [{
                    # 'optimizer':[opt2, opt4, opt5,opt7],
                    'output_dim': [256, 128, 62, 512],
                    'hidden_layers': [(128, 64), (128, 64, 32), (64, 32)],
                    'dropout_rate': [(0.3,), (0.2,), (0.4,)],
                    'l1': [0, 1e-4, 1e-5],
                    'l2': [0, 1e-4, 1e-5],
                    # 'batch_size': [256, 512,1024],
                }],
                'distribution':
                    [{
                        # 'optimizer': [opt2, opt4,opt5, opt7],
                        'hidden_layers': [(64,), (128, 64), (128, 64, 32), (64, 32)],
                        'output_dim': [128, 64, 256],
                        'dropout_rate': [(0.1,), (0.2,), (0.25,), (0.3,), (0.35,), (0.4,), (0.5,)],
                        'l1': [0, 1e-3, 1e-4, 1e-5],
                        'l2': [0, 1e-3, 1e-4, 1e-5],
                        # 'batch_size': [128, 256, 512,1024],
                    }],
            },
        'run_lstm_simple':
            {'param_grid':
                [{
                    # 'optimizer':[opt2, opt4, opt5,opt7],
                    'lstm_layers': [(128, 64), (128, 64, 32), (64, 32)],
                    'dense_layers': [(32,), (64,), (64, 32)],
                    'dropout_rate': [(0.3,), (0.2,), (0.4,)],
                    'recurrent_dropout_rate': [(0.3,), (0.2,), (0.4,)],
                    'l1': [0, 1e-4, 1e-5],
                    'l2': [0, 1e-4, 1e-5],
                    'dropout_rate_dense': [(0.0,), (0.1,), (0.2,), (0.25,), (0.3,), (0.35,), (0.4,), (0.5,)],
                    # 'batch_size': [256, 512,1024],
                }],
                'distribution':
                    [{
                        # 'optimizer':[opt2, opt4, opt5,opt7],
                        'lstm_layers': [(128, 64), (128, 64, 32), (64, 32)],
                        'dense_layers': [(32,), (64,), (64, 32)],
                        'dropout_rate': [(0.1,), (0.2,), (0.25,), (0.3,), (0.35,), (0.4,), (0.5,)],
                        'recurrent_dropout_rate': [(0.1,), (0.2,), (0.25,), (0.3,), (0.35,), (0.4,), (0.5,)],
                        'l1': [0, 1e-4, 1e-5],
                        'l2': [0, 1e-4, 1e-5],
                        'dropout_rate_dense': [(0.0,), (0.1,), (0.2,), (0.25,), (0.3,), (0.35,), (0.4,), (0.5,)],
                        # 'batch_size': [256, 512,1024],
                    }],
            },

        'run_lstm_embedding':
            {'param_grid':
                [{
                    # 'optimizer':[opt2, opt4, opt5,opt7],
                    'output_dim': [128, 64, 32],
                    'lstm_layers': [(128, 64), (128, 64, 32), (64, 32)],
                    'dense_layers': [(32,), (64,), (64, 32)],
                    'dropout_rate': [(0.3,), (0.2,), (0.4,)],
                    'recurrent_dropout_rate': [(0.3,), (0.2,), (0.4,)],
                    'l1': [0, 1e-4, 1e-5],
                    'l2': [0, 1e-4, 1e-5],
                    'dropout_rate_dense': [(0.0,), (0.1,), (0.2,), (0.25,), (0.3,), (0.35,), (0.4,), (0.5,)]
                    # 'batch_size': [256, 512,1024],
                }],
                'distribution':
                    [{
                        # 'optimizer':[opt2, opt4, opt5,opt7],
                        'output_dim': [128, 64, 32],
                        'lstm_layers': [(128, 64), (128, 64, 32), (64, 32)],
                        'dense_layers': [(32,), (64,), (64, 32), (128, 64, 32)],
                        'dropout_rate': [(0.0,), (0.1,), (0.2,), (0.25,), (0.3,), (0.35,), (0.4,), (0.5,)],
                        'recurrent_dropout_rate': [(0.0,), (0.1,), (0.2,), (0.25,), (0.3,), (0.35,), (0.4,), (0.5,)],
                        'l1': [0, 1e-4, 1e-5],
                        'l2': [0, 1e-4, 1e-5],
                        'dropout_rate_dense': [(0.0,), (0.1,), (0.2,), (0.25,), (0.3,), (0.35,), (0.4,), (0.5,)],
                        # 'batch_size': [256, 512,1024],
                    }],
            },
        'run_cnn_1D':
            {'param_grid':
                [{
                    # 'optimizer':[opt2, opt4, opt5,opt7],
                    'filter_count': [(32, 64, 128), (32, 64)],
                    'kernel_size': [(3,), (5,), (10,), (20,)],
                    'dropout_cnn': [(0.0, 0.2, 0.2), (0,), (0.3,), (0.2,)],
                    'dense_layers': [(32,), (64,), (64, 32)],
                    'dropout_rate': [(0.3,), (0.2,), (0.4,), (0.5,)],
                    'l1': [0, 1e-4, 1e-5],
                    'l2': [0, 1e-4, 1e-5],
                    # 'batch_size': [256, 512,1024],
                }],
                'distribution':
                    [{
                        # 'optimizer':[opt2, opt4, opt5,opt7],
                        'filter_count': [(32, 64, 128), (32, 64)],
                        'kernel_size': [(3,), (5,), (10,), (20,)],
                        'dropout_cnn': [(0.0, 0.2, 0.2), (0,), (0.3,), (0.2,)],
                        'dense_layers': [(32,), (64,), (64, 32)],
                        'dropout_rate': [(0.3,), (0.2,), (0.4,), (0.5,)],
                        'l1': [0, 1e-4, 1e-5],
                        'l2': [0, 1e-4, 1e-5],
                    }],
            },

        'run_cnn_2D':
            {'param_grid':
                [{
                    # 'optimizer':[opt2, opt4, opt5,opt7],
                    'filter_count': [(32, 64, 128), (32, 64)],
                    'kernel_size': [(3,), (5,), (10,), (20,)],
                    'dropout_cnn': [(0.0, 0.2, 0.2), (0,), (0.3,), (0.2,)],
                    'pool_size': [((2, 2),), ((10, 10),)],
                    'dense_layers': [(32,), (64,), (64, 32)],
                    'dropout_rate': [(0.3,), (0.2,), (0.4,), (0.5,)],
                    'l1': [0, 1e-4, 1e-5],
                    'l2': [0, 1e-4, 1e-5],
                    # 'batch_size': [256, 512,1024],
                }],
                'distribution':
                    [{
                        # 'optimizer':[opt2, opt4, opt5,opt7],
                        'filter_count': [(32, 64, 128), (32, 64)],
                        'kernel_size': [(3,), (5,), (10,), (20,)],
                        'dropout_cnn': [(0.0, 0.2, 0.2), (0,), (0.3,), (0.2,)],
                        'pool_size': [((2, 2),), ((10, 10),)],
                        'dense_layers': [(32,), (64,), (64, 32)],
                        'dropout_rate': [(0.3,), (0.2,), (0.4,), (0.5,)],
                        'l1': [0, 1e-4, 1e-5],
                        'l2': [0, 1e-4, 1e-5],
                    }],
            },
        'run_cnn_lstm':
            {'param_grid':
                [{
                    # 'optimizer':[opt2, opt4, opt5,opt7],
                    'filter_count': [(32, 64, 128), (32, 64)],
                    'kernel_size': [(3,), (5,), (10,), (20,)],
                    'dropout_cnn': [(0.0, 0.2, 0.2), (0,), (0.3,), (0.2,)],
                    'lstm_layers': [(128, 64), (128, 64, 32), (64, 32)],
                    'dropout_rate': [(0.3,), (0.2,), (0.4,)],
                    'recurrent_dropout_rate': [(0.3,), (0.2,), (0.4,)],
                    'l1': [0, 1e-4, 1e-5],
                    'l2': [0, 1e-4, 1e-5],
                    'dense_layers': [(32,), (64,), (64, 32)],
                    'dropout_rate_dense': [(0.0,), (0.1,), (0.2,), (0.3,), (0.4,), (0.5,)]
                    # 'batch_size': [256, 512,1024],
                }],
                'distribution':
                    [{
                        # 'optimizer':[opt2, opt4, opt5,opt7],
                        'filter_count': [(32, 64, 128), (32, 64)],
                        'kernel_size': [(3,), (5,), (10,), (20,)],
                        'dropout_cnn': [(0.0, 0.2, 0.2), (0,), (0.3,), (0.2,)],
                        'lstm_layers': [(128, 64), (128, 64, 32), (64, 32)],
                        'dropout_rate': [(0.3,), (0.2,), (0.4,)],
                        'recurrent_dropout_rate': [(0.3,), (0.2,), (0.4,)],
                        'l1': [0, 1e-4, 1e-5],
                        'l2': [0, 1e-4, 1e-5],
                        'dense_layers': [(32,), (64,), (64, 32)],
                        'dropout_rate_dense': [(0.0,), (0.1,), (0.2,), (0.3,), (0.4,), (0.5,)]
                    }],
            },
    }
    return param
