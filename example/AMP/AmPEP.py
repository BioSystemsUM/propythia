# -*- coding: utf-8 -*-
"""
##############################################################################

File that describes the Antimicrobial peptides case study.
This section will present a comparative analysis to demonstrate the application and performance of proPythia
for addressing sequence-based prediction problems.


The first case study is with antimicrobial peptides and tries to replicate the study made by P. Bhadra and all,
“AMP: Sequence-based prediction of antimicrobial peptides using distribution patterns of amino acid properties
and random forest” which is described to highly perform on AMP prediction methods.
In the publication, Bhadra et al., used a dataset with a positive:negative ratio (AMP/non-AMP) of 1:3
, based on the distribution patterns of aa properties along the sequence (CTD features),
with a 10 fold cross validation RF model. The collection of data with sets of AMP and non-AMP data is freely
available at https://sourceforge.net/projects/axpep/files/).
Their model obtained a sensitivity of 0.95, a specificity and accuracy of 0.96, MCC of 0.9 and AUC-ROC of 0.98.


P. Bhadra, J. Yan, J. Li, S. Fong, and S. W. Siu, “AMP: Sequence-based prediction
of antimicrobial peptides using distribution patterns of amino acid properties and
random forest,” Scientific Reports, vol. 8, no. 1, pp. 1–10, 2018.


Authors: Ana Marta Sequeira

Date: 01/2021

Email:

##############################################################################
"""
import sys
import os

sys.path.append('/home/amsequeira/propythia')
import csv
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier

from propythia.sequence import ReadSequence
from propythia.descriptors import Descriptor
from propythia.preprocess import Preprocess
from propythia.feature_selection import FeatureSelection
from propythia.shallow_ml import ShallowML
from propythia.deep_ml import DeepML

##################################### AMPEP ###########################################################################
# The datasets were available and retrieved directly from  https://cbbio.online/AxPEP/?action=dataset
# The dataset is composed of 3268 AMP and 166791 non AMP.
# The authors of the study perform an analyse to check the better ratio to use. the ratio 1:1 yielding sensitivity and MCC
# whereas ratio 1:3 better Precision recall curve and same accuracy.

# to avoid the calculus of features multiple times. The top features analysed in the article were calculated previous.
# CTD
# AAC
# PAAC
# KMER 400 ( Dipeptide composition)
index = str('')
for path in [os.path.split(__file__)[0]]:
    if os.path.exists(os.path.join(path, index)):
        break

AMP_file = path + r'/datasets/M_model_train_AMP_sequence.fasta'
# AMP 3268  sequences
non_AMP_file = path + r'/datasets/M_model_train_nonAMP_sequence.fasta'
# non-AMP 166791 sequences

# the article performs subsets of datasets negatives until consumes the 166791 sequences.

AMP_file = '/home/amsequeira/propythia/propythia/example/AMP/datasets/M_model_train_AMP_sequence.fasta'
non_AMP_file = '/home/amsequeira/propythia/propythia/example/AMP/datasets/M_model_train_nonAMP_sequence.fasta'

# positive dataframe
AMP_data = pd.read_csv(AMP_file)
AMP_data = AMP_data.rename({'>': 'sequence'}, axis=1)
AMP_data = AMP_data[AMP_data['sequence'] != '>']

# negative dataset
non_AMP_data = pd.read_csv(non_AMP_file)
non_AMP_data = non_AMP_data.rename({'>': 'sequence'}, axis=1)
non_AMP_data = non_AMP_data[non_AMP_data['sequence'] != '>']

# get a negative dataset with 1:1 and 1:3 ratio

# 1:3 ratio 3268 *3
non_AMP_data_1_3 = non_AMP_data.sample(n=AMP_data.shape[0] * 3, replace=False)
# 1:1 ratio
non_AMP_data_1_1 = non_AMP_data.sample(n=AMP_data.shape[0], replace=False)


# calculate features
def calculate_feature(data):
    list_feature = []
    count = 0
    for seq in data['sequence']:
        count += 1
        res = {'sequence': seq}
        sequence = ReadSequence()  # creating sequence object
        ps = sequence.read_protein_sequence(seq)
        protein = Descriptor(ps)  # creating object to calculate descriptors
        # feature = protein.adaptable([32,20,24]) # using the function adaptable. calculate CTD, aac, dpc, paac and  feature
        feature = protein.adaptable([19, 20, 21, 24, 26, 32], lamda_paac=4, lamda_apaac=4)
        # feature = protein.get_all(lamda_paac=5, lamda_apaac=5) #minimal seq len = 5
        # lambda should not be larger than len(sequence)
        res.update(feature)
        list_feature.append(res)
        print(count)
    df = pd.DataFrame(list_feature)
    return df


amp_feature = calculate_feature(AMP_data)
no_amp_feature = calculate_feature(non_AMP_data_1_3)

amp_feature.to_csv(path + r'/datasets/ampep_feature.csv', index=False)
no_amp_feature.to_csv(path + r'/datasets/no_ampep_feature.csv', index=False)


def get_dataset_feature_selected(feature, df):
    if feature == 'CTD':
        filter_col = [col for col in df if col.startswith('_')]
        data = df[filter_col]
    elif feature == 'C':
        # startswith _ any charachter any time have  a C digits no matter how many
        data = df.filter(regex=r'_.+C\d', axis=1)
    elif feature == 'T':
        data = df.filter(regex=r'_.+T\d', axis=1)
    elif feature == 'D':
        data = df.filter(regex=r'_.+D\d', axis=1)
    elif feature == 'AAC':
        filter_col = [col for col in df if len(col) == 1]
        data = df[filter_col]
    elif feature == 'DPC':
        filter_col = [col for col in df if len(col) == 2]
        data = df[filter_col]
    elif feature == 'PAAC':
        filter_col = [col for col in df if col.startswith('PAAC')]
        data = df[filter_col]
    else:
        return
    return data


# MIMIC AMPEP MODEL  ratio 1:3  Feature D RF  CV = 10 No reference to scaling
# define parameters by article In the article AMPEP suggests that tested 100 200 and 500 estimators
# with 100 being the best
# Besides the article did not clarify which measure they use. here we optimize MCC


AMP_data = pd.read_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/ampep_feat.csv')
AMP_data['label'] = 1
non_AMP_data_1_3 = pd.read_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/non_ampep_feat.csv')
non_AMP_data_1_3['label'] = 0

dataset = pd.concat([AMP_data, non_AMP_data_1_3])

fps_y = dataset['label']
fps_x = dataset.loc[:, dataset.columns != 'label']
fps_x = fps_x.filter(regex=r'_.+D\d', axis=1)  # select just CTD D feature 105 columns
# dataset.shape [13072 rows x 105 columns]


X_train, X_test, y_train, y_test = train_test_split(fps_x, fps_y, stratify=fps_y)
# standard scaler article does not refer scaling and do not validate in x_test
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

ml = ShallowML(X_train, X_test, y_train, y_test, report_name=None, columns_names=fps_x.columns)
param_grid = [{'clf__n_estimators': [100, 200, 500], 'clf__max_features': ['sqrt']}]

# optimize MCC
best_rf_model_AMPEP = ml.train_best_model('rf', score=make_scorer(matthews_corrcoef), param_grid=param_grid, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_rf_model_AMPEP)
# Model with rank: 1
# Mean validation score: 0.895 (std: 0.015)
# Parameters: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# Model with rank: 2
# Mean validation score: 0.894 (std: 0.016)
# Parameters: {'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
#
# Model with rank: 3
# Mean validation score: 0.894 (std: 0.018)
# Parameters: {'clf__max_features': 'sqrt', 'clf__n_estimators': 200}
#
# Best score rf (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.895
# Parameters:	{'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.894770 (0.014569) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.893823 (0.017550) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 200}
# 0.894115 (0.015774) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
#
# {'Accuracy': 0.9623623011015912,
#  'MCC': 0.9018996380241933,
#  'log_loss': 0.1577294137777963,
#  'f1 score': 0.9266547406082289,
#  'roc_auc': 0.9585883312933496,
#  'Precision': array([0.25      , 0.90348837, 1.        ]),
#  'Recall': array([1.        , 0.95104039, 0.        ]),
#  'fdr': 0.09651162790697675,
#  'sn': 0.9510403916768666,
#  'sp': 0.9661362709098327}
#
# print(report)
# precision    recall  f1-score   support
# 0       0.98      0.97      0.97      2451
# 1       0.90      0.95      0.93       817
# accuracy                           0.96      3268
# macro avg       0.94      0.96      0.95      3268
# weighted avg       0.96      0.96      0.96      3268
#
# Out[16]:
# array([[2368,   83],
#        [  40,  777]])
# optimize ROC_AUC
best_rf_model_AMPEPparameters = ml.train_best_model('rf', score='roc_auc', param_grid=param_grid)
scores, report, cm, cm2 = ml.score_testset(best_rf_model_AMPEPparameters)
# Model with rank: 1
# Mean validation score: 0.989 (std: 0.002)
# Parameters: {'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
#
# Model with rank: 2
# Mean validation score: 0.989 (std: 0.002)
# Parameters: {'clf__max_features': 'sqrt', 'clf__n_estimators': 200}
#
# Model with rank: 3
# Mean validation score: 0.988 (std: 0.002)
# Parameters: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# Best score rf (scorer: roc_auc) and parameters from a 10-fold cross validation:
# MCC score:	0.989
# Parameters:	{'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
# 0.988301 (0.002223) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.988574 (0.002286) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 200}
# 0.988755 (0.002226) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 500}

# precision    recall  f1-score   support
# 0       0.98      0.97      0.97      2451
# 1       0.90      0.94      0.92       817
# accuracy                           0.96      3268
# macro avg       0.94      0.95      0.95      3268
# weighted avg       0.96      0.96      0.96      3268

# {'Accuracy': 0.9602203182374541,
#  'MCC': 0.8958318653758695,
#  'log_loss': 0.13154182944385587,
#  'f1 score': 0.9221556886227545,
#  'roc_auc': 0.9543043655650754,
#  'Precision': array([0.25      , 0.90269637, 1.        ]),
#  'Recall': array([1.        , 0.94247246, 0.        ]),
#  'fdr': 0.09730363423212192,
#  'sn': 0.9424724602203183,
#  'sp': 0.9661362709098327}

# optimize accuracy
best_rf_model_AMPEPparameters = ml.train_best_model('rf', score='accuracy', param_grid=param_grid)
scores, report, cm, cm2 = ml.score_testset(best_rf_model_AMPEPparameters)
# Model with rank: 1
# Mean validation score: 0.960 (std: 0.006)
# Parameters: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# Model with rank: 2
# Mean validation score: 0.959 (std: 0.006)
# Parameters: {'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
#
# Model with rank: 3
# Mean validation score: 0.959 (std: 0.007)
# Parameters: {'clf__max_features': 'sqrt', 'clf__n_estimators': 200}
#
# Best score rf (scorer: accuracy) and parameters from a 10-fold cross validation:
# MCC score:	0.960
# Parameters:	{'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.959711 (0.005786) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.959303 (0.006921) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 200}
# 0.959405 (0.006230) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
#
# Out[20]:
# {'Accuracy': 0.9623623011015912,
#  'MCC': 0.9018996380241933,
#  'log_loss': 0.1577294137777963,
#  'f1 score': 0.9266547406082289,
#  'roc_auc': 0.9585883312933496,
#  'Precision': array([0.25      , 0.90348837, 1.        ]),
#  'Recall': array([1.        , 0.95104039, 0.        ]),
#  'fdr': 0.09651162790697675,
#  'sn': 0.9510403916768666,
#  'sp': 0.9661362709098327}
# print(report)
# precision    recall  f1-score   support
# 0       0.98      0.97      0.97      2451
# 1       0.90      0.95      0.93       817
# accuracy                           0.96      3268
# macro avg       0.94      0.96      0.95      3268
# weighted avg       0.96      0.96      0.96      3268
#
# array([[2368,   83],
#        [  40,  777]])


# just with 100
param_grid = [{'clf__n_estimators': [100], 'clf__max_features': ['sqrt']}]
best_rf_model_AMPEPparameters = ml.train_best_model('rf', score='roc_auc', param_grid=param_grid)
scores, report, cm, cm2 = ml.score_testset(best_rf_model_AMPEPparameters)
# 0.988301 (0.002223) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# {'Accuracy': 0.9583843329253366,
#  'MCC': 0.8910047515974634,
#  'log_loss': 0.16009858649630554,
#  'f1 score': 0.918562874251497,
#  'roc_auc': 0.9518563851489189,
#  'Precision': array([0.25      , 0.89917937, 1.        ]),
#  'Recall': array([1.        , 0.93880049, 0.        ]),
#  'fdr': 0.10082063305978899,
#  'sn': 0.9388004895960832,
#  'sp': 0.9649122807017544}

# print(report)
# precision    recall  f1-score   support
# 0       0.98      0.96      0.97      2451
# 1       0.90      0.94      0.92       817
# accuracy                           0.96      3268
# macro avg       0.94      0.95      0.95      3268
# weighted avg       0.96      0.96      0.96      3268

# optimize MCC
best_rf_model_AMPEPparameters = ml.train_best_model('rf', score=make_scorer(matthews_corrcoef), param_grid=param_grid)
scores, report, cm, cm2 = ml.score_testset(best_rf_model_AMPEPparameters)

# Model with rank: 1
# Mean validation score: 0.894 (std: 0.018)
# Parameters: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# Best score rf (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.894
# Parameters:	{'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.893532 (0.017748) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# {'Accuracy': 0.9583843329253366,
#  'MCC': 0.8910047515974634,
#  'log_loss': 0.16009858649630554,
#  'f1 score': 0.918562874251497,
#  'roc_auc': 0.9518563851489189,
#  'Precision': array([0.25      , 0.89917937, 1.        ]),
#  'Recall': array([1.        , 0.93880049, 0.        ]),
#  'fdr': 0.10082063305978899,
#  'sn': 0.9388004895960832,
#  'sp': 0.9649122807017544}
#
# precision    recall  f1-score   support
# 0       0.98      0.96      0.97      2451
# 1       0.90      0.94      0.92       817
# accuracy                           0.96      3268
# macro avg       0.94      0.95      0.95      3268
# weighted avg       0.96      0.96      0.96      3268


best_rf_model_AMPEPparameters = ml.train_best_model('rf', score='accuracy', param_grid=param_grid)
scores, report, cm, cm2 = ml.score_testset(best_rf_model_AMPEPparameters)

# Model with rank: 1
# Mean validation score: 0.959 (std: 0.007)
# Parameters: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# Best score rf (scorer: accuracy) and parameters from a 10-fold cross validation:
# MCC score:	0.959
# Parameters:	{'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.959099 (0.006803) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}

# precision    recall  f1-score   support
# 0       0.98      0.96      0.97      2451
# 1       0.90      0.94      0.92       817
# accuracy                           0.96      3268
# macro avg       0.94      0.95      0.95      3268
# weighted avg       0.96      0.96      0.96      3268
#
# Out[35]:
# {'Accuracy': 0.9583843329253366,
#  'MCC': 0.8910047515974634,
#  'log_loss': 0.16009858649630554,
#  'f1 score': 0.918562874251497,
#  'roc_auc': 0.9518563851489189,
#  'Precision': array([0.25      , 0.89917937, 1.        ]),
#  'Recall': array([1.        , 0.93880049, 0.        ]),
#  'fdr': 0.10082063305978899,
#  'sn': 0.9388004895960832,
#  'sp': 0.9649122807017544}


# AMPEP SVM
best_rf_model_AMPEP = ml.train_best_model('svm', score=make_scorer(matthews_corrcoef), param_grid=None, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_rf_model_AMPEP)
# Model with rank: 1
# Mean validation score: 0.858 (std: 0.015)
# Parameters: {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#
# Model with rank: 2
# Mean validation score: 0.856 (std: 0.020)
# Parameters: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#
# Model with rank: 3
# Mean validation score: 0.833 (std: 0.024)
# Parameters: {'clf__C': 0.1, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#
# Best score svm (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.858
# Parameters:	{'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.743637 (0.017994) with: {'clf__C': 0.01, 'clf__kernel': 'linear'}
# 0.745704 (0.017870) with: {'clf__C': 0.1, 'clf__kernel': 'linear'}
# 0.745775 (0.017884) with: {'clf__C': 1.0, 'clf__kernel': 'linear'}
# 0.745775 (0.017884) with: {'clf__C': 10, 'clf__kernel': 'linear'}
# 0.786302 (0.018447) with: {'clf__C': 0.01, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.000000 (0.000000) with: {'clf__C': 0.01, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.000000 (0.000000) with: {'clf__C': 0.01, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.832922 (0.023620) with: {'clf__C': 0.1, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.650580 (0.019923) with: {'clf__C': 0.1, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.000000 (0.000000) with: {'clf__C': 0.1, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.857759 (0.014934) with: {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.764387 (0.017258) with: {'clf__C': 1.0, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.630991 (0.019671) with: {'clf__C': 1.0, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.856443 (0.019541) with: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.800600 (0.014066) with: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.739123 (0.022577) with: {'clf__C': 10, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
#
#
# {'Accuracy': 0.9519583843329253,
#  'MCC': 0.8708317539683759,
#  'f1 score': 0.9025450031036624,
#  'roc_auc': 0.9312525499796,
#  'Precision': array([0.25      , 0.91561713, 1.        ]),
#  'Recall': array([1.        , 0.88984088, 0.        ]),
#  'fdr': 0.08438287153652393,
#  'sn': 0.8898408812729498,
#  'sp': 0.9726642186862505}
#
# Out[49]:
# array([[2384,   67],
#        [  90,  727]])
#
# print(report)
# precision    recall  f1-score   support
# 0       0.96      0.97      0.97      2451
# 1       0.92      0.89      0.90       817
# accuracy                           0.95      3268
# macro avg       0.94      0.93      0.94      3268
# weighted avg       0.95      0.95      0.95      3268

# AMPEP RF with CTD
fps_x = dataset.loc[:, dataset.columns != 'label']
fps_x = fps_x.filter(regex=r'_.+', axis=1)  # select just CTD features
# dataset.shape [13072 rows x 147 columns]
X_train, X_test, y_train, y_test = train_test_split(fps_x, fps_y, stratify=fps_y)
# standard scaler article does not refer scaling and do not validate in x_test
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
ml = ShallowML(X_train, X_test, y_train, y_test, report_name=None, columns_names=fps_x.columns)
param_grid = [{'clf__n_estimators': [100], 'clf__max_features': ['sqrt']}]
# optimize MCC
best_rf_model_AMPEP = ml.train_best_model('rf', score=make_scorer(matthews_corrcoef), param_grid=param_grid, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_rf_model_AMPEP)

# Model with rank: 1
# Mean validation score: 0.893 (std: 0.013)
# Parameters: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# Best score rf (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.893
# Parameters:	{'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.892510 (0.013208) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# {'Accuracy': 0.9614443084455324,
#  'MCC': 0.8989281704127183,
#  'log_loss': 0.13634901537089303,
#  'f1 score': 0.9244604316546763,
#  'roc_auc': 0.9555283557731538,
#  'Precision': array([0.25      , 0.90599295, 1.        ]),
#  'Recall': array([1.        , 0.94369645, 0.        ]),
#  'fdr': 0.09400705052878966,
#  'sn': 0.9436964504283966,
#  'sp': 0.9673602611179111}
#
# print(report)
# precision    recall  f1-score   support
# 0       0.98      0.97      0.97      2451
# 1       0.91      0.94      0.92       817
# accuracy                           0.96      3268
# macro avg       0.94      0.96      0.95      3268
# weighted avg       0.96      0.96      0.96      3268

# AMPEP RF with AAC
fps_x = dataset.loc[:, dataset.columns != 'label']
filter_col = [col for col in fps_x if len(col) == 1]
fps_x = fps_x[filter_col]
# dataset.shape (13072, 20)
X_train, X_test, y_train, y_test = train_test_split(fps_x, fps_y, stratify=fps_y)
# standard scaler article does not refer scaling and do not validate in x_test
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
ml = ShallowML(X_train, X_test, y_train, y_test, report_name=None, columns_names=fps_x.columns)
param_grid = [{'clf__n_estimators': [100], 'clf__max_features': ['sqrt']}]

# optimize MCC
best_rf_model_AMPEP = ml.train_best_model('rf', score=make_scorer(matthews_corrcoef), param_grid=param_grid, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_rf_model_AMPEP)

# Model with rank: 1
# Mean validation score: 0.865 (std: 0.020)
# Parameters: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# Best score rf (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.865
# Parameters:	{'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.864972 (0.020232) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# print(report)
# precision    recall  f1-score   support
# 0       0.97      0.97      0.97      2451
# 1       0.91      0.89      0.90       817
# accuracy                           0.95      3268
# macro avg       0.94      0.93      0.93      3268
# weighted avg       0.95      0.95      0.95      3268
#
# {'Accuracy': 0.9513463892288861,
#  'MCC': 0.8696146699200294,
#  'log_loss': 0.1750591979857633,
#  'f1 score': 0.9019123997532387,
#  'roc_auc': 0.9324765401876786,
#  'Precision': array([0.25      , 0.90920398, 1.        ]),
#  'Recall': array([1.        , 0.89473684, 0.        ]),
#  'fdr': 0.09079601990049752,
#  'sn': 0.8947368421052632,
#  'sp': 0.9702162382700938}

# second time
# Model with rank: 1
# Mean validation score: 0.867 (std: 0.015)
# Parameters: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# Best score rf (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.867
# Parameters:	{'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.866783 (0.015059) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# Out[42]:
# {'Accuracy': 0.9574663402692778,
#  'MCC': 0.8860224938807848,
#  'log_loss': 0.16707701144213824,
#  'f1 score': 0.9142504626773597,
#  'roc_auc': 0.9406364749082006,
#  'Precision': array([0.25      , 0.92164179, 1.        ]),
#  'Recall': array([1.        , 0.90697674, 0.        ]),
#  'fdr': 0.07835820895522388,
#  'sn': 0.9069767441860465,
#  'sp': 0.9742962056303549}
#
# print(report)
# precision    recall  f1-score   support
# 0       0.97      0.97      0.97      2451
# 1       0.92      0.91      0.91       817
# accuracy                           0.96      3268
# macro avg       0.95      0.94      0.94      3268
# weighted avg       0.96      0.96      0.96      3268
# Out[44]:
# array([[2388,   63],
#        [  76,  741]])

# optimize Accuracy
# best_rf_model_AMPEP=ml.train_best_model('rf', score='accuracy',param_grid=param_grid, cv=10)
# scores, report, cm, cm2 = ml.score_testset(best_rf_model_AMPEP)
#
# Model with rank: 1
# Mean validation score: 0.950 (std: 0.006)
# Parameters: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# Best score rf (scorer: accuracy) and parameters from a 10-fold cross validation:
# MCC score:	0.950
# Parameters:	{'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.950123 (0.005751) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# {'Accuracy': 0.9574663402692778,
#  'MCC': 0.8860224938807848,
#  'log_loss': 0.16707701144213824,
#  'f1 score': 0.9142504626773597,
#  'roc_auc': 0.9406364749082006,
#  'Precision': array([0.25      , 0.92164179, 1.        ]),
#  'Recall': array([1.        , 0.90697674, 0.        ]),
#  'fdr': 0.07835820895522388,
#  'sn': 0.9069767441860465,
#  'sp': 0.9742962056303549}
#
# print(report)
# precision    recall  f1-score   support
# 0       0.97      0.97      0.97      2451
# 1       0.92      0.91      0.91       817
# accuracy                           0.96      3268
# macro avg       0.95      0.94      0.94      3268
# weighted avg       0.96      0.96      0.96      3268
#
# cm
# Out[33]:
# array([[2388,   63],
#        [  76,  741]])

# optimize ROC AUC
best_rf_model_AMPEP = ml.train_best_model('rf', score='roc_auc', param_grid=param_grid, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_rf_model_AMPEP)


#
# Model with rank: 1
# Mean validation score: 0.983 (std: 0.004)
# Parameters: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# Best score rf (scorer: roc_auc) and parameters from a 10-fold cross validation:
# MCC score:	0.983
# Parameters:	{'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.983464 (0.003775) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# Out[37]:
# {'Accuracy': 0.9574663402692778,
#  'MCC': 0.8860224938807848,
#  'log_loss': 0.16707701144213824,
#  'f1 score': 0.9142504626773597,
#  'roc_auc': 0.9406364749082006,
#  'Precision': array([0.25      , 0.92164179, 1.        ]),
#  'Recall': array([1.        , 0.90697674, 0.        ]),
#  'fdr': 0.07835820895522388,
#  'sn': 0.9069767441860465,
#  'sp': 0.9742962056303549}
# print(report)
# precision    recall  f1-score   support
# 0       0.97      0.97      0.97      2451
# 1       0.92      0.91      0.91       817
# accuracy                           0.96      3268
# macro avg       0.95      0.94      0.94      3268
# weighted avg       0.96      0.96      0.96      3268
# array([[2388,   63],
#        [  76,  741]])


#######################################################################################
########################################################################################
#######################################################################################
########################################################################################
# sem test set. cross validate

# multiple metrics
def sp(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    scores['fdr'] = float(fp) / (tp + fp)
    sn = float(tp) / (tp + fn)
    sp = float(tn) / (tn + fp)
    # fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # speci = 1 - fpr
    return sp


def sn(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    scores['fdr'] = float(fp) / (tp + fp)
    sn = float(tp) / (tp + fn)
    sp = float(tn) / (tn + fp)
    # fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # speci = 1 - fpr
    return sn


scoring = {'AUC': 'roc_auc', 'Accuracy': 'accuracy', 'MCC': make_scorer(matthews_corrcoef),
           'sn': make_scorer(sn), 'sp': make_scorer(sp)}
AMP_data = pd.read_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/ampep_feat.csv')
AMP_data['label'] = 1
non_AMP_data_1_3 = pd.read_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/non_ampep_feat.csv')
non_AMP_data_1_3['label'] = 0

dataset = pd.concat([AMP_data, non_AMP_data_1_3])

fps_y = dataset['label']
fps_x = dataset.loc[:, dataset.columns != 'label']
fps_x_d = fps_x.filter(regex=r'_.+D\d', axis=1)  # select just CTD D feature 105 columns
# dataset.shape [13072 rows x 105 columns]
scaler_d = StandardScaler().fit_transform(fps_x_d)

ml = ShallowML(X_train=scaler_d, X_test=None, y_train=fps_y, y_test=None, report_name=None, columns_names=fps_x.columns)
scores = ml.cross_val_score_model('rf', score=scoring, n_estimators=100, max_features='sqrt', cv=10)

# no scaled
# 'test_mean_AUC': 0.9801640184129852,
# 'test_std_AUC': 0.0178828958628194,
# 'test_mean_Accuracy': 0.9447659509252695,
# 'test_std_Accuracy': 0.022148444314490194,
# 'test_mean_MCC': 0.8518662471840021,
# 'test_std_MCC': 0.06381619760122965,
# 'test_mean_sn': 0.883405564623553,
# 'test_std_sn': 0.09194935281880816,
# 'test_mean_sp': 0.9652197882210988,
# 'test_std_sp': 0.005511157738946883}

# scaled
# 'test_mean_AUC': 0.9801593395579717,
# 'test_std_AUC': 0.017924468325709396,
# 'test_mean_Accuracy': 0.944842462019378,
# 'test_std_Accuracy': 0.022136919996916096,
# 'test_mean_MCC': 0.8520501976502475,
# 'test_std_MCC': 0.0637965785901641,
# 'test_mean_sn': 0.883405564623553,
# 'test_std_sn': 0.09194935281880816,
# 'test_mean_sp': 0.9653217250202832,
# 'test_std_sp': 0.005349840284171315}
# 200 estimators
scores = ml.cross_val_score_model('rf', score=scoring, n_estimators=200, max_features='sqrt', cv=10)
# 'test_mean_AUC': 0.9818036833138383,
# 'test_std_AUC': 0.014054065045333908,
# 'test_mean_Accuracy': 0.9436949710919093,
# 'test_std_Accuracy': 0.02321799827755699,
# 'test_mean_MCC': 0.848765748364389,
# 'test_std_MCC': 0.06709529226048634,
# 'test_mean_sn': 0.8800407121817602,
# 'test_std_sn': 0.09557155428410208,
# 'test_mean_sp': 0.9649136657721193,
# 'test_std_sp': 0.004971244662948331}
scores = ml.cross_val_score_model('rf', score=scoring, n_estimators=300, max_features='sqrt', cv=10)
# 'test_mean_AUC': 0.9824285635780592,
# 'test_std_AUC': 0.013082526615299026,
# 'test_mean_Accuracy': 0.9441538621724002,
# 'test_std_Accuracy': 0.023124152620470108,
# 'test_mean_MCC': 0.8500450440945793,
# 'test_std_MCC': 0.06694140817631423,
# 'test_mean_sn': 0.8809590814431247,
# 'test_std_sn': 0.09735003740430861,
# 'test_mean_sp': 0.965219788221099,
# 'test_std_sp': 0.004912329301811906}

fps_x_ctd = fps_x.filter(regex=r'_.+', axis=1)  # select just CTD features 147
scaler_ctd = StandardScaler().fit_transform(fps_x_ctd)

ml = ShallowML(X_train=fps_x_ctd, X_test=None, y_train=fps_y, y_test=None, report_name=None,
               columns_names=fps_x_ctd.columns)
scores = ml.cross_val_score_model('rf', score=scoring, n_estimators=100, max_features='sqrt', cv=10)

# 'test_mean_AUC': 0.9823699984895237,
# 'test_std_AUC': 0.014526871487410045,
# 'test_mean_Accuracy': 0.9453014700893096,
# 'test_std_Accuracy': 0.023590674938835852,
# 'test_mean_MCC': 0.8528147440748042,
# 'test_std_MCC': 0.06811840913842604,
# 'test_mean_sn': 0.880354027128947,
# 'test_std_sn': 0.09756912714768473,
# 'test_mean_sp': 0.9669533379100874,
# 'test_std_sp': 0.004824345485157682}

filter_col = [col for col in fps_x if len(col) == 1]
fps_x_aac = fps_x[filter_col]
scaler_aac = StandardScaler().fit_transform(fps_x_aac)
ml = ShallowML(X_train=fps_x_aac, X_test=None, y_train=fps_y, y_test=None, report_name=None,
               columns_names=fps_x_aac.columns)
scores = ml.cross_val_score_model('rf', score=scoring, n_estimators=100, max_features='sqrt', cv=10)
# 'test_mean_AUC': 0.9752663617144627,
# 'test_std_AUC': 0.01322046732447928,
# 'test_mean_Accuracy': 0.9380343200222747,
# 'test_std_Accuracy': 0.016618816146075948,
# 'test_mean_MCC': 0.831708419442359,
# 'test_std_MCC': 0.04786804133368506,
# 'test_mean_sn': 0.8405714714545693,
# 'test_std_sn': 0.07582426653295792,
# 'test_mean_sp': 0.9705234142586697,
# 'test_std_sp': 0.004495045974773174}
#######################################################################################
# #######################################################################################
#######################################################################################
# #######################################################################################
# Calculate all features
# [19, 20, 21, 24,26,32 ALL physchemical, AAC, DPC, PAAC, APAAC, CTD


AMP_data = pd.read_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/ampep_feat_complete.csv')
AMP_data = AMP_data.drop({'Unnamed: 0'}, axis=1)
AMP_data['label'] = 1
non_AMP_data_1_3 = pd.read_csv(
    '/home/amsequeira/propythia/propythia/example/AMP/datasets/non_ampep_feat_complete.csv')
non_AMP_data_1_3['label'] = 0

dataset = pd.concat([AMP_data, non_AMP_data_1_3])  # (13072, 642)

fps_y = dataset['label']
fps_x = dataset.drop(['label', 'sequence'], axis=1)
#  Preprocess
prepro = Preprocess()  # Create Preprocess object
fps_x_clean, columns_deleted = prepro.preprocess(fps_x, columns_names=True, threshold=0,
                                                 standard=True)  # [13072 rows x 628 columns]

# NO FEATURE SELECTION
X_train, X_test, y_train, y_test = train_test_split(fps_x_clean, fps_y, stratify=fps_y)
# standard scaler article does not refer scaling and do not validate in x_test
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
ml = ShallowML(X_train, X_test, y_train, y_test, report_name=None, columns_names=fps_x_clean.columns)

best_svm_model_AMPEP = ml.train_best_model('svm', score=make_scorer(matthews_corrcoef), param_grid=None, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_rf_model_AMPEP)

# GridSearchCV took 967.01 seconds for 16 candidate parameter settings.
#
# Model with rank: 1
# Mean validation score: 0.886 (std: 0.009)
# Parameters: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
#
# Model with rank: 2
# Mean validation score: 0.882 (std: 0.010)
# Parameters: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#
# Model with rank: 3
# Mean validation score: 0.872 (std: 0.013)
# Parameters: {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#
# Best score svm (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.886
# Parameters:	{'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.853616 (0.015016) with: {'clf__C': 0.01, 'clf__kernel': 'linear'}
# 0.853937 (0.019569) with: {'clf__C': 0.1, 'clf__kernel': 'linear'}
# 0.848943 (0.016687) with: {'clf__C': 1.0, 'clf__kernel': 'linear'}
# 0.849841 (0.018029) with: {'clf__C': 10, 'clf__kernel': 'linear'}
# 0.761175 (0.013455) with: {'clf__C': 0.01, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.761884 (0.015228) with: {'clf__C': 0.01, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.020002 (0.036148) with: {'clf__C': 0.01, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.827961 (0.016308) with: {'clf__C': 0.1, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.833562 (0.017486) with: {'clf__C': 0.1, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.790102 (0.020027) with: {'clf__C': 0.1, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.871676 (0.013139) with: {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.865082 (0.011854) with: {'clf__C': 1.0, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.844189 (0.013043) with: {'clf__C': 1.0, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.882052 (0.009819) with: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.885699 (0.009457) with: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.863233 (0.016291) with: {'clf__C': 10, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# clf__C clf__kernel clf__gamma     means      stds
# 0     0.01      linear        NaN  0.853616  0.015016
# 1     0.10      linear        NaN  0.853937  0.019569
# 2     1.00      linear        NaN  0.848943  0.016687
# 3    10.00      linear        NaN  0.849841  0.018029
# 4     0.01         rbf      scale  0.761175  0.013455
# 5     0.01         rbf      0.001  0.761884  0.015228
# 6     0.01         rbf     0.0001  0.020002  0.036148
# 7     0.10         rbf      scale  0.827961  0.016308
# 8     0.10         rbf      0.001  0.833562  0.017486
# 9     0.10         rbf     0.0001  0.790102  0.020027
# 10    1.00         rbf      scale  0.871676  0.013139
# 11    1.00         rbf      0.001  0.865082  0.011854
# 12    1.00         rbf     0.0001  0.844189  0.013043
# 13   10.00         rbf      scale  0.882052  0.009819
# 14   10.00         rbf      0.001  0.885699  0.009457
# 15   10.00         rbf     0.0001  0.863233  0.016291
# {'Accuracy': 0.988984088127295,
#  'MCC': 0.9706772515054289,
#  'f1 score': 0.978021978021978,
#  'roc_auc': 0.9861281109751121,
#  'Precision': array([0.25      , 0.97563946, 1.        ]),
#  'Recall': array([1.        , 0.98041616, 0.        ]),
#  'fdr': 0.024360535931790498,
#  'sn': 0.9804161566707467,
#  'sp': 0.9918400652794778}
#
# array([[2431,   20],
#        [  16,  801]])
#
# print(report)
# precision    recall  f1-score   support
# 0       0.99      0.99      0.99      2451
# 1       0.98      0.98      0.98       817
# accuracy                           0.99      3268
# macro avg       0.98      0.99      0.99      3268
# weighted avg       0.99      0.99      0.99      3268

best_rf_model_AMPEP = ml.train_best_model('rf', score=make_scorer(matthews_corrcoef), param_grid=None, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_rf_model_AMPEP)
# Model with rank: 1
# Mean validation score: 0.904 (std: 0.010)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
#
# Model with rank: 2
# Mean validation score: 0.901 (std: 0.007)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# Model with rank: 3
# Mean validation score: 0.890 (std: 0.011)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 500}
#
# Best score rf (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.904
# Parameters:	{'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
# 0.886567 (0.007516) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 10}
# 0.900775 (0.007423) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.904313 (0.009885) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
# 0.863913 (0.016292) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 10}
# 0.889077 (0.010723) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 100}
# 0.890482 (0.011338) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 500}
# clf__bootstrap clf__criterion  ...     means      stds
# 0            True           gini  ...  0.886567  0.007516
# 1            True           gini  ...  0.900775  0.007423
# 2            True           gini  ...  0.904313  0.009885
# 3            True           gini  ...  0.863913  0.016292
# 4            True           gini  ...  0.889077  0.010723
# 5            True           gini  ...  0.890482  0.011338
# [6 rows x 6 columns]
# precision    recall  f1-score   support
# 0       0.98      0.97      0.98      2451
# 1       0.92      0.94      0.93       817
# accuracy                           0.96      3268
# macro avg       0.95      0.95      0.95      3268
# weighted avg       0.96      0.96      0.96      3268
#
# array([[2382,   69],
#        [  53,  764]])
#
# {'Accuracy': 0.9626682986536108,
#  'MCC': 0.9011702619832975,
#  'log_loss': 0.107565029479726,
#  'f1 score': 0.9260606060606061,
#  'roc_auc': 0.9534883720930234,
#  'Precision': array([0.25      , 0.91716687, 1.        ]),
#  'Recall': array([1.        , 0.93512852, 0.        ]),
#  'fdr': 0.08283313325330131,
#  'sn': 0.9351285189718482,
#  'sp': 0.9718482252141983}
#

# FEATURE SELECTION
X_train, X_test, y_train, y_test = train_test_split(fps_x_clean, fps_y, stratify=fps_y)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# Select features
fsel = FeatureSelection(X_train, y_train, columns_names=fps_x_clean.columns)
# # KBest com *mutual info classif*
transf, X_fit_uni, X_transf_uni, column_selected, scores, scores_df = \
    fsel.run_univariate(score_func=mutual_info_classif, mode='k_best', param=300)
# Finished 'run_univariate' in 35.3539 secs
# SCORES

# scores_ranking
# _HydrophobicityD3001                         0.376546
# _PolarityD1001                               0.375712
# _SolventAccessibilityD3001                   0.374121
# _SecondaryStrD1001                           0.362802
# _PolarizabilityD3001                         0.337832
# lenght                                       0.336494
# hydrogen                                     0.329063
# formulaH                                     0.327961
# tot                                          0.325986
# single                                       0.325882
# _ChargeD2001                                 0.325401
# MW_modlamp                                   0.325097
# formulaC                                     0.324503
# formulaN                                     0.314752
# formulaO                                     0.314392
# double                                       0.312212
# M                                            0.293370
# charge                                       0.286321
# _ChargeD3025                                 0.263827
# E                                            0.252552
# APAAC13                                      0.249817
# _NormalizedVDWVD2001                         0.243705
# _ChargeD3001                                 0.242130
# _ChargeD3100                                 0.241970
# PAAC13                                       0.240843
# _ChargeD3075                                 0.240498
# SecStruct_sheet                              0.238668
# Molar_extinction_coefficient_oxidized        0.235841
# _ChargeD3050                                 0.235656
# _PolarityD2025                               0.234994


X_train = X_transf_uni
X_test = transf.transform(X_test)
columns = fps_x_clean.columns[column_selected]
ml = ShallowML(X_train, X_test, y_train, y_test, report_name=None, columns_names=columns)

best_svm_model_P = ml.train_best_model('svm', score=make_scorer(matthews_corrcoef), param_grid=None, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_svm_model_P)

# Model with rank: 1
# Mean validation score: 0.880 (std: 0.012)
# Parameters: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#
# Model with rank: 2
# Mean validation score: 0.875 (std: 0.020)
# Parameters: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
#
# Model with rank: 3
# Mean validation score: 0.863 (std: 0.023)
# Parameters: {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#
# Best score svm (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.880
# Parameters:	{'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.852376 (0.016638) with: {'clf__C': 0.01, 'clf__kernel': 'linear'}
# 0.848865 (0.016069) with: {'clf__C': 0.1, 'clf__kernel': 'linear'}
# 0.849324 (0.017024) with: {'clf__C': 1.0, 'clf__kernel': 'linear'}
# 0.847313 (0.018398) with: {'clf__C': 10, 'clf__kernel': 'linear'}
# 0.757019 (0.021388) with: {'clf__C': 0.01, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.731416 (0.028036) with: {'clf__C': 0.01, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.000000 (0.000000) with: {'clf__C': 0.01, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.829406 (0.021525) with: {'clf__C': 0.1, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.820587 (0.028961) with: {'clf__C': 0.1, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.758923 (0.025227) with: {'clf__C': 0.1, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.863303 (0.022742) with: {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.855986 (0.019786) with: {'clf__C': 1.0, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.819306 (0.025486) with: {'clf__C': 1.0, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.879862 (0.012023) with: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.875078 (0.020041) with: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.848085 (0.020313) with: {'clf__C': 10, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
#
# print(report)
# precision    recall  f1-score   support
# 0       0.98      0.96      0.97      2451
# 1       0.89      0.94      0.92       817
# accuracy                           0.96      3268
# macro avg       0.94      0.95      0.94      3268
# weighted avg       0.96      0.96      0.96      3268
#
# {'Accuracy': 0.956548347613219,
#  'MCC': 0.886446826666222,
#  'f1 score': 0.9151732377538829,
#  'roc_auc': 0.9502243982048143,
#  'Precision': array([0.25      , 0.89381564, 1.        ]),
#  'Recall': array([1.       , 0.9375765, 0.       ]),
#  'fdr': 0.10618436406067679,
#  'sn': 0.9375764993880049,
#  'sp': 0.9628722970216238}
#

best_model = ml.train_best_model('rf', score=make_scorer(matthews_corrcoef), param_grid=None, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_model)

#
# Model with rank: 1
# Mean validation score: 0.902 (std: 0.014)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
#
# Model with rank: 2
# Mean validation score: 0.898 (std: 0.014)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# Model with rank: 3
# Mean validation score: 0.898 (std: 0.018)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 500}
#
# Best score rf (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.902
# Parameters:	{'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
# 0.890029 (0.016945) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 10}
# 0.897963 (0.013868) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.902170 (0.013630) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
# 0.883497 (0.013203) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 10}
# 0.894999 (0.015958) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 100}
# 0.897777 (0.018017) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 500}
#
# print(report)
# precision    recall  f1-score   support
# 0       0.98      0.96      0.97      2451
# 1       0.89      0.94      0.92       817
# accuracy                           0.96      3268
# macro avg       0.94      0.95      0.94      3268
# weighted avg       0.96      0.96      0.96      3268
#
# scores
# Out[101]:
# {'Accuracy': 0.956548347613219,
#  'MCC': 0.886446826666222,
#  'f1 score': 0.9151732377538829,
#  'roc_auc': 0.9502243982048143,
#  'Precision': array([0.25      , 0.89381564, 1.        ]),
#  'Recall': array([1.       , 0.9375765, 0.       ]),
#  'fdr': 0.10618436406067679,
#  'sn': 0.9375764993880049,
#  'sp': 0.9628722970216238}

# todo fazer o sns. correlation das features

# FEATURE SELECTION
X_train, X_test, y_train, y_test = train_test_split(fps_x_clean, fps_y, stratify=fps_y)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Select features
fsel = FeatureSelection(X_train, y_train, columns_names=fps_x_clean.columns)
# # Select from model L1
transformer, x_transf_model, column_selected, feat_impo, feat_impo_df = \
    fsel.run_from_model(model=LinearSVC(C=0.1, penalty="l1", dual=False))
# Finished 'run_from_model' in 34.2453 secs
# SCORES
# Out[105]:
# features importance
# GM                                     0.291445
# LM                                     0.285206
# AM                                     0.251052
# KM                                     0.205045
# SM                                     0.204688
# IM                                     0.195578
# NM                                     0.187421
# EM                                     0.187233
# APAAC12                                0.179748
# TM                                     0.178698
# DM                                     0.171085
# VM                                     0.164954
# _SolventAccessibilityD3001             0.164668
# PM                                     0.164262
# FM                                     0.156343
# RM                                     0.153948
# _SecondaryStrD3001                     0.149520
# CM                                     0.140805
# _HydrophobicityD3001                   0.140172
# _PolarityD1001                         0.128717
# Out[106]:
# features importance
# WP                                    -0.074238
# _SecondaryStrD2050                    -0.075793
# PAAC23                                -0.079597
# MA                                    -0.079989
# _PolarityD2075                        -0.086465
# _SecondaryStrD1025                    -0.087214
# _ChargeD1100                          -0.088213
# _HydrophobicityD2001                  -0.089549
# APAAC11                               -0.096364
# _ChargeD3100                          -0.098322
# _PolarityD1025                        -0.111471
# formulaS                              -0.111507
# _SolventAccessibilityD1050            -0.117247
# PAAC13                                -0.120637
# _PolarizabilityD2025                  -0.133008
# _PolarityD1050                        -0.143586
# _ChargeD2100                          -0.153805
# tot                                   -0.180208
# charge                                -0.182859
# M                                     -0.767044

# second time run
# features importance
# GM                             0.333754
# LM                             0.308609
# AM                             0.290799
# SM                             0.250635
# KM                             0.230467
# EM                             0.211120
# FM                             0.195855
# TM                             0.190094
# NM                             0.187896
# VM                             0.186067
# RM                             0.184531
# IM                             0.184517
# DM                             0.176614
# PM                             0.166780
# CM                             0.166427
# APAAC18                        0.155234
# MM                             0.145650
# _SecondaryStrD3001             0.134732
# YM                             0.130810
# QM                             0.128964
# Out[21]:
# features importance
# _SecondaryStrD2050                    -0.070021
# MK                                    -0.070847
# _PolarityD2075                        -0.071069
# formulaS                              -0.073504
# _SolventAccessibilityD1050            -0.074082
# _SecondaryStrD3050                    -0.075601
# MA                                    -0.076073
# _NormalizedVDWVD2025                  -0.076284
# _ChargeD1100                          -0.081787
# _SecondaryStrD1025                    -0.085201
# PAAC23                                -0.090372
# APAAC11                               -0.102876
# APAAC6                                -0.118372
# _PolarityD1050                        -0.128329
# PAAC13                                -0.140517
# _PolarizabilityD2025                  -0.152319
# charge                                -0.155028
# _ChargeT23                            -0.179967
# hydrogen                              -0.201124
# M                                     -0.880114


### SCORES 3º time running
# Finished 'run_from_model' in 29.2556 secs
X_train = x_transf_model
X_test = transformer.transform(X_test)

columns = fps_x_clean.columns[column_selected]  # 387 columns
ml = ShallowML(X_train, X_test, y_train, y_test, report_name=None, columns_names=columns)

best_model = ml.train_best_model('svm', score=make_scorer(matthews_corrcoef), param_grid=None, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_model)

# Model with rank: 1
# Mean validation score: 0.891 (std: 0.016)
# Parameters: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#
# Model with rank: 2
# Mean validation score: 0.890 (std: 0.016)
# Parameters: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
#
# Model with rank: 3
# Mean validation score: 0.879 (std: 0.020)
# Parameters: {'clf__C': 0.01, 'clf__kernel': 'linear'}
#
# Best score svm (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.891
# Parameters:	{'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.878577 (0.020237) with: {'clf__C': 0.01, 'clf__kernel': 'linear'}
# 0.871614 (0.021896) with: {'clf__C': 0.1, 'clf__kernel': 'linear'}
# 0.868689 (0.019707) with: {'clf__C': 1.0, 'clf__kernel': 'linear'}
# 0.866242 (0.020332) with: {'clf__C': 10, 'clf__kernel': 'linear'}
# 0.769209 (0.031258) with: {'clf__C': 0.01, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.730218 (0.029966) with: {'clf__C': 0.01, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.000000 (0.000000) with: {'clf__C': 0.01, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.834116 (0.026460) with: {'clf__C': 0.1, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.837130 (0.027620) with: {'clf__C': 0.1, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.725425 (0.023472) with: {'clf__C': 0.1, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.873842 (0.021198) with: {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.872312 (0.021467) with: {'clf__C': 1.0, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.837296 (0.025261) with: {'clf__C': 1.0, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.891394 (0.015909) with: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.889740 (0.016202) with: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.873897 (0.019586) with: {'clf__C': 10, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
#
# print(report)
# precision    recall  f1-score   support
# 0       0.98      0.96      0.97      2451
# 1       0.88      0.93      0.90       817
# accuracy                           0.95      3268
# macro avg       0.93      0.94      0.94      3268
# weighted avg       0.95      0.95      0.95      3268
#
# Out[14]:
# {'Accuracy': 0.950734394124847,
#  'MCC': 0.8718701327976202,
#  'f1 score': 0.9043374925727866,
#  'roc_auc': 0.9443084455324359,
#  'Precision': array([0.25      , 0.87875289, 1.        ]),
#  'Recall': array([1.        , 0.93145655, 0.        ]),
#  'fdr': 0.12124711316397228,
#  'sn': 0.9314565483476133,
#  'sp': 0.9571603427172583}


ml = ShallowML(X_train, X_test, y_train, y_test, report_name=None, columns_names=columns)
best_model = ml.train_best_model('rf', score=make_scorer(matthews_corrcoef), param_grid=None, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_model)

# Model with rank: 1
# Mean validation score: 0.902 (std: 0.021)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
#
# Model with rank: 2
# Mean validation score: 0.897 (std: 0.019)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# Model with rank: 3
# Mean validation score: 0.894 (std: 0.020)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 500}
#
# Best score rf (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.902
# Parameters:	{'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
# 0.879131 (0.021330) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 10}
# 0.896703 (0.019183) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.901798 (0.021232) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
# 0.868719 (0.025009) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 10}
# 0.893546 (0.020910) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 100}
# 0.893857 (0.020110) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 500}
#
# {'Accuracy': 0.9574663402692778,
#  'MCC': 0.8893408561623815,
#  'log_loss': 0.14321283191279438,
#  'f1 score': 0.9173111243307556,
#  'roc_auc': 0.9528763769889841,
#  'Precision': array([0.25      , 0.89236111, 1.        ]),
#  'Recall': array([1.        , 0.94369645, 0.        ]),
#  'fdr': 0.1076388888888889,
#  'sn': 0.9436964504283966,
#  'sp': 0.9620563035495716}
#
# print(report)
# precision    recall  f1-score   support
# 0       0.98      0.96      0.97      2451
# 1       0.89      0.94      0.92       817
# accuracy                           0.96      3268
# macro avg       0.94      0.95      0.94      3268
# weighted avg       0.96      0.96      0.96      3268
#

# RF features

X_train, X_test, y_train, y_test = train_test_split(fps_x_clean, fps_y, stratify=fps_y)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Select features
fsel = FeatureSelection(X_train, y_train, columns_names=fps_x_clean.columns)
transformer, x_transf_model, column_selected, feat_impo, feat_impo_df = \
    fsel.run_from_model(model=ExtraTreesClassifier(n_estimators=50))
# # Finished 'run_from_model' in 1.0152 secs
# features importance
# lenght                                 0.043828
# formulaC                               0.042714
# hydrogen                               0.039702
# MW_modlamp                             0.039057
# double                                 0.032453
# formulaN                               0.031361
# single                                 0.029621
# tot                                    0.019188
# formulaO                               0.017637
# APAAC5                                 0.014905
# _PolarityD1001                         0.014327
# formulaH                               0.014176
# PAAC5                                  0.013559
# _HydrophobicityD3001                   0.012797
# _SecondaryStrD1001                     0.010838
# M                                      0.010631
# _PolarityD2001                         0.010048
# C                                      0.008927
# _ChargeC3                              0.008197
# PAAC6                                  0.007893
# _PolarizabilityD3001                   0.007277
# _SolventAccessibilityD3001             0.006988
# D                                      0.006724
# CK                                     0.006424
# SecStruct_sheet                        0.006312
# _ChargeT23                             0.005969
# PAAC20                                 0.005392
# _ChargeD2001                           0.005239
# _SecondaryStrD3025                     0.005204
# chargedensity                          0.005108
# _SolventAccessibilityD3025             0.005031
# PAAC24                                 0.004914
# APAAC3                                 0.004790
# AR                                     0.004702
# _NormalizedVDWVC2                      0.004498
# _PolarityD2025                         0.004414
# APAAC6                                 0.004337
# MA                                     0.004044
# PAAC19                                 0.003791
# EV                                     0.003767
X_train = x_transf_model
X_test = transformer.transform(X_test)

columns = fps_x_clean.columns[column_selected]
ml = ShallowML(X_train, X_test, y_train, y_test, report_name=None, columns_names=columns)  # 108 columns

best_model = ml.train_best_model('svm', score=make_scorer(matthews_corrcoef), param_grid=None, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_model)
# Model with rank: 1
# Mean validation score: 0.884 (std: 0.012)
# Parameters: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#
# Model with rank: 2
# Mean validation score: 0.872 (std: 0.016)
# Parameters: {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#
# Model with rank: 3
# Mean validation score: 0.862 (std: 0.018)
# Parameters: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
#
# Best score svm (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.884
# Parameters:	{'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.839345 (0.022195) with: {'clf__C': 0.01, 'clf__kernel': 'linear'}
# 0.838379 (0.022525) with: {'clf__C': 0.1, 'clf__kernel': 'linear'}
# 0.843053 (0.019819) with: {'clf__C': 1.0, 'clf__kernel': 'linear'}
# 0.841431 (0.016230) with: {'clf__C': 10, 'clf__kernel': 'linear'}
# 0.798118 (0.019824) with: {'clf__C': 0.01, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.732773 (0.017351) with: {'clf__C': 0.01, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.000000 (0.000000) with: {'clf__C': 0.01, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.837658 (0.015041) with: {'clf__C': 0.1, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.820150 (0.014616) with: {'clf__C': 0.1, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.735952 (0.025150) with: {'clf__C': 0.1, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.872324 (0.016362) with: {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.853543 (0.017715) with: {'clf__C': 1.0, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.813153 (0.017892) with: {'clf__C': 1.0, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.884441 (0.012142) with: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.861726 (0.017621) with: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.845638 (0.019049) with: {'clf__C': 10, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
#
#
#
# print(report)
# precision    recall  f1-score   support
# 0       0.97      0.96      0.97      2451
# 1       0.89      0.92      0.90       817
# accuracy                           0.95      3268
# macro avg       0.93      0.94      0.94      3268
# weighted avg       0.95      0.95      0.95      3268
#
# {'Accuracy': 0.9513463892288861,
#  'MCC': 0.8721496683030968,
#  'f1 score': 0.9045045045045044,
#  'roc_auc': 0.941452468380253,
#  'Precision': array([0.25     , 0.8879717, 1.       ]),
#  'Recall': array([1.        , 0.92166463, 0.        ]),
#  'fdr': 0.11202830188679246,
#  'sn': 0.9216646266829865,
#  'sp': 0.9612403100775194}

ml = ShallowML(X_train, X_test, y_train, y_test, report_name=None, columns_names=columns)

best_model = ml.train_best_model('rf', score=make_scorer(matthews_corrcoef), param_grid=None, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_model)

#
# Model with rank: 1
# Mean validation score: 0.908 (std: 0.012)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
#
# Model with rank: 2
# Mean validation score: 0.906 (std: 0.014)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# Model with rank: 3
# Mean validation score: 0.903 (std: 0.013)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 500}
#
# Best score rf (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.908
# Parameters:	{'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
# 0.887549 (0.018255) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 10}
# 0.905741 (0.013860) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.907555 (0.012197) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
# 0.880893 (0.017705) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 10}
# 0.900509 (0.014708) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 100}
# 0.903476 (0.013108) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 500}
#

# precision    recall  f1-score   support
# 0       0.99      0.96      0.97      2451
# 1       0.90      0.96      0.93       817
# accuracy                           0.96      3268
# macro avg       0.94      0.96      0.95      3268
# weighted avg       0.96      0.96      0.96      3268
#
# Out[31]:
# {'Accuracy': 0.9620563035495716,
#  'MCC': 0.9019775388661223,
#  'log_loss': 0.13880877938340802,
#  'f1 score': 0.9266272189349113,
#  'roc_auc': 0.9608323133414933,
#  'Precision': array([0.25      , 0.89690722, 1.        ]),
#  'Recall': array([1.        , 0.95838433, 0.        ]),
#  'fdr': 0.10309278350515463,
#  'sn': 0.9583843329253366,
#  'sp': 0.9632802937576499}

#######################################################################################################################
# 1:1 ratio

AMP_data = pd.read_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/ampep_feat.csv')
AMP_data['label'] = 1
non_AMP_data = pd.read_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/non_ampep_feat.csv')
non_AMP_data['label'] = 0
non_AMP_data_1_1 = non_AMP_data.sample(n=AMP_data.shape[0], replace=False)

dataset = pd.concat([AMP_data, non_AMP_data_1_1])

fps_y = dataset['label']
fps_x = dataset.loc[:, dataset.columns != 'label']

fps_x_d = fps_x.filter(regex=r'_.+D\d', axis=1)  # select just CTD D feature 105 columns [6536 rows x 105 columns]
X_train, X_test, y_train, y_test = train_test_split(fps_x_d, fps_y, stratify=fps_y)
# standard scaler article does not refer scaling and do not validate in x_test
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

ml = ShallowML(X_train, X_test, y_train, y_test, report_name=None, columns_names=fps_x.columns)
param_grid = [{'clf__n_estimators': [100], 'clf__max_features': ['sqrt']}]

best_model = ml.train_best_model('rf', score=make_scorer(matthews_corrcoef), param_grid=param_grid, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_model)

# Model with rank: 1
# Mean validation score: 0.914 (std: 0.013)
# Parameters: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# Best score rf (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.914
# Parameters:	{'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.913885 (0.012532) with: {'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# clf__max_features  clf__n_estimators     means      stds
# 0              sqrt                100  0.913885  0.012532
#
# {'Accuracy': 0.9620563035495716,
#  'MCC': 0.9245115901594282,
#  'log_loss': 0.15446818061858514,
#  'f1 score': 0.962605548854041,
#  'roc_auc': 0.9620563035495717,
#  'Precision': array([0.5       , 0.94887039, 1.        ]),
#  'Recall': array([1.        , 0.97674419, 0.        ]),
#  'fdr': 0.05112960760998811,
#  'sn': 0.9767441860465116,
#  'sp': 0.9473684210526315}
#
# array([[774,  43],
#        [ 19, 798]])
#
# precision    recall  f1-score   support
# 0       0.98      0.95      0.96       817
# 1       0.95      0.98      0.96       817
# accuracy                           0.96      1634
# macro avg       0.96      0.96      0.96      1634
# weighted avg       0.96      0.96      0.96      1634
#


AMP_data = pd.read_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/ampep_feat_complete.csv')
AMP_data = AMP_data.drop({'Unnamed: 0'}, axis=1)
AMP_data['label'] = 1
non_AMP_data = pd.read_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/non_ampep_feat_complete.csv')
non_AMP_data['label'] = 0
non_AMP_data_1_1 = non_AMP_data.sample(n=AMP_data.shape[0], replace=False)

dataset = pd.concat([AMP_data, non_AMP_data_1_1])  # (13072, 642)

fps_y = dataset['label']
fps_x = dataset.drop(['label', 'sequence'], axis=1)
#  Preprocess
prepro = Preprocess()  # Create Preprocess object
fps_x_clean, columns_deleted = prepro.preprocess(fps_x, columns_names=True, threshold=0,
                                                 standard=True)  # [13072 rows x 628 columns]

# NO FEATURE SELECTION
X_train, X_test, y_train, y_test = train_test_split(fps_x_clean, fps_y, stratify=fps_y)
# standard scaler article does not refer scaling and do not validate in x_test
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
ml = ShallowML(X_train, X_test, y_train, y_test, report_name=None, columns_names=fps_x_clean.columns)

best_svm_model_AMPEP = ml.train_best_model('svm', score=make_scorer(matthews_corrcoef), param_grid=None, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_svm_model_AMPEP)

# Model with rank: 1
# Mean validation score: 0.902 (std: 0.025)
# Parameters: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#
# Model with rank: 2
# Mean validation score: 0.896 (std: 0.030)
# Parameters: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
#
# Model with rank: 3
# Mean validation score: 0.881 (std: 0.027)
# Parameters: {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#
# Best score svm (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.902
# Parameters:	{'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.847321 (0.022099) with: {'clf__C': 0.01, 'clf__kernel': 'linear'}
# 0.844959 (0.019086) with: {'clf__C': 0.1, 'clf__kernel': 'linear'}
# 0.829576 (0.020374) with: {'clf__C': 1.0, 'clf__kernel': 'linear'}
# 0.815005 (0.018291) with: {'clf__C': 10, 'clf__kernel': 'linear'}
# 0.780379 (0.028905) with: {'clf__C': 0.01, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.784485 (0.029161) with: {'clf__C': 0.01, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.356487 (0.186618) with: {'clf__C': 0.01, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.838943 (0.033382) with: {'clf__C': 0.1, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.843499 (0.037795) with: {'clf__C': 0.1, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.797644 (0.030186) with: {'clf__C': 0.1, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.880952 (0.026961) with: {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.878177 (0.034535) with: {'clf__C': 1.0, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.856718 (0.034823) with: {'clf__C': 1.0, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.901976 (0.024524) with: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.896033 (0.029624) with: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.861330 (0.033157) with: {'clf__C': 10, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# clf__C clf__kernel clf__gamma     means      stds
# 0     0.01      linear        NaN  0.847321  0.022099
# 1     0.10      linear        NaN  0.844959  0.019086
# 2     1.00      linear        NaN  0.829576  0.020374
# 3    10.00      linear        NaN  0.815005  0.018291
# 4     0.01         rbf      scale  0.780379  0.028905
# 5     0.01         rbf      0.001  0.784485  0.029161
# 6     0.01         rbf     0.0001  0.356487  0.186618
# 7     0.10         rbf      scale  0.838943  0.033382
# 8     0.10         rbf      0.001  0.843499  0.037795
# 9     0.10         rbf     0.0001  0.797644  0.030186
# 10    1.00         rbf      scale  0.880952  0.026961
# 11    1.00         rbf      0.001  0.878177  0.034535
# 12    1.00         rbf     0.0001  0.856718  0.034823
# 13   10.00         rbf      scale  0.901976  0.024524
# 14   10.00         rbf      0.001  0.896033  0.029624
# 15   10.00         rbf     0.0001  0.861330  0.033157
#
# print(report)
# precision    recall  f1-score   support
# 0       0.96      0.94      0.95       817
# 1       0.94      0.97      0.95       817
# accuracy                           0.95      1634
# macro avg       0.95      0.95      0.95      1634
# weighted avg       0.95      0.95      0.95      1634
#
# array([[770,  47],
#        [ 28, 789]])
#
# {'Accuracy': 0.9541003671970624,
#  'MCC': 0.9084464264679808,
#  'f1 score': 0.9546279491833031,
#  'roc_auc': 0.9541003671970625,
#  'Precision': array([0.5      , 0.9437799, 1.       ]),
#  'Recall': array([1.        , 0.96572827, 0.        ]),
#  'fdr': 0.056220095693779906,
#  'sn': 0.9657282741738066,
#  'sp': 0.9424724602203183}

best_rf_model_AMPEP = ml.train_best_model('rf', score=make_scorer(matthews_corrcoef), param_grid=None, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_rf_model_AMPEP)
# #
# Model with rank: 1
# Mean validation score: 0.921 (std: 0.013)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# Model with rank: 2
# Mean validation score: 0.920 (std: 0.014)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
#
# Model with rank: 3
# Mean validation score: 0.910 (std: 0.017)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 100}
#
# Best score rf (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.921
# Parameters:	{'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.905143 (0.016119) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 10}
# 0.920911 (0.013332) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.920168 (0.014117) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
# 0.884251 (0.017784) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 10}
# 0.909814 (0.016561) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 100}
# 0.908954 (0.016965) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 500}
# clf__bootstrap clf__criterion  ...     means      stds
# 0            True           gini  ...  0.905143  0.016119
# 1            True           gini  ...  0.920911  0.013332
# 2            True           gini  ...  0.920168  0.014117
# 3            True           gini  ...  0.884251  0.017784
# 4            True           gini  ...  0.909814  0.016561
# 5            True           gini  ...  0.908954  0.016965
# [6 rows x 6 columns]
# print(report )
# precision    recall  f1-score   support
# 0       0.98      0.95      0.96       817
# 1       0.95      0.98      0.96       817
# accuracy                           0.96      1634
# macro avg       0.96      0.96      0.96      1634
# weighted avg       0.96      0.96      0.96      1634
#
# array([[777,  40],
#        [ 18, 799]])
# Out[160]:
# {'Accuracy': 0.9645042839657283,
#  'MCC': 0.9293455658611339,
#  'log_loss': 0.11504519728199614,
#  'f1 score': 0.9649758454106281,
#  'roc_auc': 0.9645042839657283,
#  'Precision': array([0.5      , 0.9523242, 1.       ]),
#  'Recall': array([1.        , 0.97796818, 0.        ]),
#  'fdr': 0.04767580452920143,
#  'sn': 0.97796817625459,
#  'sp': 0.9510403916768666}

# FEATURE SELECTION
X_train, X_test, y_train, y_test = train_test_split(fps_x_clean, fps_y, stratify=fps_y)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# Select features
fsel = FeatureSelection(X_train, y_train, columns_names=fps_x_clean.columns)
# # KBest com *mutual info classif*
transf, X_fit_uni, X_transf_uni, column_selected, scores, scores_df = \
    fsel.run_univariate(score_func=mutual_info_classif, mode='k_best', param=300)
# Finished 'run_univariate' in 29.1549 secs
# SCORES
# scores_ranking
# _HydrophobicityD3001              0.479588
# _PolarityD1001                    0.473949
# _SolventAccessibilityD3001        0.462259
# _SecondaryStrD1001                0.442256
# _PolarizabilityD3001              0.431457
# lenght                            0.406796
# tot                               0.405570
# _ChargeD2001                      0.405177
# hydrogen                          0.403018
# formulaH                          0.398977
# single                            0.398561
# formulaO                          0.396241
# formulaC                          0.394737
# MW_modlamp                        0.392755
# formulaN                          0.384796
# double                            0.373149
# M                                 0.367505
# charge                            0.332110
# _NormalizedVDWVD2001              0.313884
# SecStruct_sheet                   0.307550
# _ChargeD3025                      0.302219
# _ChargeD3001                      0.298827
# E                                 0.298060
# APAAC13                           0.297165
# _ChargeD3075                      0.295952
# _ChargeD3050                      0.295918
# _ChargeD3100                      0.293265
# _SolventAccessibilityD2001        0.291275
# C                                 0.290205
# PAAC13                            0.287135

X_train = X_transf_uni
X_test = transf.transform(X_test)
columns = fps_x_clean.columns[column_selected]
ml = ShallowML(X_train, X_test, y_train, y_test, report_name=None, columns_names=columns)

best_svm_model_P = ml.train_best_model('svm', score=make_scorer(matthews_corrcoef), param_grid=None, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_svm_model_P)
#
# Model with rank: 1
# Mean validation score: 0.897 (std: 0.011)
# Parameters: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#
# Model with rank: 2
# Mean validation score: 0.872 (std: 0.017)
# Parameters: {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#
# Model with rank: 3
# Mean validation score: 0.870 (std: 0.021)
# Parameters: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
#
# Best score svm (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.897
# Parameters:	{'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.844639 (0.017755) with: {'clf__C': 0.01, 'clf__kernel': 'linear'}
# 0.843957 (0.016079) with: {'clf__C': 0.1, 'clf__kernel': 'linear'}
# 0.843981 (0.015277) with: {'clf__C': 1.0, 'clf__kernel': 'linear'}
# 0.844400 (0.017632) with: {'clf__C': 10, 'clf__kernel': 'linear'}
# 0.766939 (0.025386) with: {'clf__C': 0.01, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.771748 (0.022061) with: {'clf__C': 0.01, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.527204 (0.264207) with: {'clf__C': 0.01, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.836768 (0.019156) with: {'clf__C': 0.1, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.827234 (0.015140) with: {'clf__C': 0.1, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.770813 (0.017723) with: {'clf__C': 0.1, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.872132 (0.016737) with: {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.861672 (0.017811) with: {'clf__C': 1.0, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.837899 (0.015880) with: {'clf__C': 1.0, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.896689 (0.011366) with: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.869771 (0.020747) with: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.853763 (0.020220) with: {'clf__C': 10, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# clf__C clf__kernel clf__gamma     means      stds
# 0     0.01      linear        NaN  0.844639  0.017755
# 1     0.10      linear        NaN  0.843957  0.016079
# 2     1.00      linear        NaN  0.843981  0.015277
# 3    10.00      linear        NaN  0.844400  0.017632
# 4     0.01         rbf      scale  0.766939  0.025386
# 5     0.01         rbf      0.001  0.771748  0.022061
# 6     0.01         rbf     0.0001  0.527204  0.264207
# 7     0.10         rbf      scale  0.836768  0.019156
# 8     0.10         rbf      0.001  0.827234  0.015140
# 9     0.10         rbf     0.0001  0.770813  0.017723
# 10    1.00         rbf      scale  0.872132  0.016737
# 11    1.00         rbf      0.001  0.861672  0.017811
# 12    1.00         rbf     0.0001  0.837899  0.015880
# 13   10.00         rbf      scale  0.896689  0.011366
# 14   10.00         rbf      0.001  0.869771  0.020747
# 15   10.00         rbf     0.0001  0.853763  0.020220
#
# precision    recall  f1-score   support
# 0       0.96      0.95      0.95       817
# 1       0.95      0.96      0.95       817
# accuracy                           0.95      1634
# macro avg       0.95      0.95      0.95      1634
# weighted avg       0.95      0.95      0.95      1634
#
# ut[168]:
# array([[775,  42],
#        [ 32, 785]])
# {'Accuracy': 0.9547123623011016,
#  'MCC': 0.9094928550823758,
#  'f1 score': 0.9549878345498783,
#  'roc_auc': 0.9547123623011015,
#  'Precision': array([0.5       , 0.94921403, 1.        ]),
#  'Recall': array([1.        , 0.96083231, 0.        ]),
#  'fdr': 0.05078597339782346,
#  'sn': 0.9608323133414932,
#  'sp': 0.9485924112607099}
#

best_model = ml.train_best_model('rf', score=make_scorer(matthews_corrcoef), param_grid=None, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_model)

# Model with rank: 1
# Mean validation score: 0.927 (std: 0.023)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
#
# Model with rank: 2
# Mean validation score: 0.927 (std: 0.022)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# Model with rank: 3
# Mean validation score: 0.924 (std: 0.017)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 500}
#
# Best score rf (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.927
# Parameters:	{'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
# 0.912063 (0.018239) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 10}
# 0.927116 (0.022356) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.927250 (0.022570) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
# 0.910866 (0.011914) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 10}
# 0.919077 (0.017444) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 100}
# 0.924282 (0.016925) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 500}
# clf__bootstrap clf__criterion  ...     means      stds
# 0            True           gini  ...  0.912063  0.018239
# 1            True           gini  ...  0.927116  0.022356
# 2            True           gini  ...  0.927250  0.022570
# 3            True           gini  ...  0.910866  0.011914
# 4            True           gini  ...  0.919077  0.017444
# 5            True           gini  ...  0.924282  0.016925
# [6 rows x 6 columns]
#
# {'Accuracy': 0.9632802937576499,
#  'MCC': 0.9267383191045214,
#  'log_loss': 0.10914790423213959,
#  'f1 score': 0.9636363636363636,
#  'roc_auc': 0.96328029375765,
#  'Precision': array([0.5       , 0.95438175, 1.        ]),
#  'Recall': array([1.        , 0.97307222, 0.        ]),
#  'fdr': 0.04561824729891957,
#  'sn': 0.9730722154222766,
#  'sp': 0.9534883720930233}
#
# array([[779,  38],
#        [ 22, 795]])
#
# print(report)
# precision    recall  f1-score   support
# 0       0.97      0.95      0.96       817
# 1       0.95      0.97      0.96       817
# accuracy                           0.96      1634
# macro avg       0.96      0.96      0.96      1634
# weighted avg       0.96      0.96      0.96      1634

# FEATURE SELECTION
X_train, X_test, y_train, y_test = train_test_split(fps_x_clean, fps_y, stratify=fps_y)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Select features
fsel = FeatureSelection(X_train, y_train, columns_names=fps_x_clean.columns)
# # Select from model L1
transformer, x_transf_model, column_selected, feat_impo, feat_impo_df = \
    fsel.run_from_model(model=LinearSVC(C=0.1, penalty="l1", dual=False))
# Finished 'run_from_model' in 16.0663 secs
# SCORES
# features importance
# GM                                     0.399803
# AM                                     0.328433
# LM                                     0.323108
# APAAC5                                 0.285788
# TM                                     0.279391
# _SolventAccessibilityD3001             0.273939
# _SecondaryStrD3001                     0.266737
# SM                                     0.246960
# KM                                     0.244477
# IM                                     0.222164
# EM                                     0.220026
# WC                                     0.194036
# NM                                     0.193728
# _SecondaryStrD1100                     0.193148
# FM                                     0.190788
# DM                                     0.186888
# _NormalizedVDWVD2001                   0.185533
# RM                                     0.183633
# _SolventAccessibilityD2025             0.165162
# PM                                     0.158717
# WM                                     0.154321
# VM                                     0.152874
# MM                                     0.147883
# PAAC27                                 0.145837
# CM                                     0.144942
# _HydrophobicityD3001                   0.142828
# _ChargeD2001                           0.135838
# NA                                     0.134302
# YM                                     0.133486
# RC                                     0.126569
# features importance
# _PolarityD2075                        -0.072622
# HF                                    -0.073396
# IV                                    -0.073923
# _NormalizedVDWVD1100                  -0.075033
# _ChargeD1075                          -0.077308
# WS                                    -0.077731
# WF                                    -0.077984
# MA                                    -0.081450
# _NormalizedVDWVD2050                  -0.085112
# LI                                    -0.085497
# PR                                    -0.085759
# AE                                    -0.085769
# _ChargeD3100                          -0.089309
# _SolventAccessibilityD1025            -0.104667
# AR                                    -0.110920
# _SecondaryStrD2050                    -0.112128
# _SolventAccessibilityD3100            -0.112272
# SS                                    -0.113624
# APAAC6                                -0.115274
# PAAC23                                -0.123123
# formulaO                              -0.127888
# SecStruct_helix                       -0.129225
# _SolventAccessibilityD2075            -0.130729
# _SecondaryStrD1025                    -0.139137
# _SolventAccessibilityD1050            -0.144552
# charge                                -0.162227
# formulaS                              -0.163011
# _PolarizabilityD2025                  -0.212323
# PAAC13                                -0.320672
# M                                     -0.838271

X_train = x_transf_model
X_test = transformer.transform(X_test)

columns = fps_x_clean.columns[column_selected]  # 345 columns
ml = ShallowML(X_train, X_test, y_train, y_test, report_name=None, columns_names=columns)

best_model = ml.train_best_model('svm', score=make_scorer(matthews_corrcoef), param_grid=None, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_model)
# Model with rank: 1
# Mean validation score: 0.903 (std: 0.017)
# Parameters: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
#
# Model with rank: 2
# Mean validation score: 0.901 (std: 0.021)
# Parameters: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#
# Model with rank: 3
# Mean validation score: 0.880 (std: 0.027)
# Parameters: {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#
# Best score svm (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.903
# Parameters:	{'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.874369 (0.019389) with: {'clf__C': 0.01, 'clf__kernel': 'linear'}
# 0.875005 (0.017117) with: {'clf__C': 0.1, 'clf__kernel': 'linear'}
# 0.868429 (0.026427) with: {'clf__C': 1.0, 'clf__kernel': 'linear'}
# 0.857834 (0.018330) with: {'clf__C': 10, 'clf__kernel': 'linear'}
# 0.780804 (0.043261) with: {'clf__C': 0.01, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.758853 (0.032457) with: {'clf__C': 0.01, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.067652 (0.069405) with: {'clf__C': 0.01, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.840240 (0.033871) with: {'clf__C': 0.1, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.839304 (0.028593) with: {'clf__C': 0.1, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.767847 (0.021182) with: {'clf__C': 0.1, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.880144 (0.026975) with: {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.870273 (0.026436) with: {'clf__C': 1.0, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.847598 (0.022090) with: {'clf__C': 1.0, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.900652 (0.021225) with: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.902975 (0.016974) with: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.866506 (0.023694) with: {'clf__C': 10, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# clf__C clf__kernel clf__gamma     means      stds
# 0     0.01      linear        NaN  0.874369  0.019389
# 1     0.10      linear        NaN  0.875005  0.017117
# 2     1.00      linear        NaN  0.868429  0.026427
# 3    10.00      linear        NaN  0.857834  0.018330
# 4     0.01         rbf      scale  0.780804  0.043261
# 5     0.01         rbf      0.001  0.758853  0.032457
# 6     0.01         rbf     0.0001  0.067652  0.069405
# 7     0.10         rbf      scale  0.840240  0.033871
# 8     0.10         rbf      0.001  0.839304  0.028593
# 9     0.10         rbf     0.0001  0.767847  0.021182
# 10    1.00         rbf      scale  0.880144  0.026975
# 11    1.00         rbf      0.001  0.870273  0.026436
# 12    1.00         rbf     0.0001  0.847598  0.022090
# 13   10.00         rbf      scale  0.900652  0.021225
# 14   10.00         rbf      0.001  0.902975  0.016974
# 15   10.00         rbf     0.0001  0.866506  0.023694
#
# precision    recall  f1-score   support
# 0       0.92      0.95      0.94       817
# 1       0.95      0.92      0.94       817
# accuracy                           0.94      1634
# macro avg       0.94      0.94      0.94      1634
# weighted avg       0.94      0.94      0.94      1634
# array([[778,  39],
#        [ 64, 753]])
# {'Accuracy': 0.9369645042839657,
#  'MCC': 0.8743384456579749,
#  'f1 score': 0.9359850839030454,
#  'roc_auc': 0.9369645042839657,
#  'Precision': array([0.5       , 0.95075758, 1.        ]),
#  'Recall': array([1.        , 0.92166463, 0.        ]),
#  'fdr': 0.04924242424242424,
#  'sn': 0.9216646266829865,
#  'sp': 0.9522643818849449}


ml = ShallowML(X_train, X_test, y_train, y_test, report_name=None, columns_names=columns)
best_model = ml.train_best_model('rf', score=make_scorer(matthews_corrcoef), param_grid=None, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_model)
# Model with rank: 1
# Mean validation score: 0.921 (std: 0.014)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
#
# Model with rank: 2
# Mean validation score: 0.919 (std: 0.014)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
#
# Model with rank: 3
# Mean validation score: 0.913 (std: 0.015)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 500}
#
# Best score rf (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.921
# Parameters:	{'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
# 0.907689 (0.018076) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 10}
# 0.919222 (0.014294) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.920512 (0.014209) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
# 0.897843 (0.020998) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 10}
# 0.913179 (0.016912) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 100}
# 0.913192 (0.014860) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 500}
# clf__bootstrap clf__criterion  ...     means      stds
# 0            True           gini  ...  0.907689  0.018076
# 1            True           gini  ...  0.919222  0.014294
# 2            True           gini  ...  0.920512  0.014209
# 3            True           gini  ...  0.897843  0.020998
# 4            True           gini  ...  0.913179  0.016912
# 5            True           gini  ...  0.913192  0.014860
# [6 rows x 6 columns]
# {'Accuracy': 0.9632802937576499,
#  'MCC': 0.9265855748412875,
#  'log_loss': 0.128885879746879,
#  'f1 score': 0.9634146341463415,
#  'roc_auc': 0.9632802937576499,
#  'Precision': array([0.5       , 0.95990279, 1.        ]),
#  'Recall': array([1.        , 0.96695226, 0.        ]),
#  'fdr': 0.040097205346294046,
#  'sn': 0.966952264381885,
#  'sp': 0.9596083231334149}
#
# array([[784,  33],
#        [ 27, 790]])
#
# precision    recall  f1-score   support
# 0       0.97      0.96      0.96       817
# 1       0.96      0.97      0.96       817
# accuracy                           0.96      1634
# macro avg       0.96      0.96      0.96      1634
# weighted avg       0.96      0.96      0.96      1634
# RF features

X_train, X_test, y_train, y_test = train_test_split(fps_x_clean, fps_y, stratify=fps_y)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Select features
fsel = FeatureSelection(X_train, y_train, columns_names=fps_x_clean.columns)
transformer, x_transf_model, column_selected, feat_impo, feat_impo_df = \
    fsel.run_from_model(model=ExtraTreesClassifier(n_estimators=50))
# # Finished 'run_from_model' in 1.8745 secs
# features importance
# features importance
# formulaC                               0.070737
# tot                                    0.070591
# formulaH                               0.039938
# lenght                                 0.033210
# single                                 0.029072
# MW_modlamp                             0.027918
# formulaO                               0.025642
# hydrogen                               0.023004
# formulaN                               0.020017
# APAAC5                                 0.017041
# PAAC5                                  0.016932
# _ChargeC3                              0.015883
# _ChargeT23                             0.012531
# APAAC4                                 0.011708
# double                                 0.010916
# AE                                     0.010184
# PAAC6                                  0.008971
# _SolventAccessibilityD3001             0.008349
# _PolarityD1001                         0.007863
# _HydrophobicityD3001                   0.007589
# APAAC6                                 0.007294
# PAAC4                                  0.006462
# _PolarityT23                           0.006210
# aliphatic_index                        0.005877
# _SecondaryStrD1001                     0.005725
# E                                      0.005559
# _NormalizedVDWVC2                      0.005450
# _PolarityD2025                         0.005337
# PAAC13                                 0.005117
# _ChargeD2001                           0.004761
X_train = x_transf_model
X_test = transformer.transform(X_test)

columns = fps_x_clean.columns[column_selected]
ml = ShallowML(X_train, X_test, y_train, y_test, report_name=None, columns_names=columns)  # 96 columns

best_model = ml.train_best_model('svm', score=make_scorer(matthews_corrcoef), param_grid=None, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_model)

# Model with rank: 1
# Mean validation score: 0.903 (std: 0.018)
# Parameters: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#
# Model with rank: 2
# Mean validation score: 0.881 (std: 0.018)
# Parameters: {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#
# Model with rank: 3
# Mean validation score: 0.877 (std: 0.016)
# Parameters: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
#
# Best score svm (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.903
# Parameters:	{'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.859826 (0.016561) with: {'clf__C': 0.01, 'clf__kernel': 'linear'}
# 0.861900 (0.014455) with: {'clf__C': 0.1, 'clf__kernel': 'linear'}
# 0.871231 (0.012422) with: {'clf__C': 1.0, 'clf__kernel': 'linear'}
# 0.874127 (0.012379) with: {'clf__C': 10, 'clf__kernel': 'linear'}
# 0.808132 (0.017763) with: {'clf__C': 0.01, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.796388 (0.019461) with: {'clf__C': 0.01, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.623433 (0.312312) with: {'clf__C': 0.01, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.853692 (0.018935) with: {'clf__C': 0.1, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.831210 (0.021060) with: {'clf__C': 0.1, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.791731 (0.021393) with: {'clf__C': 0.1, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.881151 (0.018413) with: {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.863816 (0.019459) with: {'clf__C': 1.0, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.828923 (0.019791) with: {'clf__C': 1.0, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# 0.902637 (0.018134) with: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
# 0.876569 (0.016258) with: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
# 0.860508 (0.015751) with: {'clf__C': 10, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
# clf__C clf__kernel clf__gamma     means      stds
# 0     0.01      linear        NaN  0.859826  0.016561
# 1     0.10      linear        NaN  0.861900  0.014455
# 2     1.00      linear        NaN  0.871231  0.012422
# 3    10.00      linear        NaN  0.874127  0.012379
# 4     0.01         rbf      scale  0.808132  0.017763
# 5     0.01         rbf      0.001  0.796388  0.019461
# 6     0.01         rbf     0.0001  0.623433  0.312312
# 7     0.10         rbf      scale  0.853692  0.018935
# 8     0.10         rbf      0.001  0.831210  0.021060
# 9     0.10         rbf     0.0001  0.791731  0.021393
# 10    1.00         rbf      scale  0.881151  0.018413
# 11    1.00         rbf      0.001  0.863816  0.019459
# 12    1.00         rbf     0.0001  0.828923  0.019791
# 13   10.00         rbf      scale  0.902637  0.018134
# 14   10.00         rbf      0.001  0.876569  0.016258
# 15   10.00         rbf     0.0001  0.860508  0.015751
#
# print(report)
# precision    recall  f1-score   support
# 0       0.96      0.93      0.94       817
# 1       0.93      0.96      0.94       817
# accuracy                           0.94      1634
# macro avg       0.94      0.94      0.94      1634
# weighted avg       0.94      0.94      0.94      1634
#
# array([[757,  60],
#        [ 33, 784]])
#
# {'Accuracy': 0.9430844553243574,
#  'MCC': 0.8866532233340255,
#  'f1 score': 0.9440096327513546,
#  'roc_auc': 0.9430844553243574,
#  'Precision': array([0.5       , 0.92890995, 1.        ]),
#  'Recall': array([1.        , 0.95960832, 0.        ]),
#  'fdr': 0.07109004739336493,
#  'sn': 0.9596083231334149,
#  'sp': 0.9265605875152999}

ml = ShallowML(X_train, X_test, y_train, y_test, report_name=None, columns_names=columns)
best_model = ml.train_best_model('rf', score=make_scorer(matthews_corrcoef), param_grid=None, cv=10)
scores, report, cm, cm2 = ml.score_testset(best_model)

# Model with rank: 1
# Mean validation score: 0.933 (std: 0.018)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 100}
#
# Model with rank: 2
# Mean validation score: 0.931 (std: 0.020)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
#
# Model with rank: 3
# Mean validation score: 0.931 (std: 0.018)
# Parameters: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 500}
#
# Best score rf (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# MCC score:	0.933
# Parameters:	{'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 100}
# 0.919317 (0.016579) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 10}
# 0.930281 (0.018288) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}
# 0.930801 (0.020253) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'sqrt', 'clf__n_estimators': 500}
# 0.915676 (0.021058) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 10}
# 0.932737 (0.017984) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 100}
# 0.930753 (0.017945) with: {'clf__bootstrap': True, 'clf__criterion': 'gini', 'clf__max_features': 'log2', 'clf__n_estimators': 500}
# clf__bootstrap clf__criterion  ...     means      stds
# 0            True           gini  ...  0.919317  0.016579
# 1            True           gini  ...  0.930281  0.018288
# 2            True           gini  ...  0.930801  0.020253
# 3            True           gini  ...  0.915676  0.021058
# 4            True           gini  ...  0.932737  0.017984
# 5            True           gini  ...  0.930753  0.017945
# [6 rows x 6 columns]
# print(report)
# precision    recall  f1-score   support
# 0       0.97      0.93      0.95       817
# 1       0.93      0.97      0.95       817
# accuracy                           0.95      1634
# macro avg       0.95      0.95      0.95      1634
# weighted avg       0.95      0.95      0.95      1634
#
# Out[210]:
# array([[761,  56],
#        [ 23, 794]])
#
# {'Accuracy': 0.9516523867809058,
#  'MCC': 0.9040425416797966,
#  'log_loss': 0.18482891317028868,
#  'f1 score': 0.9526094781043791,
#  'roc_auc': 0.9516523867809059,
#  'Precision': array([0.5       , 0.93411765, 1.        ]),
#  'Recall': array([1.        , 0.97184823, 0.        ]),
#  'fdr': 0.06588235294117648,
#  'sn': 0.9718482252141983,
#  'sp': 0.9314565483476133}
#

# try deep learning

AMP_data = pd.read_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/ampep_feat_complete.csv')
AMP_data = AMP_data.drop({'Unnamed: 0'}, axis=1)
AMP_data['label'] = 1
non_AMP_data_1_3 = pd.read_csv(
    '/home/amsequeira/propythia/propythia/example/AMP/datasets/non_ampep_feat_complete.csv')
non_AMP_data_1_3['label'] = 0

dataset = pd.concat([AMP_data, non_AMP_data_1_3])  # (13072, 642)

fps_y = dataset['label']
fps_x = dataset.drop(['label', 'sequence'], axis=1)
#  Preprocess
prepro = Preprocess()  # Create Preprocess object
fps_x_clean, columns_deleted = prepro.preprocess(fps_x, columns_names=True, threshold=0,
                                                 standard=True)  # [13072 rows x 628 columns]

# NO FEATURE SELECTION
X_train, X_test, y_train, y_test = train_test_split(fps_x_clean, fps_y, stratify=fps_y)
# standard scaler article does not refer scaling and do not validate in x_test
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# non_AMP_data_1_1 = non_AMP_data.sample(n=AMP_data.shape[0], replace=False)

dl = DeepML(X_train, y_train, X_test, y_test, number_classes=2, problem_type='binary',
            x_dval=None, y_dval=None, epochs=500, batch_size=512,
            path='/home/amsequeira/propythia/propythia/example/AMP', report_name='ampep', verbose=1)

dnn = dl.run_dnn_simple(
    input_dim=X_train.shape[1],
    optimizer='Adam',
    hidden_layers=(128, 64),
    dropout_rate=(0.3,),
    batchnormalization=(True,),
    l1=1e-5, l2=1e-4,
    final_dropout_value=0.3,
    initial_dropout_value=0.0,
    loss_fun=None, activation_fun=None,
    cv=10, optType='randomizedSearch', param_grid=None, n_iter_search=15, n_jobs=1,
    scoring=make_scorer(matthews_corrcoef))

# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy', 'lr'])
# ('Training Accuracy mean: ', 0.9830462055073844)
# ('Validation Accuracy mean: ', 0.9467804945177503)
# ('Training Loss mean: ', 0.15677526779472828)
# ('Validation Loss mean: ', 0.33493885232342613)
# Model: "sequential_531"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense_1577 (Dense)           (None, 64)                40256
# _________________________________________________________________
# batch_normalization_1046 (Ba (None, 64)                256
# _________________________________________________________________
# dropout_1046 (Dropout)       (None, 64)                0
# _________________________________________________________________
# dense_1578 (Dense)           (None, 1)                 65
#                                                        =================================================================
# Total params: 40,577
# Trainable params: 40,449
# Non-trainable params: 128
# _________________________________________________________________
# [['Model with rank: 1\n', 'Mean validation score: 0.762 (std: 0.020)\n', "Parameters: {'l2': 0, 'l1': 0.0001, 'hidden_layers': (64,), 'dropout_rate': 0.35}\n", '\n'], ['Model with rank: 2\n', 'Mean validation score: 0.755 (std: 0.045)\n', "Parameters: {'l2': 0.0001, 'l1': 0, 'hidden_layers': (64,), 'dropout_rate': 0.25}\n", '\n'], ['Model with rank: 3\n', 'Mean validation score: 0.754 (std: 0.031)\n', "Parameters: {'l2': 0.0001, 'l1': 1e-05, 'hidden_layers': (64,), 'dropout_rate': 0.35}\n", '\n']]
# Best score (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# df
# means      stds       l2       l1  hidden_layers  dropout_rate
# 3   0.762421  0.020273  0.00000  0.00010          (64,)          0.35
# 12  0.754631  0.044862  0.00010  0.00000          (64,)          0.25
# 10  0.754242  0.031282  0.00010  0.00001          (64,)          0.35
# 2   0.748781  0.032160  0.00100  0.00100          (64,)          0.25
# 8   0.747967  0.032983  0.00100  0.00001          (64,)          0.25
# 7   0.738935  0.025555  0.00000  0.00001          (64,)          0.50
# 6   0.601430  0.097305  0.00100  0.00000       (64, 32)          0.20
# 1   0.594943  0.097451  0.00001  0.00001       (64, 32)          0.40
# 11  0.573989  0.077354  0.00000  0.00000       (64, 32)          0.25
# 13  0.534826  0.071987  0.00001  0.00100      (128, 64)          0.50
# 14  0.531790  0.037323  0.00000  0.00010      (128, 64)          0.20
# 9   0.483757  0.066665  0.00001  0.00010  (128, 64, 32)          0.25
# 5   0.444696  0.108971  0.00001  0.00001  (128, 64, 32)          0.25
# 4   0.441913  0.072444  0.00001  0.00010  (128, 64, 32)          0.30
# 0   0.437849  0.096189  0.00100  0.00010  (128, 64, 32)          0.20

scores, report, cm, cm2 = dl.model_complete_evaluate()

# === Confusion Matrix ===
# [[2389   62]
#  [  69  748]]
# === Classification Report ===
# precision    recall  f1-score   support
# 0       0.97      0.97      0.97      2451
# 1       0.92      0.92      0.92       817
# accuracy                           0.96      3268
# macro avg       0.95      0.95      0.95      3268
# weighted avg       0.96      0.96      0.96      3268
# metrics                           scores
# 0   Accuracy                         0.959914
# 1        MCC                         0.892813
# 2   log_loss                         0.204321
# 3   f1 score                         0.919484
# 4    roc_auc                         0.945124
# 5  Precision  [0.25, 0.9234567901234568, 1.0]
# 6     Recall   [1.0, 0.9155446756425949, 0.0]
# https://www.omnicalculator.com/statistics/sensitivity-and-specificity
dl = DeepML(X_train, y_train, X_test, y_test, number_classes=2, problem_type='binary',
            x_dval=None, y_dval=None, epochs=500, batch_size=512,
            path='/home/amsequeira/propythia/propythia/example/AMP', report_name='ampep', verbose=1)

dnn_emb= dl.run_dnn_embedding(input_dim=X_train.shape[1],
                           optimizer='Adam',
                           input_dim_emb=21, output_dim=256, input_length=X_train.shape[1], mask_zero=True,
                           hidden_layers=(128, 64),
                           dropout_rate=0.3,
                           batchnormalization=True,
                           l1=1e-5, l2=1e-4,
                           final_dropout_value=0.3,
                           loss_fun = None, activation_fun = None,
                        cv=10, optType='randomizedSearch', param_grid=None,
                              n_iter_search=15, n_jobs=1, scoring=make_scorer(matthews_corrcoef))
# Epoch 00047: early stopping
# dict_keys(['accuracy', 'loss', 'val_accuracy', 'val_loss', 'lr'])
# ('Training Accuracy mean: ', 0.9857746109049371)
# ('Validation Accuracy mean: ', 0.8745526696773286)
# ('Training Loss mean: ', 0.05575073859158983)
# ('Validation Loss mean: ', 0.46936905447472915)
# Model: "sequential_151"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_151 (Embedding)    (None, 628, 64)           1344
# _________________________________________________________________
# flatten_151 (Flatten)        (None, 40192)             0
# _________________________________________________________________
# dense_514 (Dense)            (None, 128)               5144704
# _________________________________________________________________
# batch_normalization_363 (Bat (None, 128)               512
# _________________________________________________________________
# dropout_363 (Dropout)        (None, 128)               0
# _________________________________________________________________
# dense_515 (Dense)            (None, 64)                8256
# _________________________________________________________________
# batch_normalization_364 (Bat (None, 64)                256
# _________________________________________________________________
# dropout_364 (Dropout)        (None, 64)                0
# _________________________________________________________________
# dense_516 (Dense)            (None, 32)                2080
# _________________________________________________________________
# batch_normalization_365 (Bat (None, 32)                128
# _________________________________________________________________
# dropout_365 (Dropout)        (None, 32)                0
# _________________________________________________________________
# dense_517 (Dense)            (None, 1)                 33
#                                                        =================================================================
# Total params: 5,157,313
# Trainable params: 5,156,865
# Non-trainable params: 448
# _________________________________________________________________
# [['Model with rank: 1\n', 'Mean validation score: 0.432 (std: 0.340)\n', "Parameters: {'output_dim': 64, 'l2': 1e-05, 'l1': 0, 'hidden_layers': (128, 64, 32), 'dropout_rate': 0.25}\n", '\n'], ['Model with rank: 2\n', 'Mean validation score: 0.389 (std: 0.343)\n', "Parameters: {'output_dim': 128, 'l2': 0.0001, 'l1': 0, 'hidden_layers': (64,), 'dropout_rate': 0.25}\n", '\n'], ['Model with rank: 3\n', 'Mean validation score: 0.323 (std: 0.271)\n', "Parameters: {'output_dim': 128, 'l2': 0.0001, 'l1': 0, 'hidden_layers': (128, 64, 32), 'dropout_rate': 0.2}\n", '\n']]
# Best score (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# df
# means      stds  output_dim  ...       l1  hidden_layers dropout_rate
# 11  0.431848  0.339559          64  ...  0.00000  (128, 64, 32)         0.25
# 7   0.389322  0.343483         128  ...  0.00000          (64,)         0.25
# 0   0.322500  0.270799         128  ...  0.00000  (128, 64, 32)         0.20
# 6   0.212505  0.273431          64  ...  0.00100       (64, 32)         0.10
# 13  0.202879  0.296148         256  ...  0.00010  (128, 64, 32)         0.25
# 4   0.195444  0.210277         256  ...  0.00100  (128, 64, 32)         0.10
# 14  0.166272  0.290165         256  ...  0.00010  (128, 64, 32)         0.40
# 12  0.154322  0.241917         128  ...  0.00001      (128, 64)         0.40
# 1   0.148638  0.267957          64  ...  0.00000  (128, 64, 32)         0.10
# 2   0.136840  0.208744         128  ...  0.00010      (128, 64)         0.30
# 3   0.126069  0.249424         256  ...  0.00000       (64, 32)         0.50
# 10  0.074783  0.165298          64  ...  0.00000  (128, 64, 32)         0.50
# 5   0.059248  0.096716         128  ...  0.00001          (64,)         0.35
# 9   0.051125  0.147609         256  ...  0.00010       (64, 32)         0.50
# 8   0.040324  0.120971         256  ...  0.00000  (128, 64, 32)         0.40
# [15 rows x 7 columns]

scores, report, cm, cm2 = dl.model_complete_evaluate()

# [[2338  113]
#     [  72  745]]
# === Classification Report ===
# precision    recall  f1-score   support
# 0       0.97      0.95      0.96      2451
# 1       0.87      0.91      0.89       817
# accuracy                           0.94      3268
# macro avg       0.92      0.93      0.93      3268
# weighted avg       0.94      0.94      0.94      3268
# metrics                           scores
# 0   Accuracy                          0.94339
# 1        MCC                         0.851986
# 2   log_loss                         0.417523
# 3   f1 score                         0.889552
# 4    roc_auc                         0.932885
# 5  Precision  [0.25, 0.8682983682983683, 1.0]
# 6     Recall   [1.0, 0.9118727050183598, 0.0]
#################################### ratio 1:1

AMP_data = pd.read_csv('/home/amsequeira/propythia/propythia/example/AMP/datasets/ampep_feat_complete.csv')
AMP_data = AMP_data.drop({'Unnamed: 0'}, axis=1)
AMP_data['label'] = 1
non_AMP_data_1_3 = pd.read_csv(
    '/home/amsequeira/propythia/propythia/example/AMP/datasets/non_ampep_feat_complete.csv')
non_AMP_data_1_3['label'] = 0
non_AMP_data_1_1 = non_AMP_data_1_3.sample(n=AMP_data.shape[0], replace=False)

dataset = pd.concat([AMP_data, non_AMP_data_1_1])  # (13072, 642)

fps_y = dataset['label']
fps_x = dataset.drop(['label', 'sequence'], axis=1)
#  Preprocess
prepro = Preprocess()  # Create Preprocess object
fps_x_clean, columns_deleted = prepro.preprocess(fps_x, columns_names=True, threshold=0,
                                                 standard=True)  # [13072 rows x 628 columns]

# NO FEATURE SELECTION
X_train, X_test, y_train, y_test = train_test_split(fps_x_clean, fps_y, stratify=fps_y)
# standard scaler article does not refer scaling and do not validate in x_test
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


dl = DeepML(X_train, y_train, X_test, y_test, number_classes=2, problem_type='binary',
            x_dval=None, y_dval=None, epochs=500, batch_size=512,
            path='/home/amsequeira/propythia/propythia/example/AMP', report_name='ampep', verbose=1)


dnn = dl.run_dnn_simple(
    input_dim=X_train.shape[1],
    optimizer='Adam',
    hidden_layers=(128, 64),
    dropout_rate=(0.3,),
    batchnormalization=(True,),
    l1=1e-5, l2=1e-4,
    final_dropout_value=0.3,
    initial_dropout_value=0.0,
    loss_fun=None, activation_fun=None,
    cv=10, optType='randomizedSearch', param_grid=None, n_iter_search=15, n_jobs=1,
    scoring=make_scorer(matthews_corrcoef))

# poch 00085: early stopping
# dict_keys(['accuracy', 'loss', 'val_accuracy', 'val_loss', 'lr'])
# ('Training Accuracy mean: ', 0.9776547957869137)
# ('Validation Accuracy mean: ', 0.9494668680078843)
# ('Training Loss mean: ', 0.13744489686454045)
# ('Validation Loss mean: ', 0.2436266220667783)
# Model: "sequential_303"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense_970 (Dense)            (None, 64)                40256
# _________________________________________________________________
# batch_normalization_667 (Bat (None, 64)                256
# _________________________________________________________________
# dropout_667 (Dropout)        (None, 64)                0
# _________________________________________________________________
# dense_971 (Dense)            (None, 1)                 65
#                                                        =================================================================
# Total params: 40,577
# Trainable params: 40,449
# Non-trainable params: 128
# _________________________________________________________________
# [['Model with rank: 1\n', 'Mean validation score: 0.775 (std: 0.045)\n', "Parameters: {'l2': 0.001, 'l1': 1e-05, 'hidden_layers': (64,), 'dropout_rate': 0.5}\n", '\n'], ['Model with rank: 2\n', 'Mean validation score: 0.767 (std: 0.022)\n', "Parameters: {'l2': 0, 'l1': 0.0001, 'hidden_layers': (64,), 'dropout_rate': 0.2}\n", '\n'], ['Model with rank: 3\n', 'Mean validation score: 0.766 (std: 0.037)\n', "Parameters: {'l2': 1e-05, 'l1': 0.001, 'hidden_layers': (64,), 'dropout_rate': 0.25}\n", '\n']]
# Best score (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# df
# means      stds       l2       l1  hidden_layers  dropout_rate
# 7   0.774849  0.045332  0.00100  0.00001          (64,)          0.50
# 9   0.766523  0.021553  0.00000  0.00010          (64,)          0.20
# 14  0.766256  0.037012  0.00001  0.00100          (64,)          0.25
# 5   0.764587  0.054932  0.00001  0.00000      (128, 64)          0.10
# 10  0.752659  0.028550  0.00001  0.00001      (128, 64)          0.10
# 6   0.727601  0.036609  0.00000  0.00010      (128, 64)          0.50
# 12  0.711491  0.091575  0.00010  0.00100       (64, 32)          0.10
# 0   0.696570  0.060498  0.00001  0.00001      (128, 64)          0.20
# 8   0.690625  0.079078  0.00000  0.00100      (128, 64)          0.50
# 4   0.687983  0.078922  0.00010  0.00001       (64, 32)          0.30
# 13  0.680060  0.122756  0.00010  0.00000       (64, 32)          0.35
# 11  0.671693  0.125734  0.00000  0.00000  (128, 64, 32)          0.30
# 2   0.643858  0.069794  0.00000  0.00100       (64, 32)          0.30
# 3   0.606302  0.121438  0.00100  0.00010  (128, 64, 32)          0.40
# 1   0.502403  0.145623  0.00000  0.00100  (128, 64, 32)          0.50

scores, report, cm, cm2 = dl.model_complete_evaluate()

# === Confusion Matrix ===
# [[775  42]
#  [ 31 786]]
# === Classification Report ===
# precision    recall  f1-score   support
# 0       0.96      0.95      0.96       817
# 1       0.95      0.96      0.96       817
# accuracy                           0.96      1634
# macro avg       0.96      0.96      0.96      1634
# weighted avg       0.96      0.96      0.96      1634
# metrics                          scores
# 0   Accuracy                        0.955324
# 1        MCC                        0.910731
# 2   log_loss                        0.270332
# 3   f1 score                        0.955623
# 4    roc_auc                        0.955324
# 5  Precision  [0.5, 0.9492753623188406, 1.0]
# 6     Recall  [1.0, 0.9620563035495716, 0.0]


dl = DeepML(X_train, y_train, X_test, y_test, number_classes=2, problem_type='binary',
            x_dval=None, y_dval=None, epochs=500, batch_size=512,
            path='/home/amsequeira/propythia/propythia/example/AMP', report_name='ampep', verbose=1)

dnn_emb= dl.run_dnn_embedding(input_dim=X_train.shape[1],
                              optimizer='Adam',
                              input_dim_emb=21, output_dim=256, input_length=1000, mask_zero=True,
                              hidden_layers=(128, 64),
                              dropout_rate=0.3,
                              batchnormalization=True,
                              l1=1e-5, l2=1e-4,
                              final_dropout_value=0.3,
                              loss_fun = None, activation_fun = None,
                              cv=10, optType='randomizedSearch', param_grid=None,
                              n_iter_search=15, n_jobs=1, scoring=make_scorer(matthews_corrcoef))
#
# Epoch 00072: early stopping
# dict_keys(['accuracy', 'loss', 'val_accuracy', 'val_loss', 'lr'])
# ('Training Accuracy mean: ', 0.9883970725867484)
# ('Validation Accuracy mean: ', 0.6892113619380527)
# ('Training Loss mean: ', 0.18016060990177923)
# ('Validation Loss mean: ', 1.7215693650974169)
# Model: "sequential_151"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_151 (Embedding)    (None, 628, 256)          5376
# _________________________________________________________________
# flatten_151 (Flatten)        (None, 160768)            0
# _________________________________________________________________
# dense_483 (Dense)            (None, 64)                10289216
# _________________________________________________________________
# batch_normalization_332 (Bat (None, 64)                256
# _________________________________________________________________
# dropout_332 (Dropout)        (None, 64)                0
# _________________________________________________________________
# dense_484 (Dense)            (None, 32)                2080
# _________________________________________________________________
# batch_normalization_333 (Bat (None, 32)                128
# _________________________________________________________________
# dropout_333 (Dropout)        (None, 32)                0
# _________________________________________________________________
# dense_485 (Dense)            (None, 1)                 33
#                                                        =================================================================
# Total params: 10,297,089
# Trainable params: 10,296,897
# Non-trainable params: 192
# _________________________________________________________________
# [['Model with rank: 1\n', 'Mean validation score: 0.311 (std: 0.342)\n', "Parameters: {'output_dim': 256, 'l2': 0.001, 'l1': 0, 'hidden_layers': (64, 32), 'dropout_rate': 0.1}\n", '\n'], ['Model with rank: 2\n', 'Mean validation score: 0.297 (std: 0.313)\n', "Parameters: {'output_dim': 128, 'l2': 1e-05, 'l1': 0.001, 'hidden_layers': (64, 32), 'dropout_rate': 0.1}\n", '\n'], ['Model with rank: 3\n', 'Mean validation score: 0.218 (std: 0.320)\n', "Parameters: {'output_dim': 128, 'l2': 0.0001, 'l1': 0, 'hidden_layers': (128, 64), 'dropout_rate': 0.5}\n", '\n']]
# Best score (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
# df
# means      stds  output_dim  ...       l1  hidden_layers dropout_rate
# 10  0.310679  0.341739         256  ...  0.00000       (64, 32)         0.10
# 6   0.296788  0.312526         128  ...  0.00100       (64, 32)         0.10
# 1   0.218349  0.320178         128  ...  0.00000      (128, 64)         0.50
# 12  0.204546  0.238518          64  ...  0.00001      (128, 64)         0.10
# 7   0.190307  0.316703          64  ...  0.00000          (64,)         0.35
# 11  0.170141  0.297769         128  ...  0.00001          (64,)         0.40
# 4   0.169560  0.266429         256  ...  0.00001  (128, 64, 32)         0.35
# 3   0.164313  0.261398         128  ...  0.00100  (128, 64, 32)         0.25
# 5   0.139088  0.279206         256  ...  0.00000  (128, 64, 32)         0.20
# 2   0.133811  0.231163         256  ...  0.00001          (64,)         0.20
# 0   0.123515  0.247621         256  ...  0.00001  (128, 64, 32)         0.50
# 14  0.117444  0.249835         128  ...  0.00100          (64,)         0.25
# 9   0.111057  0.222214          64  ...  0.00000  (128, 64, 32)         0.50
# 13  0.090084  0.203985          64  ...  0.00001  (128, 64, 32)         0.30
# 8   0.067666  0.202998         128  ...  0.00001  (128, 64, 32)         0.10
# [15 rows x 7 columns]

scores, report, cm, cm2 = dl.model_complete_evaluate()

# === Confusion Matrix ===
# [[814   3]
#  [640 177]]
# === Classification Report ===
# precision    recall  f1-score   support
# 0       0.56      1.00      0.72       817
# 1       0.98      0.22      0.36       817
# accuracy                           0.61      1634
# macro avg       0.77      0.61      0.54      1634
# weighted avg       0.77      0.61      0.54      1634
# metrics                           scores
# 0   Accuracy                         0.606487
# 1        MCC                         0.340119
# 2   log_loss                          2.66743
# 3   f1 score                         0.355065
# 4    roc_auc                         0.606487
# 5  Precision   [0.5, 0.9833333333333333, 1.0]
# 6     Recall  [1.0, 0.21664626682986537, 0.0]


# todo
# analyse the best models int erms of leearning curves and features
# analyse eatures !!!!!





dataset = pd.concat([AMP_data, non_AMP_data_1_1])  #
df = ml.features_importances_df(classifier=best_svm_model_AMPEP, model_name='svm', top_features=30,
                                column_to_sort='mean_coef')
ml.features_importances_plot(classifier=best_svm_model_AMPEP, top_features=30, model_name='svm',
                             column_to_plot=None,
                             show=True, path_save='feat_impo.png',
                             title='FI SVM all feature 1:3',
                             kind='barh', figsize=(9, 7), color='r', edgecolor='black')

ml.plot_learning_curve(classifier=best_svm_model_AMPEP, title='Learning curve', ylim=None,
                       cv=10,
                       n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5),
                       path_save='plot_learning_curve', show=True, scalability=True, performance=True)

# MY WAY SVM and RF ALL FEATURES feature selection and preprocess
# PLOTS AND FEATURE IMPORTANCE
#
# def get_dataset_ratio(ratio, neg_data, pos_data):
#     scores_rf_ampep=[]
#     scores_rf_paramoptim = []
#     scores_svm_paramoptim = []
#     cm_rf_ampep=[]
#     cm_rf_paramoptim = []
#     cm_svm_paramoptim = []
#     n_neg = neg_data.shape[0]
#     n_pos = pos_data.shape[0]
#     n_split = n_neg/(n_pos*ratio)
#     # the indices used to select parts from dataframe
#     ixs = np.arange(neg_data.shape[0])
#     np.random.shuffle(ixs)
#     # np.split cannot work when there is no equal division
#     # so we need to find out the split points ourself
#     # we need (n_split-1) split points
#     split_points = [i*neg_data.shape[0]//n_split for i in range(1, n_split)]
#     # use these indices to select the part we want
#     for ix in np.split(ixs, split_points):
#         dataset_negative = neg_data.iloc[ix]
#
#         dataset_negative['label'] = 0
#         dataset_positive['label'] = 1
#         dataset = pd.concat(dataset_negative, dataset_positive)
#
#         fps_x = dataset.loc[:, dataset.columns != 'label']
#         fps_y = dataset['label']
#
#         X_train, X_test,  y_train, y_test = train_test_split(fps_x, fps_y, stratify = fps_y)
#         ml = ShallowML(X_train, X_test,  y_train, y_test, report_name=None, columns_names=fps_x.columns)
#
#
#         # por standar scaler todo
#
#         # with parameters defined by article
#         param_grid = [{'clf__n_estimators': [100], 'clf__max_features': ['sqrt']}]
#         # optimize MCC
#         # best_rf_model_AMPEPparameters=ml.train_best_model('rf',score=make_scorer(matthews_corrcoef),param_grid=param_grid)
#
#         # optimize ROC_AUC
#         best_rf_model_AMPEPparameters = ml.train_best_model('rf', score='roc_auc', param_grid=param_grid)
#         scores, report, cm = ml.score_testset(best_rf_model_AMPEPparameters)
#         #update scores to then get the average results
#         scores_rf_ampep.append(scores)
#         cm_rf_ampep.append(scores)
#
#         # using grid search with default parameter search
#         # optimize MCC
#         # best_rf_model = ml.train_best_model('rf')
#         # optimize ROC-AUC
#         best_rf_model = ml.train_best_model('rf', score='roc_auc', cv=10)
#         scores, report, cm = ml.score_testset(best_rf_model)
#         scores_rf_paramoptim.append(scores)
#         cm_rf_paramoptim.append(scores)
#
#         # using grid search with default parameter search  for SVM
#         best_svm_model = ml.train_best_model('svm', param_grid=param_grid, scaler=None)
#         scores, report, cm = ml.score_testset(best_svm_model)
#         scores_svm_paramoptim.append(scores)
#         cm_svm_paramoptim.append(cm)
#
#         #  get the average scores
#         # scores_rf_ampep=[]
#         # scores_rf_paramoptim = []
#         # scores_svm_paramoptim = []


# tabela AMPEP  RF CTD  C T D AAC DPC PAAC
# same 1:1
# tabela c o melhor modelo RF same features
# tabela  c o melhor SVM same feature
# plots dos melhores modelos
# FI
# features_importances_df
# features_importances_plot



# FI for ampep RF
ml = ShallowML(X_train=fps_x, X_test=None, y_train=fps_y, y_test=None, report_name=None, columns_names=fps_x.columns)
param_grid = [{'clf__n_estimators': [100], 'clf__max_features': ['sqrt']}]
clf = ml.train_best_model('rf', score=make_scorer(matthews_corrcoef), param_grid=param_grid, cv=10)
df = ml.features_importances_df()
ml.features_importances_plot(top_features=20, show=True, path_save='feat_impo.png')


# importance
# _SolventAccessibilityD3001    0.117784
# _SecondaryStrD1001            0.093922
# _PolarizabilityD3001          0.090054
# _HydrophobicityD3001          0.085964
# _PolarityD1001                0.077245
# _NormalizedVDWVD3001          0.057542
# _ChargeD2001                  0.056585
# _HydrophobicityD1001          0.023364
# _SolventAccessibilityD2001    0.021304
# _PolarityD2001                0.017292
# _ChargeD3075                  0.016007
# _SolventAccessibilityD1075    0.015848
# _NormalizedVDWVD2001          0.014398
# _PolarizabilityD2001          0.013281
# _ChargeD3100                  0.012794
# _SolventAccessibilityD1025    0.011316
# _NormalizedVDWVD2100          0.007420
# _PolarityD2025                0.007255
# _SolventAccessibilityD1001    0.006280
# _SolventAccessibilityD1100    0.006216
#using all features
# preprocess

# NO FEATURE SELECTION
# standard scaler article does not refer scaling and do not validate in x_test
scaler = StandardScaler().fit_transform(fps_x_clean)
ml = ShallowML(X_train = fps_x_clean, X_test=None, y_train=fps_y, y_test=None, report_name=None, columns_names=fps_x_clean.columns)

clf = ml.train_best_model('rf', score=make_scorer(matthews_corrcoef), param_grid=None, cv=10)
df = ml.features_importances_df()
ml.features_importances_plot(top_features=20, show=True, path_save='feat_impo.png')

# importance
# _SolventAccessibilityD3001    0.057254
# _PolarityD1001                0.045801
# _HydrophobicityD3001          0.044771
# _SecondaryStrD1001            0.039257
# MW_modlamp                    0.034471
# _PolarizabilityD3001          0.034002
# formulaN                      0.033183
# tot                           0.031318
# hydrogen                      0.029849
# lenght                        0.028938
# formulaH                      0.025474
# single                        0.025008
# formulaC                      0.022339
# formulaO                      0.020229
# double                        0.018064
# PAAC13                        0.017846
# _ChargeD2001                  0.017657
# APAAC13                       0.017313
# M                             0.014306
# PAAC5                         0.010580