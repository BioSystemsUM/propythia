# -*- coding: utf-8 -*-
"""
##############################################################################

File to do a comparative analysis using the MLACP article as case study.
MLACP:machine-learning-based prediction of anticancer peptides


B. Manavalan, S. Basith, T. Hwan Shin, S. Choi, M. Ok Kim, and G. Lee, “MLACP:
machine-learning-based prediction of anticancer peptides” Oncotarget, vol. 8, no. 44,
pp. 77121–77136, 2017.

Authors: Ana Marta Sequeira

Date: 07/2019

Email:

##############################################################################
"""
import csv
import pandas as pd
from propythia.sequence import ReadSequence
from propythia.descriptors import Descriptor
from propythia.machine_learning import MachineLearning
from sklearn.metrics import make_scorer, accuracy_score, recall_score, confusion_matrix

# ##### BUILT SEQUENCE DATASET FROM COLLECTION OF DATA ######


def create_dataset():
    acp_data = r'datasets/Tyagi-B-positive_ori.txt'  # 187
    non_acp_data = r'datasets/Tyagi-B-negative_ori.txt' # 398

    with open('datasets/test_MLACP.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        with open(acp_data, newline='') as csvfile_ACP:
            spamreader = csv.reader(csvfile_ACP, delimiter=' ', quotechar='|')
            for row in spamreader:
                if row[0].startswith('>'):
                    pass
                else:# just sequences. not '>acp_number' character
                    spamwriter.writerow(row)

        with open(non_acp_data, newline='') as csvfile_nonACP:
            spamreader = csv.reader(csvfile_nonACP, delimiter=' ', quotechar='|')
            for row in spamreader:
                if row[0].startswith('>'):
                    pass
                else:
                    spamwriter.writerow(row)

    with open('datasets/test_MLACP.csv', 'r', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        row_count = sum(1 for row in spamreader)
        print(row_count)


def add_features_article():
    dataset_in=r'datasets/test_MLACP.csv'
    rows_list = [] #creating an empty list of dataset rows

    #opening dataset
    with open(dataset_in) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            res={'sequence':row[0]}
            sequence=ReadSequence() # creating sequence object
            ps=sequence.read_protein_sequence(row[0])
            protein = Descriptor(ps) # creating object to calculate descriptors)
            feature=protein.adaptable([21,6]) #calculate AAC, DPC, ATP AN. the PCP of the article were replaced by the ones I have
            res.update(feature)
            print(res)
            rows_list.append(res)

    df = pd.DataFrame(rows_list)
    df.set_index(['sequence'],inplace=True)

    # adding labels to dataset
    labels=['ACP']*187 + ['non_ACP']*398
    df['labels'] = labels

    dataset_out=r'datasets/test_MLACP_ART_dpc_atc.csv'
    df.to_csv(dataset_out,index=False)
    print(df.shape)
    # print(df.head(10))

# functions to make possible optimize the specificity


def tn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 0]


def fp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 1]


def fn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 0]


def tp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 1]


def specificity(y_true, y_pred):
    tn=confusion_matrix(y_true, y_pred)[0, 0]
    fp=confusion_matrix(y_true, y_pred)[0, 1]
    return tn / (tn + fp)


def machine_learning_RF(dataset_in):
    dataset = pd.read_csv(dataset_in, delimiter=',')
    x_original=dataset.loc[:, dataset.columns != 'labels']
    labels=dataset['labels']

    ml=MachineLearning(x_original, labels,classes=['ACP','non_ACP'])

    # with parameters defined by article
    param_grid = [{'clf__n_estimators': [10,100,200,300,400,500],
                   'clf__max_features': ['sqrt',2,3,5,7],
                   'clf__min_samples_split':[3,6,7,9,10]}]

    # scoring = {
    #     'accuracy': make_scorer(accuracy_score),
    #     'sensitivity': make_scorer(recall_score),
    #     'specificity': make_scorer(recall_score,pos_label=0)
    # }
    #
    # # or
    #
    # scoring2 = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
    #          'fp': make_scorer(fp), 'fn': make_scorer(fn),
    #         'specificity':make_scorer((specificity))}

    #optimize ROC_AUC
    best_rf_model = ml.train_best_model('rf',score='roc_auc',param_grid=param_grid)
    print(ml.score_testset(best_rf_model))

    # #optimize specificity
    # best_rf_model = ml.train_best_model('rf',score=scoring2['specificity'],param_grid=param_grid)
    # print(ml.score_testset(best_rf_model))


def machine_learning_SVM(dataset_in):
    dataset = pd.read_csv(dataset_in, delimiter=',')
    x_original=dataset.loc[:, dataset.columns != 'labels']
    labels=dataset['labels']

    ml=MachineLearning(x_original, labels,classes=['ACP','non_ACP'])

    #with grid search
    param_range = [0.001, 0.01, 0.1, 1.0]

    param_grid = [{'clf__C': param_range,
                   'clf__kernel': ['rbf'],
                   'clf__gamma': param_range
                   }]

    # optimize accuracy
    best_svm_model = ml.train_best_model('svm',score='roc_auc',param_grid=param_grid)
    print(ml.score_testset(best_svm_model))


# #
# create_dataset()
# add_features_article()

dataset_aac=r'datasets/test_MLACP_ART_aac.csv'
dataset_dpc=r'datasets/test_MLACP_ART_dac.csv'
dataset_atc=r'datasets/test_MLACP_ART_atc.csv'
dataset_aac_dpc=r'datasets/test_MLACP_ART_aac_dpc.csv'
dataset_aac_atc=r'datasets/test_MLACP_ART_aac_atc.csv'
dataset_dpc_atc=r'datasets/test_MLACP_ART_dpc_atc.csv'

# machine_learning_RF(dataset_aac)
# machine_learning_RF(dataset_dpc)
# machine_learning_RF(dataset_aac_dpc)

# machine_learning_SVM(dataset_aac)
# machine_learning_SVM(dataset_dpc)
# machine_learning_SVM(dataset_aac_dpc)



