# -*- coding: utf-8 -*-
"""
##############################################################################

File to do a comparative analysis using the AMPEP article as case study.
Pratiti Bhadra, Jielu Yan, Jinyan Li, Simon Fong, and Shirley W. I. Siu*
AmPEP: Sequence-based prediction of antimicrobial peptides using distribution patterns of amino acid properties and random forest.

Authors: Ana MArta Sequeira

Date: 07/2019

Email:

##############################################################################
"""
import csv
import pandas as pd
from propythia.sequence import ReadSequence
from propythia.descriptors import Descriptor
from propythia.preprocess import Preprocess
from propythia.feature_selection import FeatureSelection
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from propythia.machine_learning import MachineLearning

# ##### BUILT SEQUENCE DATASET FROM COLLECTION OF DATA 1:3 (POS/NEG)

def create_dataset():
    amp_data=r'datasets/M_model_train_AMP_sequence.fasta'
    # AMP 3268  sequences
    non_amp_data=r'datasets/M_model_train_nonAMP_sequence.fasta'
    # non-AMP 166791 sequences

    with open('datasets/test_AmPEP.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        with open(amp_data, newline='') as csvfile_AMP:
            spamreader = csv.reader(csvfile_AMP, delimiter=' ', quotechar='|')
            for row in spamreader:
                if len(row[0])>1: #just sequences. not '>' character
                    #print(row)
                    spamwriter.writerow(row)

        with open(non_amp_data, newline='') as csvfile_nonAMP:
            spamreader = csv.reader(csvfile_nonAMP, delimiter=' ', quotechar='|')
            for _ in range(5001):  # skip the first 500 rows
                next(spamreader)
            count=0
            non_amp_data=9805 #number of non AMP to add

            for row in spamreader:#arbitrary number to not start in the beggining
                if count<=non_amp_data:
                    if len(row[0])>1:
                        spamwriter.writerow(row)
                        count+=1

    with open('datasets/test_AmPEP.csv', 'r', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        row_count = sum(1 for row in spamreader)
        print(row_count)


def add_features_CTD():
    dataset_in=r'datasets/test_AmPEP.csv'
    rows_list = [] #creating an empty list of dataset rows

    #opening dataset
    with open(dataset_in) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            res={'sequence':row[0]}
            sequence=ReadSequence() #creating sequence object
            ps=sequence.read_protein_sequence(row[0])
            protein = Descriptor(ps) # creating object to calculate descriptors)
            feature=protein.adaptable([32]) #CTD feature
            res.update(feature)
            rows_list.append(res)

    df = pd.DataFrame(rows_list)
    df.set_index(['sequence'],inplace=True)
    labels=['AMP']*3268 + ['non_AMP']*9806 #adding labels to dataset


    #select only D feature
    d_cols = [col for col in df.columns if 'D' in col]
    ignore=['_NormalizedVDWVC1','_NormalizedVDWVC2','_NormalizedVDWVC3','_NormalizedVDWVT12','_NormalizedVDWVT13','_NormalizedVDWVT23']

    df=df[df.columns.intersection(d_cols)]
    df=df.drop(columns=['_NormalizedVDWVC1','_NormalizedVDWVC2','_NormalizedVDWVC3','_NormalizedVDWVT12','_NormalizedVDWVT13','_NormalizedVDWVT23'])
    df['labels'] = labels
    dataset_out=r'datasets/test_AmPEP_CTD_D.csv'
    df.to_csv(dataset_out,index=False)
    print(df.shape)
    #print(df.head(10))


def add_features_all():
    dataset_in=r'datasets/test_AmPEP.csv'
    rows_list = [] #creating an empty list of dataset rows

    #opening dataset
    with open(dataset_in) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            # print(row[0])
            res={'sequence':row[0]}
            sequence=ReadSequence() #creating sequence object
            ps=sequence.read_protein_sequence(row[0])
            protein = Descriptor(ps) # creating object to calculate descriptors)
            feature=protein.adaptable([19,20,21,32,33]) #calculate dot know each features!!!!!!!
            res.update(feature)
            rows_list.append(res)

    df = pd.DataFrame(rows_list)
    df.to_csv(r'datasets/test_AmPEP_all__BACKUP.csv',index=False)

    df.set_index(['sequence'],inplace=True)
    labels=['AMP']*3268 + ['non_AMP']*9806 #adding labels to dataset
    df['labels'] = labels

    dataset_out=r'datasets/test_AmPEP_all.csv'
    df.to_csv(dataset_out,index=False)
    print(df.shape)


def select_features():
    dataset_in=r'datasets/test_AmPEP_all.csv'
    dataset=pd.read_csv(dataset_in, delimiter=',')
    #separate labels
    labels=dataset['labels']
    dataset=dataset.loc[:, dataset.columns != 'labels']

    prepro=Preprocess() #Create Preprocess object

    #do the preprocessing
    dataset_clean,columns_deleted=prepro.preprocess(dataset, columns_names=True, threshold=0, standard=True)

    dataset_clean['labels']=labels #put labels back

    print('dataset original',dataset.shape)
    print('dataset after preprocess',dataset_clean.shape)

    pd.DataFrame(dataset_clean).to_csv(r'datasets/test_AmPEP_all_clean.csv',index=False)

    x_original=dataset_clean.loc[:, dataset_clean.columns != 'labels']
    fselect=FeatureSelection(dataset_clean, x_original, labels)

    # # #KBest com *mutual info classif*
    X_fit_univariate, X_transf_univariate,column_selected,scores,dataset_features= \
        fselect.univariate(score_func=mutual_info_classif, mode='k_best', param=250)

    # Select from model L1
    model_lr=LogisticRegression(C=0.1, penalty="l2", dual=False)
    #model= logistic regression
    X_fit_model, X_transf_model,column_selected,feature_importances,feature_importances_DF,dataset_features= \
        fselect.select_from_model_feature_elimination( model=model_lr)

    pd.DataFrame(dataset_features).to_csv(r'datasets/test_AmPEP_all_selected.csv',index=False)
    #print(df.head(10))


def machine_learning_rf(dataset_in, grid=None):
    dataset = pd.read_csv(dataset_in, delimiter=',')
    x_original=dataset.loc[:, dataset.columns != 'labels']

    labels=dataset['labels']

    ml=MachineLearning(x_original, labels,classes=['AMP','non_AMP'])

    if grid == 'AmPEP':
        #with parameters defined by article
        param_grid = [{'clf__n_estimators': [100],
                       'clf__max_features': ['sqrt']}]

        # optimize MCC
        # best_rf_model_AMPEPparameters=ml.train_best_model('rf',score=make_scorer(matthews_corrcoef),param_grid=param_grid)

        # optimize ROC_AUC
        best_rf_model_AMPEPparameters=ml.train_best_model('rf',score='roc_auc',param_grid=param_grid)
        print(ml.score_testset(best_rf_model_AMPEPparameters))

    else:
        #with grid search
        #optimize MCC
        #best_rf_model = ml.train_best_model('rf')

        #optimize ROC-AUC
        best_rf_model = ml.train_best_model('rf',score='roc_auc')
        print(ml.score_testset(best_rf_model))


def machine_learning_svm(dataset_in):
    dataset = pd.read_csv(dataset_in, delimiter=',')
    x_original=dataset.loc[:, dataset.columns != 'labels']

    labels=dataset['labels']

    ml=MachineLearning(x_original, labels,classes=['AMP','non_AMP'])

    #with grid search
    param_range = [0.001, 0.01, 0.1, 1.0]


    param_grid = [{'clf__C': param_range,
                       'clf__kernel': ['linear'],
                       'clf__gamma': param_range
                       }]

    best_svm_model = ml.train_best_model('svm',param_grid=param_grid, scaler=None)
    print(ml.score_testset(best_svm_model))


if __name__ == '__main__':
    # create_dataset()
    # add_features_CTD()
    # add_features_all()
    # select_features()

    # # RF with only D features (AMPEP PARAMETERSS)
    # machine_learning_rf('datasets/test_AmPEP_CTD_D.csv', grid = 'AmPEP')
    # # RF with only D features (GRID SEARCH)
    # machine_learning_rf('datasets/test_AmPEP_CTD_D.csv')
    #
    # # RF with more features(PARAMETERS AMPEP)
    # machine_learning_rf(r'datasets/test_AmPEP_all_selected.csv')
    # # RF with more features(GRID SEARCH)
    # machine_learning_rf(r'datasets/test_AmPEP_all_selected.csv')

    # # SVM with only CTD features (GRID SEARCH)
    machine_learning_svm(r'datasets/test_AmPEP_CTD_D.csv')
    # # SVM with more features(GRID SEARCH)
    machine_learning_svm(r'datasets/test_AmPEP_all_selected.csv')