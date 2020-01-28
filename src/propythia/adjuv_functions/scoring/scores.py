# -*- coding: utf-8 -*-
"""
##############################################################################

A function used for a brief comparation of scores between a random forest, gaussian bayes and a SVM with or
without SVC feature selection
Authors:

Date: 06/2019

Email:

##############################################################################
"""
from sklearn import datasets, svm
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


def score_methods(x_original,x_reduced,labels):
    svm_mod =svm.SVC(gamma=0.001,C=100)
    gnb = GaussianNB()
    rf=RandomForestClassifier(n_estimators=100)
    scores=cross_val_score(svm_mod,x_original,labels, cv=10)
    print('Score relative to original dataset with SVM:', scores.mean())
    scores_vt=cross_val_score(svm_mod,x_reduced,labels,cv=10)
    print('Score relative to filtered dataset with SVM:', scores_vt.mean())

    scores=cross_val_score(gnb,x_original,labels, cv=10)
    print('Score relative to original dataset with Gaussian:', scores.mean())
    scores_vt=cross_val_score(gnb,x_reduced,labels,cv=10)
    print('Score relative to filtered dataset with Gaussian:', scores_vt.mean())

    scores=cross_val_score(rf,x_original,labels,cv=10)
    print('Score relative to original dataset with random forest:', scores.mean())
    scores_vt=cross_val_score(rf,x_reduced,labels,cv=10)
    print('Score relative to filtered dataset with random forest:', scores_vt.mean())


if __name__=="__main__":
    pass