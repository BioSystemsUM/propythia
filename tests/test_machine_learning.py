"""
##############################################################################

File containing tests functions to check if all functions from machine_learning module are properly working

Authors: Ana Marta Sequeira

Date: 06/2019

Email:

##############################################################################
"""
from propythia.machine_learning import MachineLearning
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import ShuffleSplit

def test_machine_learning():
    #split dataset
    dataset = pd.read_csv(r'datasets/dataset1_test_clean_fselection.csv', delimiter=',')

    x_original=dataset.loc[:, dataset.columns != 'labels']
    labels=dataset['labels']

    #create Machine learning object
    ml=MachineLearning(x_original, labels, classes=['pos', 'neg'])

    #tests models
    print('best model svm')
    best_svm_model = ml.train_best_model('svm')
    print('best model rf')
    best_rf_model = ml.train_best_model('rf')
    print('best model sgd')
    best_sgd_model = ml.train_best_model('sgd')
    print('best model gradient boosting')
    best_gboosting_model = ml.train_best_model('gboosting')
    print('best model lr')
    best_lr_model = ml.train_best_model('lr')

    # feature importance of models
    ml.features_importances(best_svm_model,'svm')
    ml.features_importances(best_rf_model,'rf')
    # ml.features_importances(best_sgd_model,'sgd')
    # ml.features_importances(best_gboosting_model,'gboosting')
    # ml.features_importances(best_lr_model,'lr')

    print('best model nn')
    best_nn_model = ml.train_best_model('nn')

    print('best model gnb')
    best_gnb_model = ml.train_best_model('gnb')

    print('best model knn')
    best_knn_model = ml.train_best_model('knn')


    #plot validation curve
    print('plot validation_svm')
    ml.plot_validation_curve(best_svm_model, param_name='clf__C',
                             param_range=[0.00001,0.0001, 0.001, 0.01, 0.1, 1, 10,100])

    # print('plot validation_gboosting')
    # ml.plot_validation_curve(best_gboosting_model, param_name='clf__n_estimators',
    #                          param_range=[ 1, 10,100,500])


    print('score_test_set_svm')
    print(ml.score_testset(best_svm_model))

    print('score_test_set_rf')
    print(ml.score_testset(best_rf_model))

    print('score_test_set_rf')
    print(ml.score_testset(best_rf_model))

    print('score_test_set_gboosting')
    print(ml.score_testset(best_gboosting_model))

    print('score_test_set_lr')
    print(ml.score_testset(best_lr_model))
    #
    # print('roc curve')
    # ml.plot_roc_curve(best_svm_model)
    #
    # print('roc curve')
    # ml.plot_roc_curve(best_gboosting_model)
    #
    # print('roc curve')
    # ml.plot_roc_curve(best_lr_model)

    # print('plot learning curve')
    # title = "Learning Curves (SVM)"
    # # # SVC is more expensive so we do a lower number of CV iterations:
    # cv = ShuffleSplit(n_splits=50, test_size=0.3, random_state=42)
    # ml.plot_learning_curve(best_svm_model, title, ylim=(0.8, 1.01), cv=cv, n_jobs=4)
    #
    # cv = ShuffleSplit(n_splits=50, test_size=0.3, random_state=42)
    # ml.plot_learning_curve(best_lr_model, title="Learning Curves (LR)", ylim=(0.8, 1.01), cv=cv, n_jobs=4)
    # #
    # title = "Learning Curves (RF)"
    # # # SVC is more expensive so we do a lower number of CV iterations:
    # ml.plot_learning_curve(best_rf_model, title, ylim=(0.8, 1.01), cv=10, n_jobs=4)

    # title = "Learning Curves (SGD)"
    # # Cross validation with 100 iterations to get smoother mean tests and train
    # # score curves, each time with 20% data randomly selected as a validation set.
    # ml.plot_learning_curve(best_sgd_model, title, ylim=(0.4, 1.01), cv=cv, n_jobs=4)
    # #
    # title = "Learning Curves (Neural Networks)"
    # # Cross validation with 100 iterations to get smoother mean tests and train
    # # score curves, each time with 20% data randomly selected as a validation set.
    # ml.plot_learning_curve(best_nn_model, title, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
    # #
    # title = "Learning Curves (Naive Bayes - Gaussian)"
    # # Cross validation with 100 iterations to get smoother mean tests and train
    # # score curves, each time with 20% data randomly selected as a validation set.
    # ml.plot_learning_curve(best_gnb_model, title, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
    # #
    # title = "Learning Curves (Gboosting)"
    # # # SVC is more expensive so we do a lower number of CV iterations:
    # cv = ShuffleSplit(n_splits=50, test_size=0.3, random_state=42)
    # ml.plot_learning_curve(best_gboosting_model, title, ylim=(0.8, 1.01), cv=cv, n_jobs=4)
    #
    #
    #
    # #make previsions
    X_train, X_test, y_train, y_test = train_test_split(x_original,labels)
    print('predict_svm')
    df = ml.predict(best_svm_model, x=X_test, seqs=y_test)
    print(df.head(10))

    seq='IQIPSEFTIGNMEEFIQTSSPKVTIDCAAFVCGDYAACKSQLVEYGSFCDNINAILTEVNELLDTTQLQVANSLMNGVTLSTKLKDGVNFNVDDINFSSVLGCLGSECSKASSRSAIEDLLFDKVKLSDVGFVAAYNNCTGGAEIRDLICVQSYKGIKVLPPLLSENQISGYTLAATSASLFPPWTAAAGVPFYLNVQYRINGLGVTMDVLSQNQKLIANAFNNALDAIQEGFDATNSALVKIQAVVNANAEALNNLLQQLSNRFGAISSSLQEILSRLDALEAEAQIDRLINGRLTALNAYVSQQLSDSTLVKFSAAQAMEKVNECVKSQSSRINFCGNGNHIISLVQNAPYGLYFIHFSYVPTKYVTAKVSPGLCIAGDRGIAPKSGYFVNVNNTWMYTGSGYYYPEPITENNVVVMSTCAVNYTKAPYVMLNTSTPNLPDFREELDQWFKNQTSVAPDLSLDYINVTFLDLQVEMNRLQEAIKVLNQSYINLKDIGTYEYYVKWPWYVWLLIGLAGVAMLVLLFFICCCTGCGTSCFKKCGGCC'
    ml.predict_window(best_svm_model,seq=seq,x=None, window_size=15,gap=2,features=[], names=None, y=None, filename=None)


if __name__ == '__main__':
    test_machine_learning()