"""
##############################################################################

File containing tests functions to check if all functions from machine_learning module are properly working

Authors: Ana Marta Sequeira

Date: 06/2019 altered 01/2021

Email:

##############################################################################
"""
from propythia.shallow_ml import ShallowML
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.preprocessing import label_binarize, StandardScaler

def test_shallow_ml():
    # split dataset
    dataset = pd.read_csv(r'datasets/dataset1_test_clean_fselection.csv', delimiter=',')

    x_original=dataset.loc[:, dataset.columns != 'labels']
    labels=dataset['labels']
    lab = label_binarize(labels, ['neg', 'pos'])
    labels =[item for sublist in lab for item in sublist]

    scaler = StandardScaler()
    fps_x = scaler.fit_transform(x_original)

    x_train, x_test, y_train, y_test = train_test_split(x_original, labels, train_size=0.2, stratify=labels)
    # create Machine learning object
    ml=ShallowML(x_train, x_test, y_train, y_test, report_name=None, columns_names=None)

    # TRAIN BEST MODEL
    best_svm_model = ml.train_best_model(model='svm', scaler=None,
                     score=make_scorer(matthews_corrcoef),
                     cv=3, optType='gridSearch', param_grid=None,
                     n_jobs=10,
                     random_state=1, n_iter=15, refit=True)
    print('svm')
    best_linear_model = ml.train_best_model(model='linear_svm', scaler=None,
                                         score=make_scorer(matthews_corrcoef),
                                         cv=3, optType='gridSearch', param_grid=None,
                                         n_jobs=10,
                                         random_state=1, n_iter=15, refit=True)
    print('svm linear')
    best_rf_model = ml.train_best_model(model='rf', scaler=None,
                                    score=make_scorer(matthews_corrcoef),
                                    cv=3, optType='gridSearch', param_grid=None,
                                    n_jobs=10,
                                    random_state=1, n_iter=15, refit=True)
    print('rf')

    best_sgd_model = ml.train_best_model(model='sgd', scaler=None,
                                         score=make_scorer(matthews_corrcoef),
                                         cv=3, optType='gridSearch', param_grid=None,
                                         n_jobs=10,
                                         random_state=1, n_iter=15, refit=True)
    print('sgd')

    best_gboosting_model = ml.train_best_model('gboosting', cv=3)
    print('gboosting')

    best_lr_model = ml.train_best_model('lr', cv=3)
    print('lr')


    # CROSS VAL SCORE
    scores = ml.cross_val_score_model(model_name='svm',
                          score='accuracy',
                          cv=5,
                          n_jobs=10,
                          random_state=1)

    ml=ShallowML(x_train, x_test, y_train, y_test, report_name=None, columns_names=None)

    # TRAIN BEST MODEL
    best_svm_model = ml.train_best_model(model='svm', scaler=None,
                                         score=make_scorer(matthews_corrcoef),
                                         cv=3, optType='gridSearch', param_grid=None,
                                         n_jobs=10,
                                         random_state=1, n_iter=15, refit=True, probability=True)
    # evaluate on test set
    scores, report, cm, cm2 = ml.score_testset()

    # graphics
    ml.plot_roc_curve(ylim=(0.0, 1.00), xlim=(0.0, 1.0),
               title='Receiver operating characteristic (ROC) curve',
               path_save=None, show=True)

    ml.plot_validation_curve(param_name='clf__C', param_range=[0.01, 0.1, 1.0, 10],
                          cv=5,
                          score=make_scorer(matthews_corrcoef), title="Validation Curve",
                          xlab="parameter range", ylab="MCC", n_jobs=1, show=True,
                          path_save=None)

    df = ml.features_importances_df(top_features=20, column_to_sort='mean_coef')
    print(df)
    ml.features_importances_plot(column_to_plot=0, show=True, path_save='feat_impo.png',
                              title=None,
                              kind='barh', figsize=(9, 7), color='r', edgecolor='black')


    ml=ShallowML(x_train, x_test, y_train, y_test, report_name=None, columns_names=None)

    predict_df = ml.predict(x=x_test, seqs=None, classifier=best_svm_model, names=None, true_y=y_test)

    # TRAIN BEST MODEL
    best_rf_model = ml.train_best_model(model='rf', scaler=None,
                                        score=make_scorer(matthews_corrcoef),
                                        cv=3, optType='gridSearch', param_grid=None,
                                        n_jobs=10,
                                        random_state=1, n_iter=15, refit=True)
    ml.features_importances_df(top_features=20, column_to_sort='mean_coef')
    ml.features_importances_plot(show=True, path_save='feat_impo.png',
                                 title=None,
                                 kind='barh', figsize=(9, 7), color='r', edgecolor='black')

if __name__ == '__main__':
    test_machine_learning()