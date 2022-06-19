import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
import numpy as np
from xgboost import XGBClassifier, plot_importance

plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 30})
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics

from sklearn.svm import SVC

import time
import sys
import warnings
import argparse

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from numpy import mean
from numpy import std

from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso

from sklearn import linear_model
from sklearn.neural_network import MLPClassifier

import joblib

def plot_feature_importance(importance,names,model_type,namefilex):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    fi_df.to_csv(namefilex)
    #Define size of bar plot
    plt.figure()
    #Plot Searborn bar chart
    #sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])

    if fi_df.shape[0]>20 :
        nfeatures = 20
    else :
        nfeatures = fi_df.shape[0]

    sns.barplot(x=fi_df.iloc[0:nfeatures,1], y=fi_df.iloc[0:nfeatures,0])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')

def tic():
    global _start_time
    _start_time = time.time()

def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))
    return (str('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec)))

def run_prediction(path_parent,csv_file_name,output_folder,code_id,code_outcome,n_repetitions,fold_type_out,fold_type_inn,n_folds_out,n_folds_in,external_validation_file):

    foldername_figures = csv_file_name
    file_results=output_folder + '/' + csv_file_name + '.txt'

    file_input = path_parent + '/' + csv_file_name

    file_ev = path_parent + '/' + external_validation_file

    print("Input csv file: " + file_input)
    folder_figure = output_folder + '/' + foldername_figures

    if not os.path.exists(folder_figure):
        os.makedirs(folder_figure)

    seed = 45

# ===========  process ===========

    text_file_output = open(file_results, "w+")

    _start_time = time.time()
    tic()

    data = pd.read_csv(file_input)
    print(data.shape)
    X = data.drop([code_id, code_outcome], axis=1)
    y = data[code_outcome]
    Z = data[code_id]

    data_ev = pd.read_csv(file_ev)
    print(data_ev.shape)
    val_data_ev = data_ev.drop([code_id, code_outcome], axis=1)
    val_target_ev = data_ev[code_outcome]
    participants_ID_ev = data_ev[code_id]

    accuracy_lr = list()
    f1_lr = list()
    precision_lr = list()
    recall_lr = list()
    auc_lr = list()

    accuracy_lr_std = list()
    f1_lr_std = list()
    precision_lr_std = list()
    recall_lr_std = list()
    auc_lr_std = list()

    accuracy_svm = list()
    f1_svm = list()
    precision_svm = list()
    recall_svm = list()
    auc_svm = list()

    accuracy_svm_std = list()
    f1_svm_std = list()
    precision_svm_std = list()
    recall_svm_std = list()
    auc_svm_std = list()

    accuracy_ada = list()
    f1_ada = list()
    precision_ada = list()
    recall_ada = list()
    auc_ada = list()

    accuracy_ada_std = list()
    f1_ada_std = list()
    precision_ada_std = list()
    recall_ada_std = list()
    auc_ada_std = list()

    accuracy_rf = list()
    f1_rf = list()
    precision_rf = list()
    recall_rf = list()
    auc_rf = list()

    accuracy_rf_std = list()
    f1_rf_std = list()
    precision_rf_std = list()
    recall_rf_std = list()
    auc_rf_std = list()

    accuracy_xgb = list()
    f1_xgb = list()
    precision_xgb = list()
    recall_xgb = list()
    auc_xgb = list()

    accuracy_xgb_std = list()
    f1_xgb_std = list()
    precision_xgb_std = list()
    recall_xgb_std = list()
    auc_xgb_std = list()

    accuracy_nn = list()
    f1_nn = list()
    precision_nn = list()
    recall_nn = list()
    auc_nn = list()

    accuracy_nn_std = list()
    f1_nn_std = list()
    precision_nn_std = list()
    recall_nn_std = list()
    auc_nn_std = list()

    # EXTERNAL VALIDATION LISTS

    accuracy_lr_ev = list()
    f1_lr_ev = list()
    precision_lr_ev = list()
    recall_lr_ev = list()
    auc_lr_ev = list()

    accuracy_lr_std_ev = list()
    f1_lr_std_ev = list()
    precision_lr_std_ev = list()
    recall_lr_std_ev = list()
    auc_lr_std_ev = list()

    accuracy_svm_ev = list()
    f1_svm_ev = list()
    precision_svm_ev = list()
    recall_svm_ev = list()
    auc_svm_ev = list()

    accuracy_svm_std_ev = list()
    f1_svm_std_ev = list()
    precision_svm_std_ev = list()
    recall_svm_std_ev = list()
    auc_svm_std_ev = list()

    accuracy_ada_ev = list()
    f1_ada_ev = list()
    precision_ada_ev = list()
    recall_ada_ev = list()
    auc_ada_ev = list()

    accuracy_ada_std_ev = list()
    f1_ada_std_ev = list()
    precision_ada_std_ev = list()
    recall_ada_std_ev = list()
    auc_ada_std_ev = list()

    accuracy_rf_ev = list()
    f1_rf_ev = list()
    precision_rf_ev = list()
    recall_rf_ev = list()
    auc_rf_ev = list()

    accuracy_rf_std_ev = list()
    f1_rf_std_ev = list()
    precision_rf_std_ev = list()
    recall_rf_std_ev = list()
    auc_rf_std_ev = list()

    accuracy_xgb_ev = list()
    f1_xgb_ev = list()
    precision_xgb_ev = list()
    recall_xgb_ev = list()
    auc_xgb_ev = list()

    accuracy_xgb_std_ev = list()
    f1_xgb_std_ev = list()
    precision_xgb_std_ev = list()
    recall_xgb_std_ev = list()
    auc_xgb_std_ev = list()

    accuracy_nn_ev = list()
    f1_nn_ev = list()
    precision_nn_ev = list()
    recall_nn_ev = list()
    auc_nn_ev = list()

    accuracy_nn_std_ev = list()
    f1_nn_std_ev = list()
    precision_nn_std_ev = list()
    recall_nn_std_ev = list()
    auc_nn_std_ev = list()

    # PARAMETERS

    params_lr = {'solver': ['newton-cg', 'lbfgs', 'liblinear'], 'penalty': ['l2','l1'], 'C': [100, 10, 1.0, 0.1, 0.01]}

    params_svm = {'C': [0.1, 0.5, 1,5, 2, 10, 50, 100, 500, 1000],
              'gamma': [0.5, 0.1, 1.5, 2, 0.01, 0.005, 0.001, 0.0005, 0.0001],'kernel': ['rbf']}

    params_ada = {'n_estimators': [100, 150, 200, 250, 300], 'learning_rate': [1.0, 1.5, 2.0, 2.5, 3, 3.5, 4.0],
              'algorithm': ['SAMME', 'SAMME.R']}
            
    params_rf = {'bootstrap': [True, False], 'class_weight': ['balanced'], 'max_depth': [9, 5, 3, 2, None],
                 'n_estimators': [100, 500], 'min_samples_split': [2, 3, 5, 7, 10]}

    params_xgb = {'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30], 'min_child_weight': [1, 5, 10],
                  'gamma': [0.5, 1, 5], 'subsample': [0.6, 0.8, 1.0], 'colsample_bytree': [0.6, 0.8, 1.0],
                  'max_depth': [3, 4, 5]}

    params_nn = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],'activation': ['tanh', 'relu'],'solver': ['sgd', 'adam'],'alpha': [0.0001, 0.05],'learning_rate': ['constant','adaptive']}
    
# start undersampling
    for k in range(n_repetitions):

        outer_results_accuracy_lr = list()
        outer_results_f1_lr = list()
        outer_results_precision_lr = list()
        outer_results_recall_lr = list()
        outer_results_auc_lr = list()

        outer_results_accuracy_svm = list()
        outer_results_f1_svm = list()
        outer_results_precision_svm = list()
        outer_results_recall_svm = list()
        outer_results_auc_svm = list()

        outer_results_accuracy_ada = list()
        outer_results_f1_ada = list()
        outer_results_precision_ada = list()
        outer_results_recall_ada = list()
        outer_results_auc_ada = list()

        outer_results_accuracy_m5 = list()
        outer_results_f1_m5 = list()
        outer_results_precision_m5 = list()
        outer_results_recall_m5 = list()
        outer_results_auc_m5 = list()

        outer_results_accuracy_xgb = list()
        outer_results_f1_xgb = list()
        outer_results_precision_xgb = list()
        outer_results_recall_xgb = list()
        outer_results_auc_xgb = list()

        outer_results_accuracy_nn = list()
        outer_results_f1_nn = list()
        outer_results_precision_nn = list()
        outer_results_recall_nn = list()
        outer_results_auc_nn = list()

        # OUTER EXTERNAL VALIDATION

        outer_results_accuracy_lr_ev = list()
        outer_results_f1_lr_ev = list()
        outer_results_precision_lr_ev = list()
        outer_results_recall_lr_ev = list()
        outer_results_auc_lr_ev = list()

        outer_results_accuracy_svm_ev = list()
        outer_results_f1_svm_ev = list()
        outer_results_precision_svm_ev = list()
        outer_results_recall_svm_ev = list()
        outer_results_auc_svm_ev = list()

        outer_results_accuracy_ada_ev = list()
        outer_results_f1_ada_ev = list()
        outer_results_precision_ada_ev = list()
        outer_results_recall_ada_ev = list()
        outer_results_auc_ada_ev = list()

        outer_results_accuracy_m5_ev = list()
        outer_results_f1_m5_ev = list()
        outer_results_precision_m5_ev = list()
        outer_results_recall_m5_ev = list()
        outer_results_auc_m5_ev = list()

        outer_results_accuracy_xgb_ev = list()
        outer_results_f1_xgb_ev = list()
        outer_results_precision_xgb_ev = list()
        outer_results_recall_xgb_ev = list()
        outer_results_auc_xgb_ev = list()

        outer_results_accuracy_nn_ev = list()
        outer_results_f1_nn_ev = list()
        outer_results_precision_nn_ev = list()
        outer_results_recall_nn_ev = list()
        outer_results_auc_nn_ev = list()

        # save outcome distribution
        figure_data_balanced = folder_figure + '/'+ 'balancedData{0}.png'.format(str(k))
        bar_y = [(y == 1).sum(), (y == 0).sum()]
        bar_x = ["1", "0"]
        plt.figure()
        splot = sns.barplot(bar_x, bar_y)
        for p in splot.patches:
            splot.annotate(format(p.get_height(), '.1f'),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center',
                           xytext=(0, 9),
                           textcoords='offset points')
        plt.xlabel("Disease")
        plt.ylabel("Number of subjects")
        plt.savefig(figure_data_balanced, bbox_inches='tight')
        plt.clf()

        j = 1
        if fold_type_out == 1:
            cv_outer = KFold(n_splits=n_folds_out, shuffle=True, random_state=seed)
        else:
            cv_outer = StratifiedKFold(n_splits=n_folds_out, shuffle=True, random_state=seed)

        # Repeated k-folds
        for train_idx, test_idx in cv_outer.split(X, y):
            print(('{} of KFold {}'.format(j, cv_outer.n_splits)))
            text_file_output.write('{} of KFold {} \n'.format(j, cv_outer.n_splits))
            train_data = X.iloc[train_idx, :]
            train_target = y.iloc[train_idx]
            val_data = X.iloc[test_idx, :]
            val_target = y.iloc[test_idx]
            participants_ID = Z.iloc[test_idx]

            # save outcome distribution
            namefilef = "balancedTrainingkfold{0}_{1}.png".format(str(k),str(j))
            figure_data_balanced = folder_figure+'/' + namefilef
            bar_y = [(train_target == 1).sum(), (train_target == 0).sum()]
            bar_x = ["1", "0"]
            plt.figure()
            splot = sns.barplot(bar_x, bar_y)
            for p in splot.patches:
                splot.annotate(format(p.get_height(), '.1f'),
                               (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='center',
                               xytext=(0, 9),
                               textcoords='offset points')
            plt.xlabel("Disease")
            plt.ylabel("Number of subjects")
            plt.savefig(figure_data_balanced, bbox_inches='tight')
            plt.clf()

            namefilef = "balancedTestingkfold{0}_{1}.png".format(str(k),str(j))
            figure_data_balanced = folder_figure+'/' + namefilef
            bar_y = [(val_target == 1).sum(), (val_target == 0).sum()]
            bar_x = ["1", "0"]
            plt.figure()
            splot = sns.barplot(bar_x, bar_y)

            for p in splot.patches:
                splot.annotate(format(p.get_height(), '.1f'),
                               (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='center',
                               xytext=(0, 9),
                               textcoords='offset points')
            plt.xlabel("Disease")
            plt.ylabel("Number of subjects")
            plt.savefig(figure_data_balanced, bbox_inches='tight')
            plt.clf()

            if fold_type_inn == 1:
                cv_inner = KFold(n_splits=n_folds_in, shuffle=True, random_state=seed)
            if fold_type_inn == 2:
                cv_inner = StratifiedKFold(n_splits=n_folds_in, shuffle=True, random_state=seed)

            # Logistic Regression
            model_lr = LogisticRegression(solver='liblinear')
            gd_search_lr = GridSearchCV(model_lr, params_lr, n_jobs=-1, cv=cv_inner).fit(train_data, train_target)

            best_model_lr = gd_search_lr.best_estimator_
            classifier_lr = best_model_lr.fit(train_data, train_target)
            y_hat_lr = classifier_lr.predict(val_data)

            ## save model
            nameModel = "LR_Modelkfold{0}_{1}.joblib".format(str(k), str(j))
            fileModel = folder_figure + '/' + nameModel
            joblib.dump(classifier_lr,fileModel)
            ## end save model

            outer_results_accuracy_lr.append(metrics.accuracy_score(val_target, y_hat_lr))
            outer_results_f1_lr.append(metrics.f1_score(val_target, y_hat_lr))
            outer_results_auc_lr.append(metrics.roc_auc_score(val_target, y_hat_lr))
            outer_results_precision_lr.append(metrics.precision_score(val_target, y_hat_lr))
            outer_results_recall_lr.append(metrics.recall_score(val_target, y_hat_lr))

            ## start external validation
            y_hat_lr_ev = classifier_lr.predict(val_data_ev)
            outer_results_accuracy_lr_ev.append(metrics.accuracy_score(val_target_ev, y_hat_lr_ev))
            outer_results_f1_lr_ev.append(metrics.f1_score(val_target_ev, y_hat_lr_ev))
            outer_results_auc_lr_ev.append(metrics.roc_auc_score(val_target_ev, y_hat_lr_ev))
            outer_results_precision_lr_ev.append(metrics.precision_score(val_target_ev, y_hat_lr_ev))
            outer_results_recall_lr_ev.append(metrics.recall_score(val_target_ev, y_hat_lr_ev))
            ## end external validation

            # SVM
            model_svm = SVC(class_weight="balanced")
            gd_search_svm = GridSearchCV(model_svm, params_svm, scoring='f1_macro', n_jobs=-1, cv=cv_inner).fit(train_data,
                                                                                                    train_target)
            best_model_svm = gd_search_svm.best_estimator_
            classifier_svm = best_model_svm.fit(train_data, train_target)
            y_hat_svm = classifier_svm.predict(val_data)

            ## save model
            nameModel = "SVM_Modelkfold{0}_{1}.joblib".format(str(k), str(j))
            fileModel = folder_figure + '/' + nameModel
            joblib.dump(classifier_svm,fileModel)
            ## end save model

            outer_results_accuracy_svm.append(metrics.accuracy_score(val_target, y_hat_svm))
            outer_results_f1_svm.append(metrics.f1_score(val_target, y_hat_svm))
            outer_results_auc_svm.append(metrics.roc_auc_score(val_target, y_hat_svm))
            outer_results_precision_svm.append(metrics.precision_score(val_target, y_hat_svm))
            outer_results_recall_svm.append(metrics.recall_score(val_target, y_hat_svm))

            ## start external validation
            y_hat_svm_ev = classifier_svm.predict(val_data_ev)
            outer_results_accuracy_svm_ev.append(metrics.accuracy_score(val_target_ev, y_hat_svm_ev))
            outer_results_f1_svm_ev.append(metrics.f1_score(val_target_ev, y_hat_svm_ev))
            outer_results_auc_svm_ev.append(metrics.roc_auc_score(val_target_ev, y_hat_svm_ev))
            outer_results_precision_svm_ev.append(metrics.precision_score(val_target_ev, y_hat_svm_ev))
            outer_results_recall_svm_ev.append(metrics.recall_score(val_target_ev, y_hat_svm_ev))
            ## end external validation

            # ADABOOST
            model_ada = AdaBoostClassifier()
            gd_search_ada = GridSearchCV(model_ada, params_ada, scoring='f1_macro', n_jobs=-1, cv=cv_inner).fit(train_data, train_target)

            best_model_ada = gd_search_ada.best_estimator_
            classifier_ada = best_model_ada.fit(train_data, train_target)
            y_pred_prob_ada = classifier_ada.predict_proba(val_data)[:, 1]
            auc_ad = metrics.roc_auc_score(val_target, y_pred_prob_ada)

            print("Val Acc ADA:", auc_ad, "Best GS Acc ADA:", gd_search_ada.best_score_, "Best Params ADA:", gd_search_ada.best_params_)
            text_file_output.write("Val Acc ADA: %f Best GS Acc ADA: %f \n"  %(auc_ad, gd_search_ada.best_score_))
            y_hat_ada = classifier_ada.predict(val_data)

            ## save model
            nameModel = "ADA_Modelkfold{0}_{1}.joblib".format(str(k), str(j))
            fileModel = folder_figure + '/' + nameModel
            joblib.dump(classifier_ada,fileModel)
            ## end save model

            outer_results_accuracy_ada.append(metrics.accuracy_score(val_target, y_hat_ada))
            outer_results_f1_ada.append(metrics.f1_score(val_target, y_hat_ada))
            outer_results_auc_ada.append(metrics.roc_auc_score(val_target, y_hat_ada))
            outer_results_precision_ada.append(metrics.precision_score(val_target, y_hat_ada))
            outer_results_recall_ada.append(metrics.recall_score(val_target, y_hat_ada))

            ## start external validation
            y_hat_ada_ev = classifier_ada.predict(val_data_ev)
            outer_results_accuracy_ada_ev.append(metrics.accuracy_score(val_target_ev, y_hat_ada_ev))
            outer_results_f1_ada_ev.append(metrics.f1_score(val_target_ev, y_hat_ada_ev))
            outer_results_auc_ada_ev.append(metrics.roc_auc_score(val_target_ev, y_hat_ada_ev))
            outer_results_precision_ada_ev.append(metrics.precision_score(val_target_ev, y_hat_ada_ev))
            outer_results_recall_ada_ev.append(metrics.recall_score(val_target_ev, y_hat_ada_ev))
            ## end external validation

            # RandomForest
            model = RandomForestClassifier(n_estimators=80, class_weight="balanced")
            gd_search = GridSearchCV(model, params_rf, scoring='f1_macro', n_jobs=-1, cv=cv_inner).fit(train_data,
                                                                                                       train_target)
            best_model = gd_search.best_estimator_
            classifier = best_model.fit(train_data, train_target)
            y_pred_prob = classifier.predict_proba(val_data)[:, 1]
            auc = metrics.roc_auc_score(val_target, y_pred_prob)

            print("Val Acc RF:", auc, "Best GS Acc RF:", gd_search.best_score_, "Best Params RF:", gd_search.best_params_)
            text_file_output.write("Val Acc: %f Best GS Acc RF: %f \n"  %(auc, gd_search.best_score_))
            y_hat = classifier.predict(val_data)

            ## save model
            nameModel = "RF_Modelkfold{0}_{1}.joblib".format(str(k), str(j))
            fileModel = folder_figure + '/' + nameModel
            joblib.dump(classifier,fileModel)
            ## end save model

            outer_results_accuracy_m5.append(metrics.accuracy_score(val_target, y_hat))
            outer_results_f1_m5.append(metrics.f1_score(val_target, y_hat))
            outer_results_auc_m5.append(metrics.roc_auc_score(val_target, y_hat))
            outer_results_precision_m5.append(metrics.precision_score(val_target, y_hat))
            outer_results_recall_m5.append(metrics.recall_score(val_target, y_hat))

            ## start external validation
            y_hat_ev = classifier.predict(val_data_ev)
            outer_results_accuracy_m5_ev.append(metrics.accuracy_score(val_target_ev, y_hat_ev))
            outer_results_f1_m5_ev.append(metrics.f1_score(val_target_ev, y_hat_ev))
            outer_results_auc_m5_ev.append(metrics.roc_auc_score(val_target_ev, y_hat_ev))
            outer_results_precision_m5_ev.append(metrics.precision_score(val_target_ev, y_hat_ev))
            outer_results_recall_m5_ev.append(metrics.recall_score(val_target_ev, y_hat_ev))
            ## end external validation

            # # XGboost
            model_xgb = XGBClassifier()
            gd_search_xgb = GridSearchCV(model_xgb, params_xgb, scoring='f1_macro', n_jobs=-1, cv=cv_inner).fit(train_data,
                                                                                                             train_target)
            best_model_xgb = gd_search_xgb.best_estimator_
            classifier_xgb = best_model_xgb.fit(train_data, train_target)
            y_hat_xgb = classifier_xgb.predict(val_data)

            ## save model
            nameModel = "XGB_Modelkfold{0}_{1}.joblib".format(str(k), str(j))
            fileModel = folder_figure + '/' + nameModel
            joblib.dump(classifier_xgb,fileModel)
            ## end save model

            outer_results_accuracy_xgb.append(metrics.accuracy_score(val_target, y_hat_xgb))
            outer_results_f1_xgb.append(metrics.f1_score(val_target, y_hat_xgb))
            outer_results_auc_xgb.append(metrics.roc_auc_score(val_target, y_hat_xgb))
            outer_results_precision_xgb.append(metrics.precision_score(val_target, y_hat_xgb))
            outer_results_recall_xgb.append(metrics.recall_score(val_target, y_hat_xgb))

            ## start external validation
            y_hat_xgb_ev = classifier_xgb.predict(val_data_ev)
            outer_results_accuracy_xgb_ev.append(metrics.accuracy_score(val_target_ev, y_hat_xgb_ev))
            outer_results_f1_xgb_ev.append(metrics.f1_score(val_target_ev, y_hat_xgb_ev))
            outer_results_auc_xgb_ev.append(metrics.roc_auc_score(val_target_ev, y_hat_xgb_ev))
            outer_results_precision_xgb_ev.append(metrics.precision_score(val_target_ev, y_hat_xgb_ev))
            outer_results_recall_xgb_ev.append(metrics.recall_score(val_target_ev, y_hat_xgb_ev))
            ## end external validation

            # # Neural Net
            model_nn = MLPClassifier()
            gd_search_nn = GridSearchCV(model_nn, params_nn, n_jobs=-1, cv=cv_inner).fit(train_data, train_target)
           
            best_model_nn = gd_search_nn.best_estimator_
            classifier_nn = best_model_nn.fit(train_data, train_target)
            y_pred_prob_nn = classifier_nn.predict_proba(val_data)[:, 1]
            auc_n = metrics.roc_auc_score(val_target, y_pred_prob_nn)

            print("Val Acc NN:", auc_n, "Best GS Acc NN:", gd_search_nn.best_score_, "Best Params NN:", gd_search_nn.best_params_)
            text_file_output.write("Val Acc NN: %f Best GS Acc NN: %f \n"  %(auc_n, gd_search_nn.best_score_))
            y_hat_nn = classifier_nn.predict(val_data)

            ## save model
            nameModel = "NN_Modelkfold{0}_{1}.joblib".format(str(k), str(j))
            fileModel = folder_figure + '/' + nameModel
            joblib.dump(classifier_nn,fileModel)
            ## end save model

            outer_results_accuracy_nn.append(metrics.accuracy_score(val_target, y_hat_nn))
            outer_results_f1_nn.append(metrics.f1_score(val_target, y_hat_nn))
            outer_results_auc_nn.append(metrics.roc_auc_score(val_target, y_hat_nn))
            outer_results_precision_nn.append(metrics.precision_score(val_target, y_hat_nn))
            outer_results_recall_nn.append(metrics.recall_score(val_target, y_hat_nn))
           
            ## start external validation
            y_hat_nn_ev = classifier_nn.predict(val_data_ev)
            outer_results_accuracy_nn_ev.append(metrics.accuracy_score(val_target_ev, y_hat_nn_ev))
            outer_results_f1_nn_ev.append(metrics.f1_score(val_target_ev, y_hat_nn_ev))
            outer_results_auc_nn_ev.append(metrics.roc_auc_score(val_target_ev, y_hat_nn_ev))
            outer_results_precision_nn_ev.append(metrics.precision_score(val_target_ev, y_hat_nn_ev))
            outer_results_recall_nn_ev.append(metrics.recall_score(val_target_ev, y_hat_nn_ev))
            ## end external validation

            namefilefexcel = "FeatImport_AD{0}_{1}.xlsx".format(str(k), str(j))
            feature_importance_FILE_AD = folder_figure + '/' + namefilefexcel

            namefilefexcel = "FeatImport_LR{0}_{1}.xlsx".format(str(k), str(j))
            feature_importance_FILE_LR = folder_figure + '/' + namefilefexcel

            namefilefexcel = "FeatImport_RF{0}_{1}.xlsx".format(str(k), str(j))
            feature_importance_FILE_RF = folder_figure + '/' + namefilefexcel

            namefilefexcel = "FeatImport_XG{0}_{1}.xlsx".format(str(k), str(j))
            feature_importance_FILE_XG = folder_figure + '/' + namefilefexcel

            namefileOutput_ADLRRFXGB = "Output_ADLRRFXGB{0}_{1}.xlsx".format(str(k), str(j))
            output_FILE_ADLRRFXGB = folder_figure + '/' + namefileOutput_ADLRRFXGB

            print(output_FILE_ADLRRFXGB)
            
            plt.clf()
            plt.rcParams.update({'font.size': 10})
            plt.figure()
            plot_feature_importance(classifier_ada.feature_importances_, train_data.columns, 'ADA',feature_importance_FILE_AD)
            namefilef = "FeatImport_AD{0}_{1}.png".format(str(k),str(j))
            feature_importance_AD = folder_figure + '/' + namefilef
            plt.savefig(feature_importance_AD, bbox_inches='tight')
            plt.clf()

            plt.clf()
            plt.rcParams.update({'font.size': 10})
            plt.figure()
            plot_feature_importance(classifier_lr.coef_[0], train_data.columns, 'LOGISTIC REGRESSION',feature_importance_FILE_LR)
            namefilef = "FeatImport_LR{0}_{1}.png".format(str(k),str(j))
            feature_importance_LR = folder_figure + '/' + namefilef
            plt.savefig(feature_importance_LR, bbox_inches='tight')
            plt.clf()

            plt.clf()
            plt.rcParams.update({'font.size': 10})
            plt.figure()
            plot_feature_importance(classifier.feature_importances_, train_data.columns, 'RANDOM FOREST',feature_importance_FILE_RF)
            namefilef = "FeatImport_RF{0}_{1}.png".format(str(k),str(j))
            feature_importance_RF = folder_figure + '/' + namefilef
            plt.savefig(feature_importance_RF, bbox_inches='tight')
            plt.clf()

            plt.clf()
            plt.rcParams.update({'font.size': 10})
            plt.figure()
            plot_feature_importance(classifier_xgb.feature_importances_, train_data.columns, 'XGBOOST',feature_importance_FILE_XG)
            namefilef = "FeatImport_XG{0}_{1}.png".format(str(k),str(j))
            feature_importance_XG = folder_figure + '/' + namefilef
            plt.savefig(feature_importance_XG, bbox_inches='tight')
            plt.clf()
            
            datacl = {'feature_gt': val_target, 'feature_xgb': y_hat_xgb, 'feature_rf': y_hat,
                      'feature_lr': y_hat_lr, 'feature_ad': y_hat_ada, 'participants_id': participants_ID}

            fi_dfc  = pd.DataFrame(datacl)
            print(fi_dfc)
            
            fi_dfc.to_csv(output_FILE_ADLRRFXGB, index=False)

            j = j + 1

        auc_nn.append(mean(outer_results_auc_nn))
        f1_nn.append(mean(outer_results_f1_nn))
        precision_nn.append(mean(outer_results_precision_nn))
        recall_nn.append(mean(outer_results_recall_nn))
        accuracy_nn.append(mean(outer_results_accuracy_nn))

        auc_nn_std.append(std(outer_results_auc_nn))
        f1_nn_std.append(std(outer_results_f1_nn))
        precision_nn_std.append(std(outer_results_precision_nn))
        recall_nn_std.append(std(outer_results_recall_nn))
        accuracy_nn_std.append(std(outer_results_accuracy_nn))

        auc_xgb.append(mean(outer_results_auc_xgb))
        f1_xgb.append(mean(outer_results_f1_xgb))
        precision_xgb.append(mean(outer_results_precision_xgb))
        recall_xgb.append(mean(outer_results_recall_xgb))
        accuracy_xgb.append(mean(outer_results_accuracy_xgb))

        auc_xgb_std.append(std(outer_results_auc_xgb))
        f1_xgb_std.append(std(outer_results_f1_xgb))
        precision_xgb_std.append(std(outer_results_precision_xgb))
        recall_xgb_std.append(std(outer_results_recall_xgb))
        accuracy_xgb_std.append(std(outer_results_accuracy_xgb))

        auc_rf.append(mean(outer_results_auc_m5))
        f1_rf.append(mean(outer_results_f1_m5))
        precision_rf.append(mean(outer_results_precision_m5))
        recall_rf.append(mean(outer_results_recall_m5))
        accuracy_rf.append(mean(outer_results_accuracy_m5))

        auc_rf_std.append(std(outer_results_auc_m5))
        f1_rf_std.append(std(outer_results_f1_m5))
        precision_rf_std.append(std(outer_results_precision_m5))
        recall_rf_std.append(std(outer_results_recall_m5))
        accuracy_rf_std.append(std(outer_results_accuracy_m5))

        auc_lr.append(mean(outer_results_auc_lr))
        f1_lr.append(mean(outer_results_f1_lr))
        precision_lr.append(mean(outer_results_precision_lr))
        recall_lr.append(mean(outer_results_recall_lr))
        accuracy_lr.append(mean(outer_results_accuracy_lr))

        auc_lr_std.append(std(outer_results_auc_lr))
        f1_lr_std.append(std(outer_results_f1_lr))
        precision_lr_std.append(std(outer_results_precision_lr))
        recall_lr_std.append(std(outer_results_recall_lr))
        accuracy_lr_std.append(std(outer_results_accuracy_lr))

        auc_svm.append(mean(outer_results_auc_svm))
        f1_svm.append(mean(outer_results_f1_svm))
        precision_svm.append(mean(outer_results_precision_svm))
        recall_svm.append(mean(outer_results_recall_svm))
        accuracy_svm.append(mean(outer_results_accuracy_svm))

        auc_svm_std.append(std(outer_results_auc_svm))
        f1_svm_std.append(std(outer_results_f1_svm))
        precision_svm_std.append(std(outer_results_precision_svm))
        recall_svm_std.append(std(outer_results_recall_svm))
        accuracy_svm_std.append(std(outer_results_accuracy_svm))

        auc_ada.append(mean(outer_results_auc_ada))
        f1_ada.append(mean(outer_results_f1_ada))
        precision_ada.append(mean(outer_results_precision_ada))
        recall_ada.append(mean(outer_results_recall_ada))
        accuracy_ada.append(mean(outer_results_accuracy_ada))

        auc_ada_std.append(std(outer_results_auc_ada))
        f1_ada_std.append(std(outer_results_f1_ada))
        precision_ada_std.append(std(outer_results_precision_ada))
        recall_ada_std.append(std(outer_results_recall_ada))
        accuracy_ada_std.append(std(outer_results_accuracy_ada))

        # EXTERNAL VALIDATION FINAL METRICS
        auc_nn_ev.append(mean(outer_results_auc_nn_ev))
        f1_nn_ev.append(mean(outer_results_f1_nn_ev))
        precision_nn_ev.append(mean(outer_results_precision_nn_ev))
        recall_nn_ev.append(mean(outer_results_recall_nn_ev))
        accuracy_nn_ev.append(mean(outer_results_accuracy_nn_ev))

        auc_nn_std_ev.append(std(outer_results_auc_nn_ev))
        f1_nn_std_ev.append(std(outer_results_f1_nn_ev))
        precision_nn_std_ev.append(std(outer_results_precision_nn_ev))
        recall_nn_std_ev.append(std(outer_results_recall_nn_ev))
        accuracy_nn_std_ev.append(std(outer_results_accuracy_nn_ev))

        auc_xgb_ev.append(mean(outer_results_auc_xgb_ev))
        f1_xgb_ev.append(mean(outer_results_f1_xgb_ev))
        precision_xgb_ev.append(mean(outer_results_precision_xgb_ev))
        recall_xgb_ev.append(mean(outer_results_recall_xgb_ev))
        accuracy_xgb_ev.append(mean(outer_results_accuracy_xgb_ev))

        auc_xgb_std_ev.append(std(outer_results_auc_xgb_ev))
        f1_xgb_std_ev.append(std(outer_results_f1_xgb_ev))
        precision_xgb_std_ev.append(std(outer_results_precision_xgb_ev))
        recall_xgb_std_ev.append(std(outer_results_recall_xgb_ev))
        accuracy_xgb_std_ev.append(std(outer_results_accuracy_xgb_ev))

        auc_rf_ev.append(mean(outer_results_auc_m5_ev))
        f1_rf_ev.append(mean(outer_results_f1_m5_ev))
        precision_rf_ev.append(mean(outer_results_precision_m5_ev))
        recall_rf_ev.append(mean(outer_results_recall_m5_ev))
        accuracy_rf_ev.append(mean(outer_results_accuracy_m5_ev))

        auc_rf_std_ev.append(std(outer_results_auc_m5_ev))
        f1_rf_std_ev.append(std(outer_results_f1_m5_ev))
        precision_rf_std_ev.append(std(outer_results_precision_m5_ev))
        recall_rf_std_ev.append(std(outer_results_recall_m5_ev))
        accuracy_rf_std_ev.append(std(outer_results_accuracy_m5_ev))

        auc_lr_ev.append(mean(outer_results_auc_lr_ev))
        f1_lr_ev.append(mean(outer_results_f1_lr_ev))
        precision_lr_ev.append(mean(outer_results_precision_lr_ev))
        recall_lr_ev.append(mean(outer_results_recall_lr_ev))
        accuracy_lr_ev.append(mean(outer_results_accuracy_lr_ev))

        auc_lr_std_ev.append(std(outer_results_auc_lr_ev))
        f1_lr_std_ev.append(std(outer_results_f1_lr_ev))
        precision_lr_std_ev.append(std(outer_results_precision_lr_ev))
        recall_lr_std_ev.append(std(outer_results_recall_lr_ev))
        accuracy_lr_std_ev.append(std(outer_results_accuracy_lr_ev))

        auc_svm_ev.append(mean(outer_results_auc_svm_ev))
        f1_svm_ev.append(mean(outer_results_f1_svm_ev))
        precision_svm_ev.append(mean(outer_results_precision_svm_ev))
        recall_svm_ev.append(mean(outer_results_recall_svm_ev))
        accuracy_svm_ev.append(mean(outer_results_accuracy_svm_ev))

        auc_svm_std_ev.append(std(outer_results_auc_svm_ev))
        f1_svm_std_ev.append(std(outer_results_f1_svm_ev))
        precision_svm_std_ev.append(std(outer_results_precision_svm_ev))
        recall_svm_std_ev.append(std(outer_results_recall_svm_ev))
        accuracy_svm_std_ev.append(std(outer_results_accuracy_svm_ev))

        auc_ada_ev.append(mean(outer_results_auc_ada_ev))
        f1_ada_ev.append(mean(outer_results_f1_ada_ev))
        precision_ada_ev.append(mean(outer_results_precision_ada_ev))
        recall_ada_ev.append(mean(outer_results_recall_ada_ev))
        accuracy_ada_ev.append(mean(outer_results_accuracy_ada_ev))

        auc_ada_std_ev.append(std(outer_results_auc_ada_ev))
        f1_ada_std_ev.append(std(outer_results_f1_ada_ev))
        precision_ada_std_ev.append(std(outer_results_precision_ada_ev))
        recall_ada_std_ev.append(std(outer_results_recall_ada_ev))
        accuracy_ada_std_ev.append(std(outer_results_accuracy_ada_ev))
        # END FINAL EXTERNAL VALIDATION METRICS

        text_file_output.write("====> NEURAL NET \n")
        text_file_output.write('AUC: %.2f (%.2f) (%.2f) \n' % (mean(auc_nn), std(auc_nn), mean(auc_nn_std)))
        text_file_output.write('F1: %.2f (%.2f) (%.2f) \n' % (mean(f1_nn), std(f1_nn), mean(f1_nn_std)))
        text_file_output.write('precision: %.2f (%.2f) (%.2f) \n' % (mean(precision_nn), std(precision_nn), mean(precision_nn_std)))
        text_file_output.write('recall: %.2f (%.2f) (%.2f) \n' % (mean(recall_nn), std(recall_nn), mean(recall_nn_std)))
        text_file_output.write('accuracy: %.2f (%.2f) (%.2f) \n' % (mean(accuracy_nn), std(accuracy_nn), mean(accuracy_nn_std)))
        text_file_output.write("====> \n")
        text_file_output.write('max AUC: %.2f \n' % (max(auc_nn)))
        text_file_output.write('max F1: %.2f \n' % (max(f1_nn)))
        text_file_output.write('max precision: %.2f \n' % (max(precision_nn)))
        text_file_output.write('max recall: %.2f \n' % (max(recall_nn)))
        text_file_output.write('max accuracy: %.2f \n' % (max(accuracy_nn)))

        text_file_output.write("====> XGBOOST \n")
        text_file_output.write('AUC: %.2f (%.2f) (%.2f) \n' % (mean(auc_xgb), std(auc_xgb), mean(auc_xgb_std)))
        text_file_output.write('F1: %.2f (%.2f) (%.2f) \n' % (mean(f1_xgb), std(f1_xgb), mean(f1_xgb_std)))
        text_file_output.write('precision: %.2f (%.2f) (%.2f) \n' % (mean(precision_xgb), std(precision_xgb), mean(precision_xgb_std)))
        text_file_output.write('recall: %.2f (%.2f) (%.2f) \n' % (mean(recall_xgb), std(recall_xgb), mean(recall_xgb_std)))
        text_file_output.write('accuracy: %.2f (%.2f) (%.2f) \n' % (mean(accuracy_xgb), std(accuracy_xgb), mean(accuracy_xgb_std)))
        text_file_output.write("====> \n")
        text_file_output.write('max AUC: %.2f \n' % (max(auc_xgb)))
        text_file_output.write('max F1: %.2f \n' % (max(f1_xgb)))
        text_file_output.write('max precision: %.2f \n' % (max(precision_xgb)))
        text_file_output.write('max recall: %.2f \n' % (max(recall_xgb)))
        text_file_output.write('max accuracy: %.2f \n' % (max(accuracy_xgb)))

        text_file_output.write("====> RANDOM FOREST \n")
        text_file_output.write('AUC: %.2f (%.2f) (%.2f) \n' % (mean(auc_rf), std(auc_rf), mean(auc_rf_std)))
        text_file_output.write('F1: %.2f (%.2f) (%.2f) \n' % (mean(f1_rf), std(f1_rf), mean(f1_rf_std)))
        text_file_output.write('precision: %.2f (%.2f) (%.2f) \n' % (mean(precision_rf), std(precision_rf), mean(precision_rf_std)))
        text_file_output.write('recall: %.2f (%.2f) (%.2f) \n' % (mean(recall_rf), std(recall_rf), mean(recall_rf_std)))
        text_file_output.write('accuracy: %.2f (%.2f) (%.2f) \n' % (mean(accuracy_rf), std(accuracy_rf), mean(accuracy_rf_std)))
        text_file_output.write("====> \n")
        text_file_output.write('max AUC: %.2f \n' % (max(auc_rf)))
        text_file_output.write('max F1: %.2f \n' % (max(f1_rf)))
        text_file_output.write('max precision: %.2f \n' % (max(precision_rf)))
        text_file_output.write('max recall: %.2f \n' % (max(recall_rf)))
        text_file_output.write('max accuracy: %.2f \n' % (max(accuracy_rf)))

        text_file_output.write("====> LOGISTIC REGRESSION \n")
        text_file_output.write('AUC: %.2f (%.2f) (%.2f) \n' % (mean(auc_lr), std(auc_lr), mean(auc_lr_std)))
        text_file_output.write('F1: %.2f (%.2f) (%.2f) \n' % (mean(f1_lr), std(f1_lr), mean(f1_lr_std)))
        text_file_output.write('precision: %.2f (%.2f) (%.2f) \n' % (mean(precision_lr), std(precision_lr), mean(precision_lr_std)))
        text_file_output.write('recall: %.2f (%.2f) (%.2f) \n' % (mean(recall_lr), std(recall_lr), mean(recall_lr_std)))
        text_file_output.write('accuracy: %.2f (%.2f) (%.2f) \n' % (mean(accuracy_lr), std(accuracy_lr), mean(accuracy_lr_std)))
        text_file_output.write("====> \n")
        text_file_output.write('max AUC: %.2f \n' % (max(auc_lr)))
        text_file_output.write('max F1: %.2f \n' % (max(f1_lr)))
        text_file_output.write('max precision: %.2f \n' % (max(precision_lr)))
        text_file_output.write('max recall: %.2f \n' % (max(recall_lr)))
        text_file_output.write('max accuracy: %.2f \n' % (max(accuracy_lr)))
        
        text_file_output.write("====> ADA \n")
        text_file_output.write('AUC: %.2f (%.2f) (%.2f) \n' % (mean(auc_ada), std(auc_ada), mean(auc_ada_std)))
        text_file_output.write('F1: %.2f (%.2f) (%.2f) \n' % (mean(f1_ada), std(f1_ada), mean(f1_ada_std)))
        text_file_output.write('precision: %.2f (%.2f) (%.2f) \n' % (mean(precision_ada), std(precision_ada), mean(precision_ada_std)))
        text_file_output.write('recall: %.2f (%.2f) (%.2f) \n' % (mean(recall_ada), std(recall_ada), mean(recall_ada_std)))
        text_file_output.write('accuracy: %.2f (%.2f) (%.2f) \n' % (mean(accuracy_ada), std(accuracy_ada), mean(accuracy_ada_std)))
        text_file_output.write("====> \n")
        text_file_output.write('max AUC: %.2f \n' % (max(auc_ada)))
        text_file_output.write('max F1: %.2f \n' % (max(f1_ada)))
        text_file_output.write('max precision: %.2f \n' % (max(precision_ada)))
        text_file_output.write('max recall: %.2f \n' % (max(recall_ada)))
        text_file_output.write('max accuracy: %.2f \n' % (max(accuracy_ada)))

        text_file_output.write("====> SVM \n")
        text_file_output.write('AUC: %.2f (%.2f) (%.2f) \n' % (mean(auc_svm), std(auc_svm), mean(auc_svm_std)))
        text_file_output.write('F1: %.2f (%.2f) (%.2f) \n' % (mean(f1_svm), std(f1_svm), mean(f1_svm_std)))
        text_file_output.write('precision: %.2f (%.2f) (%.2f) \n' % (mean(precision_svm), std(precision_svm), mean(precision_svm_std)))
        text_file_output.write('recall: %.2f (%.2f) (%.2f) \n' % (mean(recall_svm), std(recall_svm), mean(recall_svm_std)))
        text_file_output.write('accuracy: %.2f (%.2f) (%.2f) \n' % (mean(accuracy_svm), std(accuracy_svm), mean(accuracy_svm_std)))
        text_file_output.write("====> \n")
        text_file_output.write('max AUC: %.2f \n' % (max(auc_svm)))
        text_file_output.write('max F1: %.2f \n' % (max(f1_svm)))
        text_file_output.write('max precision: %.2f \n' % (max(precision_svm)))
        text_file_output.write('max recall: %.2f \n' % (max(recall_svm)))
        text_file_output.write('max accuracy: %.2f \n' % (max(accuracy_svm)))

        # FILE EXTERNAL VALIDATION 
        text_file_output.write("====> NEURAL NET EV\n")
        text_file_output.write('AUC: %.2f (%.2f) (%.2f) \n' % (mean(auc_nn_ev), std(auc_nn_ev), mean(auc_nn_std_ev)))
        text_file_output.write('F1: %.2f (%.2f) (%.2f) \n' % (mean(f1_nn_ev), std(f1_nn_ev), mean(f1_nn_std_ev)))
        text_file_output.write('precision: %.2f (%.2f) (%.2f) \n' % (mean(precision_nn_ev), std(precision_nn_ev), mean(precision_nn_std_ev)))
        text_file_output.write('recall: %.2f (%.2f) (%.2f) \n' % (mean(recall_nn_ev), std(recall_nn_ev), mean(recall_nn_std_ev)))
        text_file_output.write('accuracy: %.2f (%.2f) (%.2f) \n' % (mean(accuracy_nn_ev), std(accuracy_nn_ev), mean(accuracy_nn_std_ev)))
        text_file_output.write("====> \n")
        text_file_output.write('max AUC: %.2f \n' % (max(auc_nn_ev)))
        text_file_output.write('max F1: %.2f \n' % (max(f1_nn_ev)))
        text_file_output.write('max precision: %.2f \n' % (max(precision_nn_ev)))
        text_file_output.write('max recall: %.2f \n' % (max(recall_nn_ev)))
        text_file_output.write('max accuracy: %.2f \n' % (max(accuracy_nn_ev)))

        text_file_output.write("====> XGBOOST EV\n")
        text_file_output.write('AUC: %.2f (%.2f) (%.2f) \n' % (mean(auc_xgb_ev), std(auc_xgb_ev), mean(auc_xgb_std_ev)))
        text_file_output.write('F1: %.2f (%.2f) (%.2f) \n' % (mean(f1_xgb_ev), std(f1_xgb_ev), mean(f1_xgb_std_ev)))
        text_file_output.write('precision: %.2f (%.2f) (%.2f) \n' % (mean(precision_xgb_ev), std(precision_xgb_ev), mean(precision_xgb_std_ev)))
        text_file_output.write('recall: %.2f (%.2f) (%.2f) \n' % (mean(recall_xgb_ev), std(recall_xgb_ev), mean(recall_xgb_std_ev)))
        text_file_output.write('accuracy: %.2f (%.2f) (%.2f) \n' % (mean(accuracy_xgb_ev), std(accuracy_xgb_ev), mean(accuracy_xgb_std_ev)))
        text_file_output.write("====> \n")
        text_file_output.write('max AUC: %.2f \n' % (max(auc_xgb_ev)))
        text_file_output.write('max F1: %.2f \n' % (max(f1_xgb_ev)))
        text_file_output.write('max precision: %.2f \n' % (max(precision_xgb_ev)))
        text_file_output.write('max recall: %.2f \n' % (max(recall_xgb_ev)))
        text_file_output.write('max accuracy: %.2f \n' % (max(accuracy_xgb_ev)))

        text_file_output.write("====> RANDOM FOREST EV\n")
        text_file_output.write('AUC: %.2f (%.2f) (%.2f) \n' % (mean(auc_rf_ev), std(auc_rf_ev), mean(auc_rf_std_ev)))
        text_file_output.write('F1: %.2f (%.2f) (%.2f) \n' % (mean(f1_rf_ev), std(f1_rf_ev), mean(f1_rf_std_ev)))
        text_file_output.write('precision: %.2f (%.2f) (%.2f) \n' % (mean(precision_rf_ev), std(precision_rf_ev), mean(precision_rf_std_ev)))
        text_file_output.write('recall: %.2f (%.2f) (%.2f) \n' % (mean(recall_rf_ev), std(recall_rf_ev), mean(recall_rf_std_ev)))
        text_file_output.write('accuracy: %.2f (%.2f) (%.2f) \n' % (mean(accuracy_rf_ev), std(accuracy_rf_ev), mean(accuracy_rf_std_ev)))
        text_file_output.write("====> \n")
        text_file_output.write('max AUC: %.2f \n' % (max(auc_rf_ev)))
        text_file_output.write('max F1: %.2f \n' % (max(f1_rf_ev)))
        text_file_output.write('max precision: %.2f \n' % (max(precision_rf_ev)))
        text_file_output.write('max recall: %.2f \n' % (max(recall_rf_ev)))
        text_file_output.write('max accuracy: %.2f \n' % (max(accuracy_rf_ev)))
        
        text_file_output.write("====> ADA EV\n")
        text_file_output.write('AUC: %.2f (%.2f) (%.2f) \n' % (mean(auc_ada_ev), std(auc_ada_ev), mean(auc_ada_std_ev)))
        text_file_output.write('F1: %.2f (%.2f) (%.2f) \n' % (mean(f1_ada_ev), std(f1_ada_ev), mean(f1_ada_std_ev)))
        text_file_output.write('precision: %.2f (%.2f) (%.2f) \n' % (mean(precision_ada_ev), std(precision_ada_ev), mean(precision_ada_std_ev)))
        text_file_output.write('recall: %.2f (%.2f) (%.2f) \n' % (mean(recall_ada_ev), std(recall_ada_ev), mean(recall_ada_std_ev)))
        text_file_output.write('accuracy: %.2f (%.2f) (%.2f) \n' % (mean(accuracy_ada_ev), std(accuracy_ada_ev), mean(accuracy_ada_std_ev)))
        text_file_output.write("====> \n")
        text_file_output.write('max AUC: %.2f \n' % (max(auc_ada_ev)))
        text_file_output.write('max F1: %.2f \n' % (max(f1_ada_ev)))
        text_file_output.write('max precision: %.2f \n' % (max(precision_ada_ev)))
        text_file_output.write('max recall: %.2f \n' % (max(recall_ada_ev)))
        text_file_output.write('max accuracy: %.2f \n' % (max(accuracy_ada_ev)))

        text_file_output.write("====> SVM EV\n")
        text_file_output.write('AUC: %.2f (%.2f) (%.2f) \n' % (mean(auc_svm_ev), std(auc_svm_ev), mean(auc_svm_std_ev)))
        text_file_output.write('F1: %.2f (%.2f) (%.2f) \n' % (mean(f1_svm_ev), std(f1_svm_ev), mean(f1_svm_std_ev)))
        text_file_output.write('precision: %.2f (%.2f) (%.2f) \n' % (mean(precision_svm_ev), std(precision_svm_ev), mean(precision_svm_std_ev)))
        text_file_output.write('recall: %.2f (%.2f) (%.2f) \n' % (mean(recall_svm_ev), std(recall_svm_ev), mean(recall_svm_std_ev)))
        text_file_output.write('accuracy: %.2f (%.2f) (%.2f) \n' % (mean(accuracy_svm_ev), std(accuracy_svm_ev), mean(accuracy_svm_std_ev)))
        text_file_output.write("====> \n")
        text_file_output.write('max AUC: %.2f \n' % (max(auc_svm_ev)))
        text_file_output.write('max F1: %.2f \n' % (max(f1_svm_ev)))
        text_file_output.write('max precision: %.2f \n' % (max(precision_svm_ev)))
        text_file_output.write('max recall: %.2f \n' % (max(recall_svm_ev)))
        text_file_output.write('max accuracy: %.2f \n' % (max(accuracy_svm_ev)))

        text_file_output.write("====> LOGISTIC REGRESSION EV\n")
        text_file_output.write('AUC: %.2f (%.2f) (%.2f) \n' % (mean(auc_lr_ev), std(auc_lr_ev), mean(auc_lr_std_ev)))
        text_file_output.write('F1: %.2f (%.2f) (%.2f) \n' % (mean(f1_lr_ev), std(f1_lr_ev), mean(f1_lr_std_ev)))
        text_file_output.write('precision: %.2f (%.2f) (%.2f) \n' % (mean(precision_lr_ev), std(precision_lr_ev), mean(precision_lr_std_ev)))
        text_file_output.write('recall: %.2f (%.2f) (%.2f) \n' % (mean(recall_lr_ev), std(recall_lr_ev), mean(recall_lr_std_ev)))
        text_file_output.write('accuracy: %.2f (%.2f) (%.2f) \n' % (mean(accuracy_lr_ev), std(accuracy_lr_ev), mean(accuracy_lr_std_ev)))
        text_file_output.write("====> \n")
        text_file_output.write('max AUC: %.2f \n' % (max(auc_lr_ev)))
        text_file_output.write('max F1: %.2f \n' % (max(f1_lr_ev)))
        text_file_output.write('max precision: %.2f \n' % (max(precision_lr_ev)))
        text_file_output.write('max recall: %.2f \n' % (max(recall_lr_ev)))
        text_file_output.write('max accuracy: %.2f \n' % (max(accuracy_lr_ev)))

    text_file_output.write(tac())
    text_file_output.close()