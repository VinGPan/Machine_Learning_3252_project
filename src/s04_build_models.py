import numpy as np
from src.utils import read_yml

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from scipy.stats import reciprocal, uniform
from scipy.stats import geom, expon

from sklearn.externals import joblib
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import os
import os.path


@ignore_warnings(category=ConvergenceWarning)
def build_models(yml_name):
    configs = read_yml(yml_name)
    X = np.load("data/" + configs["experiment"]["name"] + "_X.npy")
    y = np.load("data/" + configs["experiment"]["name"] + "_y.npy")

    train_proportion = 0.9
    train_end = int(X.shape[0] * train_proportion)

    X_train = X[0:train_end, :]
    y_train = y[0:train_end]

    cv_end = int(X_train.shape[0] * train_proportion)

    X_val = X[train_end:, :]
    y_val = y[train_end:]

    split_index = [-1 if i < cv_end else 0 for i in range(X_train.shape[0])]
    pds = PredefinedSplit(test_fold=split_index)

    try:
        os.mkdir("output/" + configs["experiment"]["name"] + "/")
    except:
        pass

    model_scores = {}
    all_scores = []
    tot_classes = np.unique(y).shape[0]

    for clf_str in configs["models"]["classifiers"]:
        for preproc_str in configs["models"]["preprocs"]:
            for transforms_str in configs["models"]["transforms"]:
                steps = [('imputer', SimpleImputer(strategy='mean'))]
                param_grid = [{}]
                ############################################
                if preproc_str == 'min_max':
                    steps.append(('preprocs', MinMaxScaler()))
                elif preproc_str == 'standard_scalar':
                    steps.append(('preprocs', StandardScaler()))
                ############################################

                ############################################
                if transforms_str == 'pca':
                    steps.append(('transforms', PCA()))
                    param_grid[0]["transforms__n_components"] = [3, 5, 10]
                elif transforms_str == 'kpca':
                    steps.append(('transforms', KernelPCA(kernel='rbf')))
                    param_grid[0]["transforms__n_components"] = [3, 4, 5]
                elif transforms_str == 'lle':
                    steps.append(('transforms', LocallyLinearEmbedding()))
                    param_grid[0]["transforms__n_components"] = [3, 4, 5]
                    param_grid[0]["transforms__n_neighbors"] = [3, 5, 7]
                # elif transforms_str == 'mds':
                #     steps.append(('transforms', MDS())) # DOES NOT HAVe transform() function
                #     param_grid[0]["transforms__n_components"] = [3, 4, 5]
                elif transforms_str == 'isomap':
                    steps.append(('transforms', Isomap()))
                    param_grid[0]["transforms__n_components"] = [3, 4, 5]
                    param_grid[0]["transforms__n_neighbors"] = [3, 5, 7]
                # elif transforms_str == 'tsne':  # DOES NOT HAVe transform() function
                #     steps.append(('transforms', TSNE()))
                #     param_grid[0]["transforms__n_components"] = [3, 4, 5]
                ############################################

                ############################################
                if clf_str == 'logistic':
                    steps.append(('clf', LogisticRegression(multi_class='auto', random_state=0, solver='liblinear')))
                    param_grid[0]["clf__penalty"] = ['l1', 'l2']
                    param_grid[0]["clf__C"] = [0.01, 0.1, 1, 10]
                    param_grid[0]["clf__class_weight"] = [None, 'balanced']
                elif clf_str == 'naive_bayes':
                    steps.append(('clf', GaussianNB()))
                elif clf_str == 'knn':
                    steps.append(('clf', KNeighborsClassifier()))
                    param_grid[0]["clf__n_neighbors"] = [3, 5, 10, 20]
                    param_grid[0]["clf__weights"] = ['uniform', 'distance']
                    param_grid[0]["clf__metric"] = ['euclidean', 'manhattan']
                elif clf_str == 'random_forest':
                    steps.append(('clf', RandomForestClassifier()))
                    param_grid[0]["clf__max_depth"] = [3, 7, 10, 20]
                    param_grid[0]["clf__min_samples_split"] = [10, 15, 30]
                    param_grid[0]["clf__n_estimators"] = [50, 100, 150, 200]
                elif clf_str == 'svc':
                    steps.append(('clf', SVC(class_weight='balanced', random_state=42)))
                    mp_tmp = dict(param_grid[0])
                    param_grid[0]["clf__kernel"] = ['linear']
                    param_grid[0]["clf__C"] = [0.01, 0.1, 1]

                    param_grid.append(dict(mp_tmp))
                    param_grid[1]["clf__kernel"] = ['rbf']
                    param_grid[1]["clf__gamma"] = [0.01, 0.1, 1]
                    param_grid[1]["clf__C"] = [0.01, 0.1, 1]
                elif clf_str == 'xgboost':
                    steps.append(('clf',
                                  xgb.XGBClassifier(random_state=42, objective="multi:softmax", num_class=tot_classes)))
                    param_grid[0]["clf__learning_rate"] = [0.001, 0.01, 0.1]
                    param_grid[0]["clf__n_estimators"] = [50, 100, 150, 200]
                elif clf_str == 'adaboost':
                    steps.append(('clf',
                                  AdaBoostClassifier(random_state=42)))
                    param_grid[0]["clf__n_estimators"] = [50, 100, 150, 200]
                elif clf_str == 'gradboost':
                    steps.append(('clf',
                                  GradientBoostingClassifier(random_state=42)))
                    param_grid[0]["clf__learning_rate"] = [0.001, 0.01, 0.1]
                    param_grid[0]["clf__n_estimators"] = [50, 100, 150, 200]
                ############################################

                pipeline = Pipeline(steps=steps)
                clf = GridSearchCV(estimator=pipeline, cv=pds, param_grid=param_grid, verbose=1, scoring='balanced_accuracy')

                res_path = "output/" + configs["experiment"][
                    "name"] + "/" + clf_str + "_" + preproc_str + "_" + transforms_str + ".pkl"

                if os.path.exists(res_path):
                    clf = joblib.load(res_path)
                else:
                    try:
                        clf.fit(X_train, y_train)
                    except:
                        print("Failed for " + res_path)
                        continue
                    joblib.dump(clf, res_path)
                if clf_str not in model_scores:
                    model_scores[clf_str] = []
                val_preds = clf.predict(X_val)
                accuracy = round(accuracy_score(y_val, val_preds), 2)
                bal_accuracy = round(balanced_accuracy_score(y_val, val_preds), 2)
                f1 = round(f1_score(y_val, val_preds, average='weighted'), 2)
                model_scores[clf_str].append([res_path, accuracy, bal_accuracy, f1, clf.best_params_])
                print([model_scores[clf_str][-1][0:4]])
                all_scores.append([res_path, accuracy, bal_accuracy, f1, clf.best_params_])

    all_scores = sorted(all_scores, key=lambda x:x[3], reverse=True)
    for score in all_scores[0:5]:
        print(score)
