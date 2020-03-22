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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost

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

    train_proportion = 0.8
    train_end = int(X.shape[0] * train_proportion)
    split_index = [-1 if i < train_end else 0 for i in range(X.shape[0])]
    pds = PredefinedSplit(test_fold=split_index)
    X_val = X[train_end:, :]
    y_val = y[train_end:]

    try:
        os.mkdir("output/" + configs["experiment"]["name"] + "/")
    except:
        pass

    model_scores = {}

    for clf_str in configs["models"]["classifiers"]:
        for preproc_str in configs["models"]["preprocs"]:
            for transforms_str in configs["models"]["transforms"]:
                steps = [('imputer', SimpleImputer(strategy='mean'))]
                param_grid = [{}]
                if preproc_str == 'min_max':
                    steps.append(('preprocs', MinMaxScaler()))
                elif preproc_str == 'standard_scalar':
                    steps.append(('preprocs', StandardScaler()))
                if transforms_str == 'pca':
                    steps.append(('transforms', PCA()))
                    param_grid[0]["transforms__n_components"] = [0.1, 0.25, 0.5, 0.75, 0.9]

                if clf_str == 'logistic':
                    steps.append(('clf', LogisticRegression(multi_class='auto', random_state=0, solver='liblinear')))
                    param_grid[0]["clf__penalty"] = ['l1', 'l2']
                    param_grid[0]["clf__C"] = [0.01, 0.1, 0.2, 0.5, 0.75, 1, 2, 10]
                    param_grid[0]["clf__class_weight"] = [None, 'balanced']

                pipeline = Pipeline(steps=steps)
                clf = GridSearchCV(estimator=pipeline, cv=pds, param_grid=param_grid, verbose=1, scoring='balanced_accuracy')

                res_path = "output/" + configs["experiment"][
                    "name"] + "/" + clf_str + "_" + preproc_str + "_" + transforms_str + ".pkl"

                if os.path.exists(res_path):
                    clf = joblib.load(res_path)
                else:
                    clf.fit(X, y)
                    joblib.dump(clf, res_path)
                if clf_str not in model_scores:
                    model_scores[clf_str] = []
                val_preds = clf.predict(X_val)
                accuracy = accuracy_score(y_val, val_preds)
                bal_accuracy = balanced_accuracy_score(y_val, val_preds)
                f1 = f1_score(y_val, val_preds, average='weighted')
                model_scores[clf_str].append([res_path, clf.best_params_, clf.best_score_,
                                              accuracy, bal_accuracy, f1])
            print(model_scores)
            return


