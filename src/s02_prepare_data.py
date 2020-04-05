from src.s03_compute_features import compute_features
from src.utils import read_yml
import pandas as pd
import numpy as np
import os


def make_data(df, config):
    history = config["data"]["history"]
    X = (df[config["data"]["cols"]]).values
    y = (df[config["data"]["target"]]).values
    lead = config["data"]["lead"]
    if history == 0:
        pass
    else:
        r, c = X.shape
        new_r = int(r/history)
        r_end = new_r * history
        X = X[0:r_end, :]
        y = y[0:r_end]
        X = X.flatten()
        X = X.reshape((new_r, c * history))
        y = y.reshape((new_r, 1 * history))
        y = y[:, -1]
        X = X[0:-lead, :]
        y = y[lead:]
    if "feature_eng" in config:
        feature_names = config["feature_eng"]["features"]
        X = compute_features(X, feature_names, history, config)
    return X, y


def prepare_data(yml_name, training):
    configs = read_yml(yml_name)
    if training:
        if os.path.exists("data/" + configs["experiment"]["name"] + "_X.npy"):
            X = np.load("data/" + configs["experiment"]["name"] + "_X.npy")
            y = np.load("data/" + configs["experiment"]["name"] + "_y.npy")
            print("Train Set Dimensionality " + str(X.shape))
        else:
            df = pd.read_csv('data/train.csv')
            X, y = make_data(df, configs)
            np.save("data/" + configs["experiment"]["name"] + "_X.npy", X)
            np.save("data/" + configs["experiment"]["name"] + "_y.npy", y)
    else:
        df = pd.read_csv('data/test.csv')
        X, y = make_data(df, configs)
        np.save("data/test_X.npy", X)
        np.save("data/test_y.npy", y)
    return X, y
