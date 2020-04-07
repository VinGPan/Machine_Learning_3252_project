from src.s03_compute_features import compute_features
from src.utils import read_yml
import pandas as pd
import numpy as np
import os


def make_data(df, config):
    '''

    Depending on the experiment, pre-process the data. There are following 3 choices:
    1) Read the data and just return the dataframe
    2) Read Data and re-arrange the data to provide time-series view to the data
    3) Read Data and re-arrange the data to provide time-series view to the data. Additionally, compute PyEEG features

    :param df: Input Dataframe
    :param config: a dict providing the details of the pre-processing
    :return: X, y
    '''

    # Read the experiment requirement
    history = config["data"]["history"]
    X = (df[config["data"]["cols"]]).values
    y = (df[config["data"]["target"]]).values
    lead = config["data"]["lead"]
    if history == 0:
        # just return the input dataframe
        pass
    else:
        # re-arrange the data to provide time-series view
        r, c = X.shape
        # determine the number rows in resultant output using history
        new_r = int(r/history)
        r_end = new_r * history
        # last few rows will be ignored if total rows are exact multiple of history
        X = X[0:r_end, :]
        y = y[0:r_end]
        X = X.flatten()
        X = X.reshape((new_r, c * history))
        y = y.reshape((new_r, 1 * history))
        y = y[:, -1]
        # Use lead to determine how to shift the labels
        X = X[0:-lead, :]
        y = y[lead:]
    if "feature_eng" in config:
        feature_names = config["feature_eng"]["features"]
        # Compute PyEEG features
        X = compute_features(X, feature_names, history, config)
    return X, y


def prepare_data(yml_name, training):
    '''

    Depending on the experiment, pre-process the data. There are following 3 choices:
    1) Read the data and just return the dataframe
    2) Read Data and re-arrange the data to provide time-series view to the data
    3) Read Data and re-arrange the data to provide time-series view to the data. Additionally, compute PyEEG features

    stores the result into two .npy files, one for X and one for y.

    NOTE: If you have already run this experiment, this function will simply load the old results and returns the same.

    :param yml_name: yml file containing the description of the experiment
    :param training: Indicate weather training or test step. This will used to read and write appropriate files
    :return: X, y
    '''

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
