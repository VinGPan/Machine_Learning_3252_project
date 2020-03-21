from src.utils import read_yml
import pandas as pd
import numpy as np


def make_data(inp_path, config):
    df = pd.read_csv(inp_path)
    history = config["data"]["history"]
    X = (df[config["data"]["cols"]]).values
    y = (df[config["data"]["target"]]).values
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

    return X, y


def prepare_data(yml_name, training):
    configs = read_yml(yml_name)
    if training:
        X, y = make_data('data/train.csv', configs)
    else:
        X, y = make_data('data/test.csv', configs)
    return X, y




