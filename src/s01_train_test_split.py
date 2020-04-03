import pandas as pd
import numpy as np
import os


def do_train_test_split():
    path = 'data/Preprocessed_data.csv'
    df = pd.read_csv(path)

    df = df.sort_values(by=['frame.time'])
    df = df.drop_duplicates('frame.time')

    if not os.path.exists("data/train.csv"):
        train_proportion = 0.8

        train_end = int(df.shape[0] * train_proportion)
        train = df[0:train_end]
        test = df[train_end:]

        train.to_csv("data/train.csv", index=False)
        test.to_csv("data/test.csv", index=False)

    # data is known to be one entire day
    tot_seconds_in_a_day = 24 * 60 * 60
    samples_per_sec = int(df.shape[0] / tot_seconds_in_a_day)
    print("Sampling Frequency = " + str(samples_per_sec))
