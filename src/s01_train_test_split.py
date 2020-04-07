import pandas as pd
import numpy as np
import os


def do_train_test_split():
    '''
    Performs Train test split.

    Stores result to data/train.csv and data/test.csv
    Before calling this we need to download the data.
    Download Preprocessed_data.csv from https://www.kaggle.com/speedwall10/iot-device-network-logs
    and copy it under data folder

    '''

    # Read Data
    path = 'data/Preprocessed_data.csv'
    df = pd.read_csv(path)

    # Sort Data as per time stamp
    df = df.sort_values(by=['frame.time'])
    df = df.drop_duplicates('frame.time')

    print("Data Set Dimensionality " + str(df.shape))

    # Use 'first' 80% of data for training and the rest as test
    train_proportion = 0.8
    train_end = int(df.shape[0] * train_proportion)
    train = df[0:train_end]
    test = df[train_end:]

    # Save the train and test files separately
    if not os.path.exists("data/train.csv"):
        train.to_csv("data/train.csv", index=False)
        test.to_csv("data/test.csv", index=False)

    # Find the sampling rate of the data
    tot_seconds_in_a_day = 24 * 60 * 60
    samples_per_sec = int(df.shape[0] / tot_seconds_in_a_day)
    print("Sampling Frequency = " + str(samples_per_sec))

    print("Train Set Dimensionality " + str(train.shape))
    print("Test Set Dimensionality " + str(test.shape))
