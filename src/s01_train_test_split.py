import pandas as pd
import numpy as np

path = '../data/Preprocessed_data.csv'
df = pd.read_csv(path)

df = df.sort_values(by=['frame.time'])
df = df.drop_duplicates('frame.time')

train_proportion = 0.8

train_end = int(df.shape[0] * train_proportion)
train = df[0:train_end]
test = df[train_end:]

train.to_csv("../data/train.csv", index=False)
test.to_csv("../data/test.csv", index=False)
