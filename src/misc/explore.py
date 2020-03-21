import pandas as pd
import numpy as np


path = '../../data/Preprocessed_data.csv'
df = pd.read_csv(path)

for col in df.columns:
    print(col)

print(df.iloc[0])
print(df.shape)

print(df.head())

nums = df['frame.number'].values
dff = np.diff(nums)

ids = np.where(dff != 1)

print(len(ids[0]))
for id in ids[0][0:100]:
    print([id, nums[id], nums[id+1]])


df = df.sort_values(by=['frame.time'])
df = df.drop_duplicates('frame.time')
tm = df['frame.time'].values
nums = df['frame.number'].values
dff = np.diff(tm)

ids = np.where(dff < 1)

print(len(ids[0]))
for id in ids[0]:
    print([id, tm[id], tm[id+1], nums[id], nums[id+1]])
