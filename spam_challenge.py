import tensorflow as tf
from collections import Counter
import numpy as np
import pandas as pd

categories = ["spam", "ham"]
df = pd.read_csv("spam.csv", encoding = "latin1", usecols = ['v1', 'v2'])
df['split'] = np.random.rand(df.shape[0], 1)
msk = np.random.rand(len(df)) <= 0.7

df_train = df[msk]
df_test = df[~msk]

print('Total messages in train: ', df_train.size)
print('Total messages in test: ', df_test.size)