import tensorflow as tf
from collections import Counter
import numpy as np
import pandas as pd

categories = ["spam", "ham"]
df = pd.read_csv("spam.csv", encoding = "latin1", usecols = ['v1', 'v2'])
df['v1'] = df['v1'].replace('ham', 0).replace('spam', 1)
df['split'] = np.random.rand(df.shape[0], 1)
msk = np.random.rand(len(df)) <= 0.7

df_train = df[msk]
df_train_x = df_train['v2']
df_train_y = df_train['v1']

df_test = df[~msk]
df_test_x = df_test['v2']
df_test_y = df_test['v1']

print('Total messages in train: ', df_train.size)
print('Total messages in test: ', df_test.size)

notSpam = Counter()
actualSpam = Counter()
for int, row in df_train.iterrows():
    if row['v1'] == 0:
        for word in row['v2'].split(' '):
            notSpam[word.lower()]+=1
    else:
        for word in row['v2'].split(' '):
            actualSpam[word.lower()]+=1

print("Total words for spam BoW:", len(actualSpam))
print("Total words for ham BoW:", len(notSpam))