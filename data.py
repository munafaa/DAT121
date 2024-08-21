import numpy as np
import pandas as pd


"""
.py file that just import the dataset and pre-processes it
"""

def data_import():
    data = pd.read_csv("heart.csv")
    t = data.iloc[:,-1] #last column of data, targets

    X_pre = data.drop(data.columns[-1], axis=1) #remove target column from the data

    #Center the data
    mean = np.mean(X_pre, axis=0)
    X_c = X_pre - mean

    #normalize the data
    min_X = X_c.min()
    max_X = X_c.max()
    X = (X_c - min_X) / (max_X - min_X)

    X_numpy = X.to_numpy()
    t_numpy = t.to_numpy()

    return X, t, X_numpy, t_numpy


