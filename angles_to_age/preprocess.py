from sklearn import preprocessing
import pandas as pd
import numpy as np


def normalize(x_train, x_test):
    scaler = preprocessing.MinMaxScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled

def to_binary(y_train, y_test):
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    np.place(y_train, y_train <= 30, 0)  # 0 for young
    np.place(y_train, y_train >= 60, 1)  # 1 for old
    np.place(y_test, y_test <= 30, 0)  # 0 for young
    np.place(y_test, y_test >= 60, 1)  # 1 for old

    return y_train, y_test
