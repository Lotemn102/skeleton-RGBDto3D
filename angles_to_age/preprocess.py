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

    train_labels = np.array([int(e[0]) for e in y_train])
    train_filenames = np.array([e[1] for e in y_train])

    binary_y_train = []
    binary_y_test = []

    for sample in y_train:
        age = sample[0]
        filename = sample[1]

        if int(age) <= 30:
            binary_y_train.append((0, filename)) # 0 for young
        else:
            binary_y_train.append((1, filename)) # 1 for old

    for sample in y_test:
        age = sample[0]
        filename = sample[1]

        if int(age) <= 30:
            binary_y_test.append((0, filename))  # 0 for young
        else:
            binary_y_test.append((1, filename))  # 1 for old

    y_train = np.array(binary_y_train)
    y_test = np.array(binary_y_test)

    return y_train, y_test

def shuffle(x_train, x_test, y_train, y_test):
    p_train = np.random.permutation(len(x_train))
    p_test = np.random.permutation(len(x_test))

    x_train = x_train[p_train]
    y_train = y_train[p_train]

    x_test = x_test[p_test]
    y_test = y_test[p_test]

    return x_train, x_test, y_train, y_test
