"""
Read vicon points and their age labels.
Save into dict of <'pcs', 'labels'>, where 'pcs' is a matrix of the points in shape (dataset_size, 39, 3) and labels is
a matrix of ages in shape (dataset_size, 1).
"""
import random
import glob
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import h5py

from rgbd_to_3d.data_cleaning.vicon_data_reader import VICONReader

def read():
    DATA_PATH = '../../data_3d_to_age/splitted/'

    for file in glob.glob(DATA_PATH + '*.npy'):
        if 'x_train' in file:
            x_train = np.load(file, allow_pickle=True)

        if 'x_test' in file:
            x_test = np.load(file, allow_pickle=True)

        if 'y_train' in file:
            y_train = np.load(file, allow_pickle=True)

        if 'y_test' in file:
            y_test = np.load(file, allow_pickle=True)

    # TODO: Just for now! Think of a better way
    x_train = np.nan_to_num(x_train, copy=True, nan=0.0)
    x_test = np.nan_to_num(x_test, copy=True, nan=0.0)

    return x_train, x_test, y_train, y_test

def print_metadata():
    x_train, x_test, y_train, y_test = read()
    total = len(y_train)+len(y_test)
    print("Total objects: {n}".format(n=total))
    print("Train objects: {n}".format(n=len(y_train)))
    print("Test objects: {n}".format(n=len(y_test)))
    print("Percentage of objects with ages <= 40: {n}".format(n=(len(np.where(y_train <= 40)[0]) + len(np.where(y_test <= 40)[0]))/total))
    print("Percentage of objects with ages > 40: {n}".format(n=(len(np.where(y_train > 40)[0]) + len(np.where(y_test > 40)[0]))/total))
    print("All ages: {n}".format(n=sorted(set(np.unique(np.concatenate((y_train, y_test)))))))


if __name__ == "__main__":
    print_metadata()