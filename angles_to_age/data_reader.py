"""
Read vicon points and their age labels.
Save into dict of <'pcs', 'labels'>, where 'pcs' is a matrix of the points in shape (dataset_size, 39, 3) and labels is
a matrix of ages in shape (dataset_size, 1).
"""
import math
import glob
import numpy as np
import pandas as pd

from rgbd_to_3d.data_cleaning.vicon_data_reader import KEYPOINTS_NAMES

def is_there_nan_coordinate(item):
    for coordinate in item:
        if math.isnan(coordinate):
            return True

    return False


def read():
    DATA_PATH = '../../data_angles_to_age/splitted/'

    for file in glob.glob(DATA_PATH + '*.npy'):
        if 'x_train' in file:
            x_train_points = np.load(file, allow_pickle=True)

        if 'x_test' in file:
            x_test_points = np.load(file, allow_pickle=True)

        if 'y_train' in file:
            y_train = np.load(file, allow_pickle=True)

        if 'y_test' in file:
            y_test = np.load(file, allow_pickle=True)

    # Calculate angles
    x_train = []
    x_test = []

    # (C7, STERN, T10)
    c7_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'C7'][0]
    t10_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'T10'][0]
    rsho_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'RSHO'][0]
    lsho_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'LSHO'][0]
    clav_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'CLAV'][0]
    rfhd_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'RFHD'][0]
    lfhd_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'LFHD'][0]
    strn_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'STRN'][0]
    rasi_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'RASI'][0]
    lasi_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'LASI'][0]

    # Calculate angles for train data
    for sample in x_train_points:
        C7 = sample[c7_index]
        STRN = sample[strn_index]
        T10 = sample[t10_index]
        RSHO = sample[rsho_index]
        LSHO = sample[lsho_index]
        CLAV = sample[clav_index]
        RFHD = sample[rfhd_index]
        LFHD = sample[lfhd_index]
        RASI = sample[rasi_index]
        LASI = sample[lasi_index]

        # (C7, STRN, T10)
        v1 = STRN - C7
        v2 = STRN - T10
        alpha_1 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (RSHO, C7, LSHO)
        v1 = C7 - RSHO
        v2 = C7 - LSHO
        alpha_2 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (CLAV, C7, middle of RFHD and LFHD)
        middle = (RFHD + LFHD) / 2
        v1 = C7 - middle
        v2 = C7 - CLAV
        alpha_3 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (STRN, C7, middle of RASI and LASI)
        middle = (RASI + LASI) / 2
        v1 = C7 - middle
        v2 = C7 - STRN
        alpha_4 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        angles = (alpha_1, alpha_2, alpha_3, alpha_4)
        x_train.append(angles)

    # Calculate angles for test data
    for sample in x_test_points:
        C7 = sample[c7_index]
        STRN = sample[strn_index]
        T10 = sample[t10_index]
        RSHO = sample[rsho_index]
        LSHO = sample[lsho_index]
        CLAV = sample[clav_index]
        RFHD = sample[rfhd_index]
        LFHD = sample[lfhd_index]
        RASI = sample[rasi_index]
        LASI = sample[lasi_index]

        # (C7, STRN, T10)
        v1 = STRN - C7
        v2 = STRN - T10
        alpha_1 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (RSHO, C7, LSHO)
        v1 = C7 - RSHO
        v2 = C7 - LSHO
        alpha_2 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (CLAV, C7, middle of RFHD and LFHD)
        middle = (RFHD + LFHD) / 2
        v1 = C7 - middle
        v2 = C7 - CLAV
        alpha_3 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (STRN, C7, middle of RASI and LASI)
        middle = (RASI + LASI) / 2
        v1 = C7 - middle
        v2 = C7 - STRN
        alpha_4 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        angles = (alpha_1, alpha_2, alpha_3, alpha_4)
        x_test.append(angles)

    # Remove nans: If more the one angle is missing, remove the sample from both x and y
    clean_x_train = []
    clean_y_train = []
    clean_x_test = []
    clean_y_test = []

    for i, sample in enumerate(x_train):
        if sum(math.isnan(x) for x in sample) <= 1: # There is at most 1 angle that is missing
            clean_x_train.append(sample)
            clean_y_train.append(y_train[i])

    for i, sample in enumerate(x_test):
        if sum(math.isnan(x) for x in sample) <= 1: # There is at most 1 angle that is missing
            clean_x_test.append(sample)
            clean_y_test.append(y_test[i])

    # For all other missing nan values, fill them by the average angle.
    clean_x_train = pd.DataFrame(clean_x_train)
    clean_x_test = pd.DataFrame(clean_x_test)
    clean_x_train = clean_x_train.fillna(clean_x_train.mean())
    clean_x_test = clean_x_test.fillna(clean_x_test.mean())

    x_train = clean_x_train.to_numpy()
    x_test = clean_x_test.to_numpy()
    y_train = clean_y_train
    y_test = clean_y_test

    return x_train, x_test, y_train, y_test

def print_metadata():
    x_train, x_test, y_train, y_test = read()
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    total = len(y_train)+len(y_test)
    total_train = len(y_train)
    total_test = len(y_test)
    print("Total objects: {n}".format(n=total))
    print("Train objects: {n}".format(n=total_train))
    print("Test objects: {n}".format(n=total_test))
    print("Percentage of objects of age 'old' in train: {n}%".format(n=round(100*(len(np.where(y_train >= 30)[0])/ total_train), 3)))
    print("Percentage of objects of age 'old' in test: {n}%".format(n=round(100*(len(np.where(y_test >= 30)[0]) / total_test), 3)))
    print("Percentage of objects of age 'young' in train: {n}%".format(n=round(100*(len(np.where(y_train < 30)[0])/ total_train), 3)))
    print("Percentage of objects of age 'young' in test: {n}%".format(n=round(100*(len(np.where(y_test < 30)[0])/ total_test), 3)))
    print("All ages: {n}".format(n=sorted(set(np.unique(np.concatenate((y_train, y_test)))))))


if __name__ == "__main__":
    print_metadata()
