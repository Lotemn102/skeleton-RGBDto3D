"""
Data reader for the angles. This reader has different angles than the original "data_reader.py", since Omer suggested
more angles.
"""

"""
Read vicon points and their age labels.
Save into dict of <'pcs', 'labels'>, where 'pcs' is a matrix of the points in shape (dataset_size, 39, 3) and labels is
a matrix of ages in shape (dataset_size, 1).
"""
import math
import glob
import numpy as np
import pandas as pd
import open3d as o3d

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

    c7_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'C7'][0]
    t10_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'T10'][0]
    rsho_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'RSHO'][0]
    lsho_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'LSHO'][0]
    clav_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'CLAV'][0]
    rfhd_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'RFHD'][0]
    lfhd_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'LFHD'][0]
    lbhd_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'LBHD'][0]
    rbhd_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'RBHD'][0]
    strn_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'STRN'][0]
    rasi_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'RASI'][0]
    lasi_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'LASI'][0]
    lpsi_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'LPSI'][0]
    rpsi_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'RPSI'][0]

    # Calculate angles for train data
    for sample in x_train_points:
        # ------------------------------- FOR DEBUGGING ----------------------------
        # pcd = o3d.geometry.PointCloud()
        # points = np.array([(point[0], point[1], point[2]) for point in sample])
        # pcd.points = o3d.utility.Vector3dVector(points)
        # visualizer = o3d.visualization.Visualizer()
        # visualizer.create_window(window_name='train')
        # visualizer.add_geometry(pcd)
        # visualizer.run()
        # visualizer.close()
        # --------------------------------------------------------------------------

        C7 = sample[c7_index]
        STRN = sample[strn_index]
        T10 = sample[t10_index]
        RSHO = sample[rsho_index]
        LSHO = sample[lsho_index]
        CLAV = sample[clav_index]
        RFHD = sample[rfhd_index]
        LFHD = sample[lfhd_index]
        LBHD = sample[lbhd_index]
        RBHD = sample[rbhd_index]
        RASI = sample[rasi_index]
        LASI = sample[lasi_index]
        LPSI = sample[lpsi_index]
        RPSI = sample[rpsi_index]

        # (C7, T10, middle of RPSI & LPSI)
        middle = (RPSI + LPSI) / 2
        v1 = T10 - middle
        v2 = T10 - C7
        alpha_1 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (LSHO, STRN, RSHO)
        v1 = STRN - LSHO
        v2 = STRN - RSHO
        alpha_2 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (middle of LBHD & RBHD, middle of RSHO & LSHO, C7)
        middle_1 = (LBHD + RBHD) / 2
        middle_2 = (RSHO + LSHO) / 2
        v1 = middle_2 - middle_1
        v2 = middle_2 - C7
        alpha_3 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (middle of RSHO & LSHO, CLAV, C7)
        middle = (RSHO + LSHO) / 2
        v1 = CLAV - middle
        v2 = CLAV - C7
        alpha_4 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (C7, CLAV, T10)
        v1 = CLAV - C7
        v2 = CLAV - T10
        alpha_5 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (C7, CLAV, middle of RPSI & LPSI)
        middle = (RPSI + LPSI) / 2
        v1 = CLAV - C7
        v2 = CLAV - middle
        alpha_6 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (C7, CLAV, STRN)
        v1 = CLAV - C7
        v2 = CLAV - STRN
        alpha_7 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (middle of RFHD & LFHD, C7, middle of RSHO & LSHO)
        middle_1 = (RFHD + LFHD) / 2
        middle_2 = (RSHO + LSHO) / 2
        v1 = C7 - middle_1
        v2 = C7 - middle_2
        alpha_8 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (CLAV, T10, STRN)
        v1 = T10 - CLAV
        v2 = T10 - STRN
        alpha_9 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (T10, C7, STRN)
        v1 = C7 - T10
        v2 = C7 - STRN
        alpha_10 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (LSHO, T10, RSHO)
        v1 = T10 - LSHO
        v2 = T10 - RSHO
        alpha_11 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        angles = (alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, alpha_7, alpha_8, alpha_9, alpha_10, alpha_11)
        x_train.append(angles)

    # Calculate angles for test data
    for sample in x_test_points:
        # ------------------------------- FOR DEBUGGING ----------------------------
        # pcd = o3d.geometry.PointCloud()
        # points = np.array([(point[0], point[1], point[2]) for point in sample])
        # pcd.points = o3d.utility.Vector3dVector(points)
        # visualizer = o3d.visualization.Visualizer()
        # visualizer.create_window(window_name='test')
        # visualizer.add_geometry(pcd)
        # visualizer.run()
        # visualizer.close()
        # --------------------------------------------------------------------------

        C7 = sample[c7_index]
        STRN = sample[strn_index]
        T10 = sample[t10_index]
        RSHO = sample[rsho_index]
        LSHO = sample[lsho_index]
        CLAV = sample[clav_index]
        RFHD = sample[rfhd_index]
        LFHD = sample[lfhd_index]
        LBHD = sample[lbhd_index]
        RBHD = sample[rbhd_index]
        RASI = sample[rasi_index]
        LASI = sample[lasi_index]
        LPSI = sample[lpsi_index]
        RPSI = sample[rpsi_index]

        # (C7, T10, middle of RPSI & LPSI)
        middle = (RPSI + LPSI) / 2
        v1 = T10 - middle
        v2 = T10 - C7
        alpha_1 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (LSHO, STRN, RSHO)
        v1 = STRN - LSHO
        v2 = STRN - RSHO
        alpha_2 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (middle of LBHD & RBHD, middle of RSHO & LSHO, C7)
        middle_1 = (LBHD + RBHD) / 2
        middle_2 = (RSHO + LSHO) / 2
        v1 = middle_2 - middle_1
        v2 = middle_2 - C7
        alpha_3 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (middle of RSHO & LSHO, CLAV, C7)
        middle = (RSHO + LSHO) / 2
        v1 = CLAV - middle
        v2 = CLAV - C7
        alpha_4 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (C7, CLAV, T10)
        v1 = CLAV - C7
        v2 = CLAV - T10
        alpha_5 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (C7, CLAV, middle of RPSI & LPSI)
        middle = (RPSI + LPSI) / 2
        v1 = CLAV - C7
        v2 = CLAV - middle
        alpha_6 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (C7, CLAV, STRN)
        v1 = CLAV - C7
        v2 = CLAV - STRN
        alpha_7 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (middle of RFHD & LFHD, C7, middle of RSHO & LSHO)
        middle_1 = (RFHD + LFHD) / 2
        middle_2 = (RSHO + LSHO) / 2
        v1 = C7 - middle_1
        v2 = C7 - middle_2
        alpha_8 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (CLAV, T10, STRN)
        v1 = T10 - CLAV
        v2 = T10 - STRN
        alpha_9 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (T10, C7, STRN)
        v1 = C7 - T10
        v2 = C7 - STRN
        alpha_10 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # (LSHO, T10, RSHO)
        v1 = T10 - LSHO
        v2 = T10 - RSHO
        alpha_11 = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        angles = (alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, alpha_7, alpha_8, alpha_9, alpha_10, alpha_11)
        x_test.append(angles)

    # For all other missing nan values, fill them by the average angle.
    clean_x_train = pd.DataFrame(x_train)
    clean_x_test = pd.DataFrame(x_test)
    clean_x_train = clean_x_train.fillna(clean_x_train.mean())
    clean_x_test = clean_x_test.fillna(clean_x_test.mean())

    x_train = clean_x_train.to_numpy()
    x_test = clean_x_test.to_numpy()
    y_train = y_train
    y_test = y_test

    return x_train, x_test, y_train, y_test

def print_metadata():
    x_train, x_test, y_train, y_test = read()
    y_train = np.array([int(e[0]) for e in y_train])
    y_test = np.array([int(e[0]) for e in y_test])
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
