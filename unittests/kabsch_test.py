import unittest
import cv2
import numpy as np
import pandas as pd
from math import sqrt

from data_cleaning.kabsch import rigid_transform_3D
from data_cleaning.vicon_data_reader import VICONReader, KEYPOINTS_NAMES

VICON_POINTS_CSV_PATH = 'assets/vicon_points.csv'
PIXEL_POINTS_CSV_PATH = 'assets/2d_points.csv'
IMAGE_PATH = 'assets/rgb_frame.png'

class TestKabsch(unittest.TestCase):
    # Load 2d points.
    data_2d = pd.read_csv(PIXEL_POINTS_CSV_PATH, header=None, usecols=[0, 1, 2])
    data_2d = data_2d.T
    data_2d_dict = {}

    for i in range(len(data_2d.T)):
        keypoint_name = data_2d[i][0]
        x = data_2d[i][1]
        y = data_2d[i][2]
        data_2d_dict[keypoint_name] = [x, y, 0]

    # Sort according to the KEYPOINTS_NAMES.
    data_2d_dict = sorted(data_2d_dict.items(), key=lambda pair: KEYPOINTS_NAMES.index(pair[0]))
    all_points_names_2d = [c[0] for c in data_2d_dict]

    # Init matrix with the data. Order of points is according to KEYPOINTS_NAMES.
    A = np.zeros((len(data_2d_dict), 3))

    for i, key in enumerate(data_2d_dict):
        keypoint = key[1]
        A[i][0] = keypoint[0]
        A[i][1] = keypoint[1]
        A[i][2] = keypoint[2]

    # Load 3d points.
    reader_3d = VICONReader(VICON_POINTS_CSV_PATH)
    data_3d = reader_3d.get_points()
    points = list(data_3d.items())[0][1]
    data_3d_dict = {}

    for i, p in enumerate(points):
        x = p.x
        y = p.y
        z = p.z
        data_3d_dict[KEYPOINTS_NAMES[i]] = [x, y, z]

    # I was not able to pin-pick all points in the images. I did not pinned-pick the points that can't be seen in the
    # image. That's lead to the that the number of 2d points is smaller than the vicon points.
    for key in list(data_3d_dict.items()):
        if key[0] not in all_points_names_2d:
            data_3d_dict.pop(key[0], None)

    B = np.zeros((len(data_3d_dict), 3))

    for i, key in enumerate(data_3d_dict):
        keypoint = data_3d_dict[key]
        B[i][0] = keypoint[0]
        B[i][1] = keypoint[1]
        B[i][2] = keypoint[2]

    # Find transformation.
    s, ret_R, ret_t = rigid_transform_3D(A=A, B=B, scale=False)

    # Find the error.
    n = B.shape[0]
    a = (ret_R.dot(B.T))
    b = np.tile(ret_t, (1, n))
    B2 = a + b
    B2 = B2.T
    err = A - B2
    err = np.multiply(err, err)
    err = np.sum(err)
    rmse = sqrt(err / n)

    # Visualize projected points.
    c = 6

    pass