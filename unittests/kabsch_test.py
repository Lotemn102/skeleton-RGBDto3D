"""
This test shows that the Kabsch algorithm is not working well for our problem...
"""

import unittest
import cv2
import numpy as np
import math
from scipy import spatial

from data_cleaning.kabsch import kabsch
from data_cleaning.vicon_data_reader import VICONReader, KEYPOINTS_NAMES
from data_cleaning.cvat_data_reader import CVATReader
from data_cleaning.visualize_functions import visualize_vicon_points

VICON_POINTS_CSV_PATH = 'assets/vicon_points.csv'
PIXEL_POINTS_CSV_PATH = 'assets/annotations_2d.json'
IMAGE_PATH = 'assets/rgb_frame.png'

def read_data(show_data=False):
    # Load 2d points.
    cvat_reader = CVATReader(PIXEL_POINTS_CSV_PATH)
    data_2d_dict = cvat_reader.get_points()
    all_points_names_2d = data_2d_dict.keys()

    # Init matrix with the data. Order of points is according to KEYPOINTS_NAMES.
    A = np.zeros((len(data_2d_dict), 3))

    for i, key in enumerate(list(data_2d_dict.keys())):
        keypoint = data_2d_dict[key]
        A[i][0] = np.round(keypoint[0])
        A[i][1] = np.round(keypoint[1])
        A[i][2] = 0

    A = np.asmatrix(A)

    # Visualize original points.
    image = cv2.imread(IMAGE_PATH)

    for p in A:
        p = p.T
        image = cv2.circle(image, (int(p[0]), int(p[1])), radius=1, color=(0, 255, 0), thickness=5)

    if show_data:
        cv2.imshow("Original", image)
        cv2.waitKey(0)

    # Load 3d points.
    reader_3d = VICONReader(VICON_POINTS_CSV_PATH)
    points = reader_3d.get_points()
    points = list(points.items())[0][1]

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

    # Visualize vicon points.
    if show_data:
        visualize_vicon_points(data_3d_dict)

    B = np.zeros((len(data_3d_dict), 3))

    for i, key in enumerate(data_3d_dict):
        keypoint = data_3d_dict[key]
        B[i][0] = keypoint[0]
        B[i][1] = keypoint[1]
        B[i][2] = keypoint[2]

    B = np.asmatrix(B)

    return A, B


class TestKabsch(unittest.TestCase):
    def test_1(self):
        two_d_data, three_d_data = read_data(show_data=False)

        # Find transformation.
        s, ret_R, ret_t = kabsch(A=two_d_data, B=three_d_data, scale=True)

        # Find the error.
        N = two_d_data.shape[0]
        target_matrix = np.dot(ret_R, np.dot(three_d_data.T, s)) + np.tile(ret_t, (1, N))
        target_matrix = target_matrix.T
        target_matrix = target_matrix[:, 0:3] # Remove last column of ones.
        err = two_d_data - target_matrix
        err = np.multiply(err, err)
        err = np.sum(err)
        rmse = math.sqrt(err / N)
        print("rmse: " + str(rmse))

        # Visualize projected points.
        image = cv2.imread(IMAGE_PATH)

        for p in target_matrix:
            p = p.T
            x = int(int(p[0]))
            y = int(int(p[1]))
            image = cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=5)

        cv2.imshow("Projected", image)
        cv2.waitKey(0)

    def test_2(self):
        """
        Trying to use scipy's kabsch implementation. Not sure why but the rotation matrix it returns is wrong...
        """
        A, B = read_data()

        rotation, rmsd = spatial.transform.Rotation.align_vectors(a=B, b=A)
        r = rotation.as_matrix()

        # Convert to 4x4 transform matrix
        K = np.zeros((4, 4))
        K[:3, :3] = r
        # Assuming translation is 0
        K[3, 3] = 1

        # Find the error.
        N = A.shape[0]
        src_matrix = np.ones((N, 4))  # Add last column of ones.
        src_matrix[:, 0:3] = B
        target_matrix = np.zeros((N, 3))

        for i, row in enumerate(src_matrix):
            uvw = np.dot(K, row)
            u = uvw[0] / uvw[2]  # Scale by Z value
            v = uvw[1] / uvw[2]  # Scale by Z value
            target_matrix[i][0] = np.round(u)
            target_matrix[i][1] = np.round(v)

        target_matrix = target_matrix[:, 0:3]  # Remove last column of ones.
        err = A - target_matrix
        err = np.multiply(err, err)
        err = np.sum(err)
        rmse = math.sqrt(err / N)
        print("rmse: " + str(rmse))

        # Visualize projected points.
        image = cv2.imread(IMAGE_PATH)

        for p in target_matrix:
            p = p.T
            x = int(int(p[0]) / 5) + 200
            y = int(int(p[1]) / 5) + 200
            image = cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=5)

        cv2.imshow("Projected", image)
        cv2.waitKey(0)






