"""
We have decided to first try to train OpenPose with our data, but use 2d points instead of 3d points as labels.
This file contains scripts for projecting the 3d vicon points in the realsense rgb image pixels.
"""
import math
import numpy as np
import json
import cv2
from shutil import copyfile

from data_cleaning.kabsch import kabsch
from data_cleaning.cvat_data_reader import CVATReader
from data_cleaning.vicon_data_reader import VICONReader, KEYPOINTS_NAMES

ANGLE = 'Front'
PATH_DATA = '../../data/Sub007/Sub007/'
PATH_ANNOTATIONS = '../../annotations_data/Sub007/{angle}/'.format(angle=ANGLE)
PIXEL_POINTS_JSON_PATH = '../../annotations_data/Sub007/{angle}/annotations.json'.format(angle=ANGLE)
VICON_POINTS_CSV_PATH = '../../annotations_data/Sub007/{angle}/vicon_points.csv'.format(angle=ANGLE)
RGB_IMAGE_PATH = '../../annotations_data/Sub007/{angle}/rgb_frame.png'.format(angle=ANGLE)


def find_rotation_matrix(annotated_2d_points, vicon_3d_points):
    s, ret_R, ret_t = kabsch(A=annotated_2d_points, B=vicon_3d_points, scale=True)
    return s, ret_R, ret_t

def calc_rmse(annotated_2d_points, vicon_3d_points, scale, rotation_matrix, translation_vector):
    N = annotated_2d_points.shape[0]
    target_matrix = np.dot(rotation_matrix, np.dot(vicon_3d_points.T, scale)) + np.tile(translation_vector, (1, N))
    target_matrix = target_matrix.T
    target_matrix = target_matrix[:, 0:3]  # Remove last column of ones.
    err = annotated_2d_points - target_matrix
    err = np.multiply(err, err)
    err = np.sum(err)
    rmse = math.sqrt(err / N)
    return rmse

def project(vicon_3d_points, scale, rotation_matrix, translation_vector):
    """
    Project vicon 3d points to pixel 2d points.

    :param vicon_3d_points: 3D vicon points.
    :param scale: Float.
    :param rotation_matrix: Numpy array.
    :param translation_vector: Numpy array.
    :return: None.
    """
    # Convert the points into matrix
    B = np.zeros((len(vicon_3d_points), 3))

    for i, keypoint in enumerate(vicon_3d_points):
        B[i][0] = keypoint.x
        B[i][1] = keypoint.y
        B[i][2] = keypoint.z

    B = np.asmatrix(B)

    N = len(vicon_3d_points)
    target_matrix = np.dot(rotation_matrix, np.dot(B.T, scale)) + np.tile(translation_vector, (1, N))
    target_matrix = target_matrix.T
    target_matrix = target_matrix[:, 0:3]  # Remove last column of ones.
    return target_matrix

def read_data():
    # Load 2d points.
    cvat_reader = CVATReader(PIXEL_POINTS_JSON_PATH)
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

    B = np.zeros((len(data_3d_dict), 3))

    for i, key in enumerate(data_3d_dict):
        keypoint = data_3d_dict[key]
        B[i][0] = keypoint[0]
        B[i][1] = keypoint[1]
        B[i][2] = keypoint[2]

    B = np.asmatrix(B)

    return A, B

def save_calibration_to_json():
    A, B = read_data()
    s, R, t = find_rotation_matrix(annotated_2d_points=A, vicon_3d_points=B)

    R = R.tolist()
    t = t.tolist()

    data = {
        'Scale': s,
        'Rotation': R,
        'Translation': t
    }

    file_name = 'calibration_sub007_{angle}.json'.format(angle=ANGLE.lower())
    json_data = json.dumps(data)
    f = open(PATH_ANNOTATIONS + file_name, 'w')
    f.write(json_data)
    f.close()
    copyfile(src=PATH_ANNOTATIONS + file_name, dst=PATH_DATA + file_name)


if __name__ == "__main__":
    save_calibration_to_json()
