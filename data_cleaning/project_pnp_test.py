import cv2
import numpy as np

from data_cleaning.cvat_data_reader import CVATReader
from data_cleaning.vicon_data_reader import VICONReader, KEYPOINTS_NAMES

ANGLE = 'Front'
SUB = 'Sub007'
PIXEL_POINTS_JSON_PATH = '../../annotations_data/{sub}/{angle}/annotations.json'.format(angle=ANGLE, sub=SUB)
VICON_POINTS_CSV_PATH = '../../annotations_data/{sub}/{angle}/vicon_points.csv'.format(angle=ANGLE, sub=SUB)

def read_data(annotations_path):
    # Load 2d points.
    cvat_reader = CVATReader(annotations_path)
    data_2d_dict = cvat_reader.get_points()
    all_points_names_2d = data_2d_dict.keys()

    # Init matrix with the data. Order of points is according to KEYPOINTS_NAMES.
    A = []

    for i, key in enumerate(list(data_2d_dict.keys())):
        keypoint = data_2d_dict[key]
        x = np.round(keypoint[0])
        y = np.round(keypoint[1])
        A.append((int(x), int(y)))

    A = np.array(A)

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

    B = []

    for i, key in enumerate(data_3d_dict):
        keypoint = data_3d_dict[key]
        B.append((keypoint[0], keypoint[1], keypoint[2]))

    B = np.array(B)

    return A, B

A, B = read_data(PIXEL_POINTS_JSON_PATH)
camera_mat = np.zeros((3, 3))
camera_mat[0][0] = 480
camera_mat[1][1] = 640
camera_mat[2][2] = 1
camera_mat[2][0] = 480/2
camera_mat[2][1] = 640/2

success, R, t = cv2.solvePnP(B, A, camera_mat, np.zeros((1, 3)))

x = 3