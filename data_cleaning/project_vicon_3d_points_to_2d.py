"""
We have decided to first try to train OpenPose with our data, but use 2d points instead of 3d points as labels.
This file contains scripts for projecting the 3d vicon points in the realsense rgb image pixels.
"""
import math
import numpy as np
import json
import cv2
from shutil import copyfile
import open3d as o3d

from data_cleaning.kabsch import kabsch
from data_cleaning.cvat_data_reader import CVATReader
from data_cleaning.vicon_data_reader import VICONReader, KEYPOINTS_NAMES
from data_cleaning.visualize_functions import visualize_2d_points_on_frame
from data_cleaning.visualize_functions import visualize_vicon_points

ANGLE = 'Front'
SUB = 'Sub007'
PATH_DATA = '../../data/{sub}/{sub}/'.format(sub=SUB)
PATH_ANNOTATIONS = '../../annotations_data/{sub}/{angle}/'.format(angle=ANGLE, sub=SUB)
PIXEL_POINTS_JSON_PATH = '../../annotations_data/{sub}/{angle}/annotations.json'.format(angle=ANGLE, sub=SUB)
PIXEL_POINTS_SUB_SET_JSON_PATH = '../../annotations_data/{sub}/{angle}/annotations_subset.json'.format(angle=ANGLE, sub=SUB)
VICON_POINTS_CSV_PATH = '../../annotations_data/{sub}/{angle}/vicon_points.csv'.format(angle=ANGLE, sub=SUB)
RGB_IMAGE_PATH = '../../annotations_data/{sub}/{angle}/rgb_frame.png'.format(angle=ANGLE, sub=SUB)
DEPTH_IMAGE_PATH = '../../annotations_data/{sub}/{angle}/depth_frame.raw'.format(angle=ANGLE, sub=SUB)

def find_rotation_matrix_iterative(annotated_2d_points, vicon_3d_points):
    original_annotated_2d_points = np.copy(annotated_2d_points)
    original_3d_points = np.copy(vicon_3d_points)
    previous_mean_rmse = np.inf
    mean_rmse = np.inf
    counter = 0
    limit_iterations = 2
    lambda_value = 0.15

    while mean_rmse <= previous_mean_rmse+lambda_value and counter < limit_iterations:
        s, ret_R, ret_t = kabsch(A=annotated_2d_points, B=vicon_3d_points, scale=True)

        previous_mean_rmse = mean_rmse
        mean_rmse, err = calc_rmse(annotated_2d_points=original_annotated_2d_points, vicon_3d_points=original_3d_points,
                                   scale=s, rotation_matrix=ret_R, translation_vector=ret_t)

        error_index_mapping = {}

        # For each point, calculate it's rmse.
        for i, point in enumerate(annotated_2d_points):
            error = err[i]
            error = np.multiply(error, error)
            error = np.sum(error)
            error = np.sqrt(error)
            error_index_mapping[i] = error

        # Remove the point with the largest error.
        sorted_points = {k: v for k, v in sorted(error_index_mapping.items(), key=lambda item: item[1])}
        largest_error_index = list(sorted_points.keys())[-1]
        annotated_2d_points = np.delete(annotated_2d_points, largest_error_index, 0)
        vicon_3d_points = np.delete(vicon_3d_points, largest_error_index, 0)

        # Visualize remaining points.
        # image = cv2.imread(RGB_IMAGE_PATH)
        #
        # for i, row in enumerate(annotated_2d_points):
        #     x = row.T[0]
        #     y = row.T[1]
        #
        #     if math.isnan(x) or math.isnan(y):
        #         continue
        #
        #     x = int(x)
        #     y = int(y)
        #
        #     image = cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=5)
        #
        # cv2.imshow("fg", image)
        # cv2.waitKey(0)


        counter = counter + 1

    return s, ret_R, ret_t

def find_rotation_matrix(annotated_2d_points, vicon_3d_points):
    s, ret_R, ret_t = kabsch(A=annotated_2d_points, B=vicon_3d_points, scale=True)
    return s, ret_R, ret_t

def calc_rmse(annotated_2d_points, vicon_3d_points, scale, rotation_matrix, translation_vector):
    N = annotated_2d_points.shape[0]
    target_matrix = np.dot(rotation_matrix, np.dot(vicon_3d_points.T, scale)) + np.tile(translation_vector, (1, N))
    target_matrix = target_matrix.T
    target_matrix = target_matrix[:, 0:3]  # Remove last column of ones.
    diff = annotated_2d_points - target_matrix
    err = annotated_2d_points - target_matrix
    err = np.multiply(err, err)
    err = np.sum(err)
    rmse = math.sqrt(err / N)
    return rmse, diff

def transform(vicon_3d_points, scale, rotation_matrix, translation_vector):
    """
    Project vicon 3d points to realsense 3d points.

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

def project(vicon_3d_points, scale, rotation_matrix, translation_vector):
    # Vicon 3d -> realsense 3d
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

    # Realsense 3d -> pixels
    projected = np.zeros((target_matrix.shape[0], 2))

    for i, row in enumerate(target_matrix):
        point = np.array(row)
        point = point.T

        u = point[0] / 1  # Scale by Z value
        v = point[1] / 1  # Scale by Z value

        '''
        x = row.T[0]
        y = row.T[1]
        z = row.T[2]

        u = x / z
        v = y / z
        '''

        projected[i] = [int(u), int(v)]

    return projected

def read_data(annotations_path, depth_path):
    # Load 2d points.
    cvat_reader = CVATReader(annotations_path)
    data_2d_dict = cvat_reader.get_points()
    all_points_names_2d = data_2d_dict.keys()

    # Read depth map.
    depth_image = np.fromfile(depth_path, dtype='int16', sep="")
    depth_image = np.uint8(depth_image)
    depth_image = depth_image.reshape([640, 480])

    # Init matrix with the data. Order of points is according to KEYPOINTS_NAMES.
    A = np.zeros((len(data_2d_dict), 3))

    for i, key in enumerate(list(data_2d_dict.keys())):
        keypoint = data_2d_dict[key]
        x = np.round(keypoint[0])
        y = np.round(keypoint[1])
        A[i][0] = x
        A[i][1] = y
        A[i][2] = depth_image[int(y)][int(x)]

    A = np.asmatrix(A)

    pcd = o3d.geometry.PointCloud()
    points = []

    for p in A:
        x = float(p.T[0])
        y = float(p.T[1])
        z = float(p.T[2])
        points.append([x, y, z])

    points = np.array(points)
    pcd.points = o3d.utility.Vector3dVector(points)
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(pcd)
    visualizer.run()
    visualizer.close()

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

def read_mean_data(annotations_path):
    FRAME = 0

    # Load 2d points.
    cvat_reader = CVATReader(annotations_path)
    data_2d_dict = cvat_reader.get_points()
    all_points_names_2d = data_2d_dict.keys()

    # Init matrix with the data. Order of points is according to KEYPOINTS_NAMES.
    A = np.zeros((len(data_2d_dict), 3))

    for i, key in enumerate(list(data_2d_dict.keys())):
        keypoint = data_2d_dict[key]
        x = np.round(keypoint[0])
        y = np.round(keypoint[1])
        A[i][0] = x
        A[i][1] = y
        A[i][2] = 0

    A = np.asmatrix(A)

    # Load 3d points.
    reader_3d = VICONReader(VICON_POINTS_CSV_PATH)
    points = reader_3d.get_points()

    #  ---------------- First frame ---------------------
    points_1 = list(points.items())[0+FRAME][1]

    data_3d_dict_1 = {}

    for i, p in enumerate(points_1):
        x = p.x
        y = p.y
        z = p.z
        data_3d_dict_1[KEYPOINTS_NAMES[i]] = [x, y, z]

    # I was not able to pin-pick all points in the images. I did not pinned-pick the points that can't be seen in the
    # image. That's lead to the that the number of 2d points is smaller than the vicon points.
    for key in list(data_3d_dict_1.items()):
        if key[0] not in all_points_names_2d:
            data_3d_dict_1.pop(key[0], None)

    # ---------------- Second frame -------------------
    points_2 = list(points.items())[1+FRAME][1]

    data_3d_dict_2 = {}

    for i, p in enumerate(points_2):
        x = p.x
        y = p.y
        z = p.z
        data_3d_dict_2[KEYPOINTS_NAMES[i]] = [x, y, z]

    # I was not able to pin-pick all points in the images. I did not pinned-pick the points that can't be seen in the
    # image. That's lead to the that the number of 2d points is smaller than the vicon points.
    for key in list(data_3d_dict_2.items()):
        if key[0] not in all_points_names_2d:
            data_3d_dict_2.pop(key[0], None)

    # ---------------- Third frame ---------------------
    points_3 = list(points.items())[2+FRAME][1]

    data_3d_dict_3 = {}

    for i, p in enumerate(points_3):
        x = p.x
        y = p.y
        z = p.z
        data_3d_dict_3[KEYPOINTS_NAMES[i]] = [x, y, z]

    # I was not able to pin-pick all points in the images. I did not pinned-pick the points that can't be seen in the
    # image. That's lead to the that the number of 2d points is smaller than the vicon points.
    for key in list(data_3d_dict_3.items()):
        if key[0] not in all_points_names_2d:
            data_3d_dict_3.pop(key[0], None)

    # Average
    data_3d_dict = {}

    for keypoint_name in list(data_3d_dict_1.keys()):
        p1 = data_3d_dict_1[keypoint_name]
        p2 = data_3d_dict_2[keypoint_name]
        p3 = data_3d_dict_3[keypoint_name]

        x_average = np.nanmean([p1[0], p2[0], p3[0]])
        y_average = np.nanmean([p1[1], p2[1], p3[1]])
        z_average = np.nanmean([p1[2], p2[2], p3[2]])

        p = [x_average, y_average, z_average]
        data_3d_dict[keypoint_name] = p

    B = np.zeros((len(data_3d_dict), 3))

    for i, key in enumerate(data_3d_dict):
        keypoint = data_3d_dict[key]
        B[i][0] = keypoint[0]
        B[i][1] = keypoint[1]
        B[i][2] = keypoint[2]

    B = np.asmatrix(B)

    return A, B

def save_calibration_to_json(path, depth_path, test_name, is_iterative=False, average_points=False):
    if average_points:
        A, B = read_mean_data(path)
    else:
        A, B = read_data(path, depth_path)

    if is_iterative:
        s, R, t = find_rotation_matrix_iterative(annotated_2d_points=A, vicon_3d_points=B)
    else:
        s, R, t = find_rotation_matrix(annotated_2d_points=A, vicon_3d_points=B)

    rmse, _ = calc_rmse(annotated_2d_points=A, vicon_3d_points=B, scale=s, rotation_matrix=R, translation_vector=t)

    R = R.tolist()
    t = t.tolist()

    data = {
        'Scale': s,
        'Rotation': R,
        'Translation': t
    }

    file_name = 'calibration_sub007_{angle}_{name}.json'.format(angle=ANGLE.lower(), name=test_name)
    json_data = json.dumps(data)
    f = open(PATH_ANNOTATIONS + file_name, 'w')
    f.write(json_data)
    f.close()
    copyfile(src=PATH_ANNOTATIONS + file_name, dst=PATH_DATA + file_name)
    return rmse

def test():
    # All points (19).
    rmse = save_calibration_to_json(PIXEL_POINTS_JSON_PATH, DEPTH_IMAGE_PATH, test_name='all_points')
    print("All points error: " + str(rmse))

    # Subset of points (10).
    '''
    rmse = save_calibration_to_json(PIXEL_POINTS_SUB_SET_JSON_PATH, test_name='subset_points')
    print("Subset of points error: " + str(rmse))

    # Remove points with high error rates, and re-rub Kabsch.
    rmse = save_calibration_to_json(PIXEL_POINTS_JSON_PATH, test_name='iterative', is_iterative=True)
    print("Removing points with high error rates error: " + str(rmse))

    # Average the vicon points
    rmse = save_calibration_to_json(PIXEL_POINTS_JSON_PATH, test_name='iterative', average_points=True)
    print("Average vicon points error: " + str(rmse))
    '''


if __name__ == "__main__":
    test()


