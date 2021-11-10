"""
Checking if calculating the Kabsch transformation using the calibration device improves projection accuracy.
"""
import numpy as np
import cv2
import pyrealsense2 as rs
import math
# import open3d as o3d

from data_cleaning.realsense_data_reader import RealSenseReader
from data_cleaning.vicon_data_reader import VICONReader
from data_cleaning.cvat_data_reader import CVATReader
from data_cleaning.projection import project_rs, find_rotation_matrix

FOLDER_PATH = '../../../data/'
DEVICE_BAG_PATH = FOLDER_PATH + 'Board_Front.bag'
DEVICE_CSV_PATH = FOLDER_PATH + 'Calibration_Board.csv'
SUB_BAG_PATH = FOLDER_PATH + 'Lotem_Front.bag'
SUB_CSV_PATH = FOLDER_PATH + 'Lotem.csv'
DEVICE_ANNOTATIONS = FOLDER_PATH + 'device_annotations.json'
SUBJECT_ANNOTATIONS = FOLDER_PATH + 'lotem_annotations.json'
DEVICE_RGB_IMAGE = FOLDER_PATH + "device.png"
DEVICE_POINTS_NAMES = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '110','111', '112', '113', '114',
                       '115', '116', '117', '118', '119', '120', '121', '122', '123', '124']


def calc_euclidean_dist(A, B):
    if type(A) == list:
        A = np.array(A)

    if type(B) == list:
        B = np.array(B)

    distance = np.linalg.norm(A - B)
    mean = np.mean(distance)
    return mean

def read_points_from_deprojection(bag_file_path, annotated_pixels_path):
    """
    For each annotated point, calculate it's reconstructed 3d realsense x,y,z coordiante.

    :param bag_file_path: Path to the bag file.
    :param annotated_pixels_path: Path to json file with annotated points.
    :param kernel_size: Size of the kernel used to average depth pixels.
    :param remove_noisy_depth_points: Whether or not to remove depth points in the background of the object.
    :param points_to_remove: List of specific points to remove, if we know they have high error rates.
    :return: None.
    """
    # Read annotated pixels
    # Load 2d points.
    cvat_reader = CVATReader(annotated_pixels_path, is_calibration=True)
    data_2d_dict = cvat_reader.get_points()

    # Open bag file
    reader = RealSenseReader(bag_file_path=bag_file_path, type='BOTH', frame_rate=30)
    pipe, config = reader.setup_pipeline()
    profile = config.resolve(rs.pipeline_wrapper(pipe))

    frames = pipe.wait_for_frames()

    while frames.frame_number < 100:
        frames = pipe.wait_for_frames()

    align_to = rs.stream.color
    align = rs.align(align_to)
    aligned_frames = align.process(frames)

    # Get depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data(), dtype=np.uint16)

    points_3d = {}

    # Find 3d coordinate.
    for keypoint_name, coordinate in data_2d_dict.items():
        x = int(np.round(coordinate[0]))
        y = int(np.round(coordinate[1]))
        depth_value = aligned_depth_image[x][y]

        depth_value_in_meters = depth_value * depth_scale  # source: https://dev.intelrealsense.com/docs/rs-align-advanced
        # the function rs2_deproject_pixel_to_point expect to get the depth value in meters.
        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth_value_in_meters)

        x = depth_point[0]
        y = depth_point[1]
        z = depth_point[2]

        if math.isnan(x) and math.isnan(y) and math.isnan(z):
            continue

        points_3d[keypoint_name] = [x, y, z]  # Points are in meters!

    # Init matrix with the data. Order of points is according to KEYPOINTS_NAMES.
    A = np.zeros((len(points_3d), 3))

    for i, key in enumerate(list(points_3d.keys())):
        keypoint = points_3d[key]
        x = keypoint[0]
        y = keypoint[1]
        z = keypoint[2]
        # Convert from meters to mm, since the vicon points are in mm!
        A[i][0] = x * 1000
        A[i][1] = y * 1000
        A[i][2] = z * 1000

    #  ----------------------- FOR DEBUGGING: Show points after reconstruction to 3D -----------------------------------
    # pcd = o3d.geometry.PointCloud()
    # points = np.array([(point[0], point[1], point[2]) for point in points_3d.values()])
    # pcd.points = o3d.utility.Vector3dVector(points)
    # visualizer = o3d.visualization.Visualizer()
    # visualizer.create_window(window_name='pixels -> realsense 3d')
    # visualizer.add_geometry(pcd)
    # visualizer.run()
    # visualizer.close()
    # ------------------------------------------------------------------------------------------------------------------

    A = np.asmatrix(A)
    points_names = points_3d.keys()
    return A, points_names

def test():
    # ---------------------------- Calculate Kabsch on the calibration device ------------------------------------------
    # 'clean_points_names' are the names of the remaining points after removing points with noisy depth values.
    points_deprojected, points_names = read_points_from_deprojection(bag_file_path=DEVICE_BAG_PATH,
                                                                           annotated_pixels_path=DEVICE_ANNOTATIONS)
    device_vicon_reader = VICONReader(DEVICE_CSV_PATH, num_points=24)
    all_vicon_points = list(device_vicon_reader.get_points().values())[0]

    # ---------------------------------------------- FOR DEBUGGING -----------------------------------------------------
    #  ----------------------------------------- DRAW ALL VICON POINTS -------------------------------------------------
    # pcd = o3d.geometry.PointCloud()
    # points = np.array([(point.x, point.y, point.z) for point in device_vicon_points])
    # pcd.points = o3d.utility.Vector3dVector(points)
    # visualizer = o3d.visualization.Visualizer()
    # visualizer.create_window(window_name='all vicon points')
    # visualizer.add_geometry(pcd)
    # visualizer.run()
    # visualizer.close()
    # ------------------------------------------------------------------------------------------------------------------

    # Remove all points that were removed in the de-projection
    sub_vicon_points = []

    for p, name in zip(all_vicon_points, DEVICE_POINTS_NAMES):
        if name in points_names:
            sub_vicon_points.append(p)

    # Convert into matrix
    vicon_mat = np.zeros((len(sub_vicon_points), 3))

    for i, p in enumerate(sub_vicon_points):
        vicon_mat[i][0] = p.x
        vicon_mat[i][1] = p.y
        vicon_mat[i][2] = p.z

    vicon_mat = np.matrix(vicon_mat)

    # Calculate transformation with Kabsch.
    s, R, t = find_rotation_matrix(realsense_3d_points=points_deprojected, vicon_3d_points=vicon_mat, scale=False)

    # Find projection error on the calibration device
    # Get annotated points names.
    reader = CVATReader(DEVICE_ANNOTATIONS, is_calibration=True)
    device_annotated_points = reader.get_points()
    projected = project_rs(vicon_3d_points=sub_vicon_points, scale=s, rotation_matrix=R, translation_vector=t,
                           bag_file_path=DEVICE_BAG_PATH, first_frame=100)

    average_dist = calc_euclidean_dist(projected, list(device_annotated_points.values()))
    print(average_dist)

    # Draw points
    image = cv2.imread(DEVICE_RGB_IMAGE)

    for i, row in enumerate(projected):
        x = int(row[0])
        y = int(row[1])

        image = cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=5)

    for i, row in enumerate(device_annotated_points.values()):
        x = int(row[0])
        y = int(row[1])

        image = cv2.circle(image, (x, y), radius=1, color=(0, 0, 255), thickness=5)

    cv2.imwrite("projected.png", image)


if __name__ == "__main__":
    test()