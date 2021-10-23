"""
Project vicon points into realsense pixels.

Pipeline
=======
    1. Annotate points on the RGB image. This should be done manually with the CVAT annotation tool. Get the json file
       of the annotations from CVAT.
    2. For each points, re-construct it's 3d realsense x,y,z coordinates.
    3. Calculate the transformation between the 3d realsense coordinates and the vicon coordinates using Kabsch.
    4. After finding the transformation, apply it on the vicon points and then project them using realsense intrinsics
       parameters.

    Examples are found in the tests below (`test_no_improvements`, `test_with_improvements_1`, `test_with_improvements_2`,
     `test_with_improvements_3`).
"""

import math
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
from typing import List

from data_cleaning.kabsch import kabsch
from data_cleaning.realsense_data_reader import RealSenseReader
from data_cleaning.cvat_data_reader import CVATReader
from data_cleaning.vicon_data_reader import VICONReader, KEYPOINTS_NAMES
from data_cleaning.structs import Point

'''
This is implementation of creating ply file of the pointcloud of the scene. Was not used for the moment.

def read_points_from_pc(bag_file_path):
    # Open bag file
    reader = RealSenseReader(bag_file_path=bag_file_path, type='DEPTH', frame_rate=30)
    pipe, _ = reader.setup_pipeline()

    # Get point cloud of first frame
    colorizer = rs.colorizer()

    # Wait for the next set of frames from the camera
    frames = pipe.wait_for_frames()

    while frames.frame_number < 5842:
        frames = pipe.wait_for_frames()

    print("Found first frame. Starting reading point cloud...")

    colorized = colorizer.process(frames)

    # Create save_to_ply object
    ply = rs.save_to_ply("pc.ply")

    # Set options to the desired values
    # In this example we'll generate a textual PLY with normals (mesh is already created by default)
    ply.set_option(rs.save_to_ply.option_ply_binary, False)
    ply.set_option(rs.save_to_ply.option_ply_normals, True)

    print("Saving to ply file...")
    # Apply the processing block to the frameset which contains the depth frame and the texture
    ply.process(colorized)
    print("Done")

    # Read points
    pcd = o3d.io.read_point_cloud("pc.ply") # TODO: Is the points are in meters?
    points = np.asarray(pcd.points)

    # Visualize pointcloud
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

    return points
'''

def read_points_from_deprojection(bag_file_path, annotated_pixels_path, kernel_size=31, remove_noisy_depth_points=True,
                                  points_to_remove=None):
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
    cvat_reader = CVATReader(annotated_pixels_path)
    data_2d_dict = cvat_reader.get_points()

    # Remove points that i found out that had high error rates
    if points_to_remove is not None:
        for point_name in points_to_remove:
            data_2d_dict.pop(point_name, None)

    # Open bag file
    reader = RealSenseReader(bag_file_path=bag_file_path, type='BOTH', frame_rate=30)
    pipe, config = reader.setup_pipeline()
    profile = config.resolve(rs.pipeline_wrapper(pipe))

    frames = pipe.wait_for_frames()

    while frames.frame_number < 5842:
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

    # Will be used to remove all points behind the clipping distance, to reduce noise.
    clipping_distance_in_meters = 3  # in meters
    clipping_distance = clipping_distance_in_meters / depth_scale

    points_3d = {}

    # Find 3d coordinate. Find z value based on average of kernel*kernel neighbor points to reduce noise
    for keypoint_name, coordinate in data_2d_dict.items():
        # Add all points of kernel size
        neighborhood_points = []
        kernel_size = kernel_size
        for i in range(-1 * int(kernel_size / 2), int(kernel_size / 2) + 1):
            for j in range(-1 * int(kernel_size / 2), int(kernel_size / 2) + 1):
                x = int(np.round(coordinate[0])) + i
                y = int(np.round(coordinate[1])) + j
                depth_pixel = [x, y]
                depth_value = aligned_depth_image[depth_pixel[0]][depth_pixel[1]]

                if remove_noisy_depth_points and depth_value > clipping_distance:
                    continue

                if depth_value < 1:
                    continue

                depth_value_in_meters = depth_value * depth_scale  # source: https://dev.intelrealsense.com/docs/rs-align-advanced
                # the function rs2_deproject_pixel_to_point expect to get the depth value in meters.
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_value_in_meters)

                # -------------------------- FOR DEBUGGING: Check re-construction works --------------------------------
                pixel = rs.rs2_project_point_to_pixel(depth_intrin, depth_point)
                pixel[0], pixel[1] = int(np.round(pixel[0])), int(np.round(pixel[1]))
                assert pixel == depth_pixel
                # ------------------------------------------------------------------------------------------------------

                neighborhood_points.append(depth_point)

        if len(neighborhood_points) == 0:
            continue

        x = np.mean([p[0] for p in neighborhood_points])
        y = np.mean([p[1] for p in neighborhood_points])
        z = np.mean([p[2] for p in neighborhood_points])

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

    #  -------------------------------------- FOR DEBUGGING: Show points -----------------------------------------------
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

def read_vicon_points(vicon_path, points_names, frames_to_average=None):
    """
    Read vicon points.

    :param vicon_path: Path to csv file.
    :param points_names: Names of the clean points we extracted from depth.
    :param frames_to_average: Number of frames to use for averaging the points. Starting from the first frames.
    :return: Matrix with the points.
    """
    # Load 3d points.
    reader_3d = VICONReader(vicon_path) # Vicon points are in mm!
    points = reader_3d.get_points()

    if frames_to_average is None:
        points = list(points.items())[0][1] # Get first frame's points
    else:
        # Average each point for 'frames_to_average' frames, starting from first frame.
        NUMBER_OF_POINTS = 39
        temp_dict = {}

        for k in range(NUMBER_OF_POINTS):
            temp_dict[k] = []

        for i in range(frames_to_average):
            frame_points = list(points.items())[i][1]

            for j in range(NUMBER_OF_POINTS):
                temp_dict[j].append(frame_points[j])

        average_points = []

        for v in temp_dict.values():
            x = []
            y = []
            z = []

            for i in range(frames_to_average):
                x.append(v[i].x)
                y.append(v[i].y)
                z.append(v[i].z)

            x = np.mean(x)
            y = np.mean(y)
            z = np.mean(z)
            average_points.append(Point(x, y, z))

        points = average_points

    data_3d_dict = {}

    for i, p in enumerate(points):
        x = p.x
        y = p.y
        z = p.z
        data_3d_dict[KEYPOINTS_NAMES[i]] = [x, y, z]

    # I was not able to pin-pick all points in the images. I did not pinned-pick the points that can't be seen in the
    # image. That's lead to the that the number of 2d points is smaller than the vicon points.
    for key in list(data_3d_dict.items()):
        if key[0] not in points_names:
            data_3d_dict.pop(key[0], None)

    # Rotate points. The axes system of the vicon points and realsense 3d points are different. I've decided to rotate
    # them all according to the the open3d default axes system.
    rotated_points = rotate_vicon_points_90_degrees_counterclockwise(rotation_axis='x', points=data_3d_dict)
    rotated_points = rotate_vicon_points_90_degrees_counterclockwise(rotation_axis='x', points=rotated_points)
    rotated_points = rotate_vicon_points_90_degrees_counterclockwise(rotation_axis='x', points=rotated_points)
    data_3d_dict = rotated_points

    B = np.zeros((len(data_3d_dict), 3))

    for i, key in enumerate(data_3d_dict):
        keypoint = data_3d_dict[key]
        B[i][0] = keypoint[0]
        B[i][1] = keypoint[1]
        B[i][2] = keypoint[2]

     #  -------------------------------------- FOR DEBUGGING: Show points -----------------------------------------------
    # pcd = o3d.geometry.PointCloud()
    # points = np.array([(point[0], point[1], point[2]) for point in data_3d_dict.values()])
    # pcd.points = o3d.utility.Vector3dVector(points)
    # visualizer = o3d.visualization.Visualizer()
    # visualizer.create_window(window_name='vicon 3d')
    # visualizer.add_geometry(pcd)
    # visualizer.run()
    # visualizer.close()
    # ------------------------------------------------------------------------------------------------------------------

    B = np.asmatrix(B)
    return B

def find_rotation_matrix(realsense_3d_points, vicon_3d_points, scale):
    s, ret_R, ret_t = kabsch(A=realsense_3d_points, B=vicon_3d_points, scale=scale)
    return s, ret_R, ret_t

''''
This is my implementation of the projection function, but i have decided to use realsense API implementation (`project_rs`)
I'm leaving it just in case i would need to implement it in the future by myself for some reason...
def project(vicon_3d_points, scale, rotation_matrix, translation_vector):
    """
    Project vicon points to realsense pixels, after applying on them the transformation found with Kabsch.

    :param vicon_3d_points: Vicon points.
    :param scale: Scale factor produced by Kabsch.
    :param rotation_matrix: Rotation matrix produced by Kabsch.
    :param translation_vector: Translation vector produced by Kabsch.
    :return: Projected 2d points.
    """

    # Vicon 3d -> realsense 3d
    B = np.zeros((len(vicon_3d_points), 3))

    for i, keypoint in enumerate(vicon_3d_points):
        B[i][0] = keypoint.x
        B[i][1] = keypoint.y
        B[i][2] = keypoint.z

    B = np.asmatrix(B) # B is in mm

    N = len(vicon_3d_points)
    B = np.array(np.dot(B.T, scale))
    k = np.array(np.dot(rotation_matrix, B))
    t = np.array(np.tile(translation_vector, (1, N)))
    target_matrix = k + t
    target_matrix = target_matrix.T

    # Realsense 3d -> pixels
    projected = np.zeros((N, 2))
    projection_matrix = np.zeros((3, 3))
    projection_matrix[0][0] = 613.2305 # fx
    projection_matrix[1][1] = 613.3134 # fy
    projection_matrix[2][2] = 1
    projection_matrix[0][2] = 319.8690 # ppx
    projection_matrix[1][2] = 245.8685 # ppy

    for i, row in enumerate(target_matrix):
        point = np.array(row)
        point = point.T
        uvw = np.dot(projection_matrix, point)
        u = (uvw[0] / uvw[2]) # Scale by Z value
        v = (uvw[1] / uvw[2]) # Scale by Z value
        projected[i] = [int(u), int(v)]

    return projected
'''

def project_rs(vicon_3d_points, scale, rotation_matrix, translation_vector, bag_file_path, first_frame=5842):
    """
    Project vicon points to realsense pixels, after applying on them the transformation found with Kabsch. Projecting is
    done with realsense API.

    :param vicon_3d_points: Vicon points.
    :param scale: Scale factor produced by Kabsch.
    :param rotation_matrix: Rotation matrix produced by Kabsch.
    :param translation_vector: Translation vector produced by Kabsch.
    :param bag_file_path: Bag file path.
    :return: Projected 2d points.
    """
    # Vicon 3d -> realsense 3d
    # Rotate points. The axes system of the vicon points and realsense 3d points are different. I've decided to rotate
    # them all according to the the open3d default axes system.
    dummy_dict = {}

    for i, p in enumerate(vicon_3d_points):
        dummy_dict[i] = p

    rotated_points = rotate_vicon_points_90_degrees_counterclockwise(rotation_axis='x', points=dummy_dict)
    rotated_points = rotate_vicon_points_90_degrees_counterclockwise(rotation_axis='x', points=rotated_points)
    rotated_points = rotate_vicon_points_90_degrees_counterclockwise(rotation_axis='x', points=rotated_points)
    vicon_3d_points = [p for k, p in rotated_points.items()]

    B = np.zeros((len(vicon_3d_points), 3))

    for i, keypoint in enumerate(vicon_3d_points):
        B[i][0] = keypoint[0]
        B[i][1] = keypoint[1]
        B[i][2] = keypoint[2]

    B = np.asmatrix(B)  # B is in mm

    N = len(vicon_3d_points)
    B = np.array(np.dot(B.T, scale))
    k = np.array(np.dot(rotation_matrix, B))
    t = np.array(np.tile(translation_vector, (1, N)))
    target_matrix = k + t
    target_matrix = target_matrix.T

    #  -------------------------------------- FOR DEBUGGING: Show points -----------------------------------------------
    # pcd = o3d.geometry.PointCloud()
    # points = np.array([(point[0], point[1], point[2]) for point in target_matrix])
    # pcd.points = o3d.utility.Vector3dVector(points)
    # visualizer = o3d.visualization.Visualizer()
    # visualizer.create_window(window_name='vicon 3d -> realsense 3d')
    # visualizer.add_geometry(pcd)
    # visualizer.run()
    # visualizer.close()
    # ------------------------------------------------------------------------------------------------------------------

    # Realsense 3d -> pixels
    depth_intrin = get_intrinsics(bag_file_path=bag_file_path, first_frame=first_frame)
    pixels = []

    for i, row in enumerate(target_matrix):
        point = np.array(row)
        pixel = rs.rs2_project_point_to_pixel(depth_intrin, point)
        pixels.append(pixel)

    return pixels

def get_intrinsics(bag_file_path, first_frame):
    """
    Get intrinsics parameters of the camera.

    :param bag_file_path: Path to the bag file.
    :param first_frame: The frame where the object is doint T-pose.
    :return: Camera intrinsics.
    """
    reader = RealSenseReader(bag_file_path=bag_file_path, type='BOTH', frame_rate=30)
    pipe, config = reader.setup_pipeline()
    frames = pipe.wait_for_frames()

    while frames.frame_number < first_frame: # 5842
        frames = pipe.wait_for_frames()

    align_to = rs.stream.color
    align = rs.align(align_to)
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()
    intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

    return intrin

def rotate_vicon_points_90_degrees_counterclockwise(rotation_axis: str,  points):
    vicon_points = points

    if rotation_axis.lower() == 'x':
        rotation_matix = np.ndarray((3, 3), dtype=float)
        rotation_matix[0] = [1, 0, 0]
        rotation_matix[1] = [0, 0, 1]
        rotation_matix[2] = [0, -1, 0]
    elif rotation_axis.lower() == 'y':
        rotation_matix = np.ndarray((3, 3), dtype=float)
        rotation_matix[0] = [0, 0, -1]
        rotation_matix[1] = [0, 1, 0]
        rotation_matix[2] = [1, 0, 0]
    elif rotation_axis.lower() == 'z':
        rotation_matix = np.ndarray((3, 3), dtype=float)
        rotation_matix[0] = [0, 1, 0]
        rotation_matix[1] = [-1, 0, 0]
        rotation_matix[2] = [0, 0, 1]
    else:
        print('Wrong axis. Use \"x\",  \"y\" or \"z\".')
        return

    final_points = {}

    for keypoint_name, point in vicon_points.items():
        try:
            transformed = rotation_matix.dot(np.array([point.x, point.y, point.z]))
        except AttributeError:
            transformed = rotation_matix.dot(np.array([point[0], point[1], point[2]]))

        rotated_point = [transformed[0], transformed[1], transformed[2]]
        final_points[keypoint_name] = rotated_point

    return final_points

def calc_rmse(A, B):
    N = A.shape[0]
    diff = A - B
    err = A - B
    err = np.multiply(err, err)
    err = np.sum(err)
    rmse = math.sqrt(err / N)
    return rmse, diff

def calc_rmse_projected_points(all_annotated_points_names, annotated_pixels_path, projected_points):
    """
    Calculate RMSE between annotated pixel points and projected points.

    :param all_annotated_points_names: N points names.
    :param annotated_pixels_path: Path to json file of annotated points.
    :param projected_points: Points after applying Kabsch transformation and projection.
    :return: rmse.
    """
    points_indices = [i for i, keypoint in enumerate(KEYPOINTS_NAMES) if keypoint in all_annotated_points_names]

    A = np.zeros((len(all_annotated_points_names), 2))  # Annotated pixels.
    B = np.zeros((len(all_annotated_points_names), 2))  # Projected pixels.

    annotated_points = CVATReader(annotated_pixels_path).get_points()

    for k in list(annotated_points.keys()):
        if k not in all_annotated_points_names:
            annotated_points.pop(k, None)

    for i, (keypoint_name, v) in enumerate(annotated_points.items()):
        A[i][0] = v[0]
        A[i][1] = v[1]

    counter = 0
    for i, row in enumerate(projected_points):
        if i not in points_indices:
            continue

        B[counter][0] = row[0]
        B[counter][1] = row[1]
        counter += 1

    error, _ = calc_rmse(A=A, B=B)
    print("{number} of points were used in RMSE calculation.".format(number=A.shape[0]))

    return error

def calc_euclidean_dist(vicon_csv_path, points_deprojected, clean_points_names, scale, rotation_matrix, translate_vector):
    """
    Calculate the euclidean distance between each transformed vicon point to it's de-projected 3d realsense.
    Assuming in both matrices, each row 'i' refers to the same keypoint.

    :param A: N vicon points after applying transformation on them.
    :param B: N realsense 3d points, after reconstructing them from pixels.
    :param points_names: List of points names.
    :return: Dictionary of <keypoint, distance>.
    """
    vicon_reader = VICONReader(vicon_csv_path)
    points_indices = [i for i, keypoint in enumerate(KEYPOINTS_NAMES) if keypoint in clean_points_names]
    A = np.zeros((len(clean_points_names), 3))  # Vicon points after applying transformation.
    B = np.copy(points_deprojected)  # 3d realsense points after reconstructing them from pixels.

    all_vicon_points = vicon_reader.get_points()
    all_vicon_points = list(all_vicon_points.items())[0][1]  # Get first frame's points
    temp_list = []

    for i, p in enumerate(all_vicon_points):
        if i in points_indices:
            temp_list.append(p)

    all_vicon_points = temp_list
    dummy_dict = {}

    for i, p in enumerate(all_vicon_points):
        dummy_dict[i] = p

    # Fix vicon axes system to fit to realsense 3d axes system.
    rotated_points = rotate_vicon_points_90_degrees_counterclockwise(rotation_axis='x', points=dummy_dict)
    rotated_points = rotate_vicon_points_90_degrees_counterclockwise(rotation_axis='x', points=rotated_points)
    rotated_points = rotate_vicon_points_90_degrees_counterclockwise(rotation_axis='x', points=rotated_points)
    vicon_3d_points = [p for k, p in rotated_points.items()]

    P = np.zeros((len(clean_points_names), 3))

    for i, keypoint in enumerate(vicon_3d_points):
        try:
            P[i][0] = keypoint[0]
            P[i][1] = keypoint[1]
            P[i][2] = keypoint[2]
        except TypeError:
            P[i][0] = keypoint.x
            P[i][1] = keypoint.y
            P[i][2] = keypoint.z

    N = len(vicon_3d_points)
    P = np.array(np.dot(P.T, scale))
    k = np.array(np.dot(rotation_matrix, P))
    t = np.array(np.tile(translate_vector, (1, N)))
    A = k + t
    A = A.T

    # --------------------------------------- FOR DEBUGGING: Visualize points ------------------------------------------
    # # A
    # pcd = o3d.geometry.PointCloud()
    # points = np.array([(point[0], point[1], point[2]) for point in A])
    # pcd.points = o3d.utility.Vector3dVector(points)
    # visualizer = o3d.visualization.Visualizer()
    # visualizer.create_window(window_name='vicon -> realsense 3d')
    # visualizer.add_geometry(pcd)
    # visualizer.run()
    # visualizer.close()
    #
    # # B
    # pcd = o3d.geometry.PointCloud()
    # points = np.array([(point[0], point[1], point[2]) for point in B])
    # pcd.points = o3d.utility.Vector3dVector(points)
    # visualizer = o3d.visualization.Visualizer()
    # visualizer.create_window(window_name='pixels -> realsense 3d')
    # visualizer.add_geometry(pcd)
    # visualizer.run()
    # visualizer.close()
    # ------------------------------------------------------------------------------------------------------------------

    if type(clean_points_names) is not List:
        clean_points_names = list(clean_points_names)

    distance_dict = {} # Distances are in mm
    N = A.shape[0]

    for i in range(N):
        p1 = A[i]
        p2 = B[i]
        distance = np.linalg.norm(p1-p2) # Numpy's implementation for euclidean distance
        keypoint_name = clean_points_names[i]
        distance_dict[keypoint_name] = distance

    distances = list(distance_dict.values())
    average_distance = np.mean(distances)
    return distance_dict, average_distance

def find_best_kernel_size():
    """
    Check several kernel sizes to decide which one is best for averaging depth pixels.

    :return: None.
    """
    # Some consts.
    bag_path = '../../data/Sub007_Left_Front.bag'
    annotated_pixels_path = '../../annotations_data/Sub007/Front/annotations_all.json'
    vicon_csv_path = '../../annotations_data/Sub007/Front/vicon_points.csv'
    rgb_frame_path = '../../annotations_data/Sub007/Front/rgb_frame.png'

    # Get annotated points names, for later visualizing.
    cvat_reader = CVATReader(annotated_pixels_path)
    data_2d_dict = cvat_reader.get_points()
    annotated_points_names = data_2d_dict.keys()

    min_error = np.inf
    min_kernel = 0
    errors = []

    # Get points.
    for kernel in range(1, 87, 2):
        points_deprojected, clean_points_names = read_points_from_deprojection(bag_file_path=bag_path,
                                                                               annotated_pixels_path=annotated_pixels_path,
                                                                               kernel_size=kernel)
        vicon_points = read_vicon_points(vicon_path=vicon_csv_path, points_names=clean_points_names)

        # Calculate transformation with Kabsch.
        s, R, t = find_rotation_matrix(realsense_3d_points=points_deprojected, vicon_3d_points=vicon_points, scale=False)

        # Project the points.
        vicon_reader = VICONReader(vicon_csv_path)
        all_vicon_points = vicon_reader.get_points()
        all_vicon_points = list(all_vicon_points.items())[0][1]  # Get first frame's points
        projected = project_rs(vicon_3d_points=all_vicon_points, scale=s, rotation_matrix=R, translation_vector=t,
                               bag_file_path=bag_path)

        # Find the error of the projection - RMSE
        error = calc_rmse_projected_points(annotated_points_names, annotated_pixels_path, projected)
        errors.append(error)

        if error < min_error:
            min_error = error
            min_kernel = kernel

        print("Kernel: {kernel}, RMSE: {error}".format(kernel=kernel, error=error))

    print("Best kernel is {kernel}, with error of {error}".format(kernel=min_kernel, error=min_error))

def test_no_improvements():
    """
    Project vicon points into the realsense pixels. No improvements on the data.

    :return: None.
    """
    # Some consts.
    bag_path = '../../data/Sub007_Left_Front.bag'
    annotated_pixels_path = '../../annotations_data/Sub007/Front/annotations_all.json'
    vicon_csv_path = '../../annotations_data/Sub007/Front/vicon_points.csv'
    rgb_frame_path = '../../annotations_data/Sub007/Front/rgb_frame.png'

    print("Finding Kabsch without improvements...")

    # Get annotated points names.
    reader = CVATReader(annotated_pixels_path)
    annotated_points_names = reader.get_points().keys()

    # Get points.
    # 'clean_points_names' are the names of the remaining points after removing points with noisy depth values.
    points_deprojected, clean_points_names = read_points_from_deprojection(bag_file_path=bag_path,
                                                                     annotated_pixels_path=annotated_pixels_path,
                                                                           kernel_size=1, # Best kernel size is 31
                                                                           remove_noisy_depth_points=False)
    vicon_points = read_vicon_points(vicon_path=vicon_csv_path, points_names=clean_points_names)

    # --------------------------------------------- Without scaling ---------------------------------------------------
    # Calculate transformation with Kabsch.
    s, R, t = find_rotation_matrix(realsense_3d_points=points_deprojected, vicon_3d_points=vicon_points, scale=False)

    # Project the points.
    vicon_reader = VICONReader(vicon_csv_path)
    all_vicon_points = vicon_reader.get_points()
    all_vicon_points = list(all_vicon_points.items())[0][1] # Get first frame's points
    projected = project_rs(vicon_3d_points=all_vicon_points, scale=s, rotation_matrix=R, translation_vector=t,
                           bag_file_path=bag_path)

    # Draw points
    image = cv2.imread(rgb_frame_path)
    points_indices = [i for i, keypoint in enumerate(KEYPOINTS_NAMES) if keypoint in annotated_points_names]

    for i, row in enumerate(projected):
        x = int(row[0])
        y = int(row[1])

        if i not in points_indices:
            continue

        if math.isnan(x) or math.isnan(y):
            continue

        image = cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=5)

    cv2.imwrite("projected_without_scale_without_improvements.png", image)

    # Find the error of the projection - RMSE
    error = calc_rmse_projected_points(clean_points_names, annotated_pixels_path, projected)
    print("Without scaling RMSE: " + str(error))

    # Find the error of Kabsch transformation - Euclidean distance.
    distances, average_dist = calc_euclidean_dist(vicon_csv_path=vicon_csv_path, points_deprojected=points_deprojected,
                                          clean_points_names=clean_points_names, scale=s, rotation_matrix=R,
                                          translate_vector=t)
    print("Without scaling average euclidean distance: " + str(average_dist))

    # ---------------------------------------------- With scaling -----------------------------------------------------
    # Calculate transformation with Kabsch.
    s, R, t = find_rotation_matrix(realsense_3d_points=points_deprojected, vicon_3d_points=vicon_points, scale=True)

    # Project the points.
    vicon_reader = VICONReader(vicon_csv_path)
    all_vicon_points = vicon_reader.get_points()
    all_vicon_points = list(all_vicon_points.items())[0][1]  # Get first frame's points
    projected = project_rs(vicon_3d_points=all_vicon_points, scale=s, rotation_matrix=R, translation_vector=t,
                           bag_file_path=bag_path)

    # Draw points
    image = cv2.imread(rgb_frame_path)

    for i, row in enumerate(projected):
        x = int(row[0])
        y = int(row[1])

        if i not in points_indices:
            continue

        if math.isnan(x) or math.isnan(y):
            continue

        image = cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=5)

    cv2.imwrite("projected_with_scale_without_improvements.png", image)

    # Find the error of the projection - RMSE
    error = calc_rmse_projected_points(clean_points_names, annotated_pixels_path, projected)
    print("With scaling RMSE: " + str(error))

    # Find the error of Kabsch transformation - Euclidean distance.
    distances, average_dist = calc_euclidean_dist(vicon_csv_path=vicon_csv_path, points_deprojected=points_deprojected,
                                          clean_points_names=clean_points_names, scale=s, rotation_matrix=R,
                                          translate_vector=t)
    print("With scaling average euclidean distance: " + str(average_dist))
    print("Scale is {s}".format(s=s))
    print("--------------------------------------------------")

def test_with_improvements_1():
    """
    Project vicon points into the realsense pixels.

    Improvements:
       - Removing noisy depth points.
       - Averaging depth values over space (best kernel is 31).
    """
    # Some consts.
    bag_path = '../../data/Sub007_Left_Front.bag'
    annotated_pixels_path = '../../annotations_data/Sub007/Front/annotations_all.json'
    vicon_csv_path = '../../annotations_data/Sub007/Front/vicon_points.csv'
    rgb_frame_path = '../../annotations_data/Sub007/Front/rgb_frame.png'

    print("Finding Kabsch with following improvements: Removing noisy depth points, averaging depth values over space...")
    # Get points.

    # Get annotated points names.
    reader = CVATReader(annotated_pixels_path)
    annotated_points_names = reader.get_points().keys()

    # 'clean_points_names' are the names of the remaining points after removing points with noisy depth values.
    points_deprojected, clean_points_names = read_points_from_deprojection(bag_file_path=bag_path,
                                                                     annotated_pixels_path=annotated_pixels_path,
                                                                           kernel_size=31, # Best kernel size is 31
                                                                           remove_noisy_depth_points=True)
    vicon_points = read_vicon_points(vicon_path=vicon_csv_path, points_names=clean_points_names)

    # --------------------------------------------- Without scaling ---------------------------------------------------
    # Calculate transformation with Kabsch.
    s, R, t = find_rotation_matrix(realsense_3d_points=points_deprojected, vicon_3d_points=vicon_points, scale=False)

    # Project the points.
    vicon_reader = VICONReader(vicon_csv_path)
    all_vicon_points = vicon_reader.get_points()
    all_vicon_points = list(all_vicon_points.items())[0][1] # Get first frame's points
    projected = project_rs(vicon_3d_points=all_vicon_points, scale=s, rotation_matrix=R, translation_vector=t,
                           bag_file_path=bag_path)

    # Draw points
    image = cv2.imread(rgb_frame_path)
    points_indices = [i for i, keypoint in enumerate(KEYPOINTS_NAMES) if keypoint in annotated_points_names]

    for i, row in enumerate(projected):
        x = int(row[0])
        y = int(row[1])

        if i not in points_indices:
            continue

        if math.isnan(x) or math.isnan(y):
            continue

        image = cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=5)

    cv2.imwrite("projected_without_scale_with_improvements_1.png", image)

    # Find the error of the projection - RMSE
    error = calc_rmse_projected_points(clean_points_names, annotated_pixels_path, projected)
    print("Without scaling RMSE: " + str(error))

    # Find the error of Kabsch transformation - Euclidean distance.
    distances, average_dist = calc_euclidean_dist(vicon_csv_path=vicon_csv_path, points_deprojected=points_deprojected,
                                          clean_points_names=clean_points_names, scale=s, rotation_matrix=R,
                                          translate_vector=t)
    print("Without scaling average euclidean distance: " + str(average_dist))

    # ---------------------------------------------- With scaling -----------------------------------------------------
    # Calculate transformation with Kabsch.
    s, R, t = find_rotation_matrix(realsense_3d_points=points_deprojected, vicon_3d_points=vicon_points, scale=True)

    # Project the points.
    vicon_reader = VICONReader(vicon_csv_path)
    all_vicon_points = vicon_reader.get_points()
    all_vicon_points = list(all_vicon_points.items())[0][1]  # Get first frame's points
    projected = project_rs(vicon_3d_points=all_vicon_points, scale=s, rotation_matrix=R, translation_vector=t,
                           bag_file_path=bag_path)

    # Draw points
    image = cv2.imread(rgb_frame_path)

    for i, row in enumerate(projected):
        x = int(row[0])
        y = int(row[1])

        if i not in points_indices:
            continue

        if math.isnan(x) or math.isnan(y):
            continue

        image = cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=5)

    cv2.imwrite("projected_with_scale_with_improvements_1.png", image)

    # Find the error of the projection - RMSE
    error = calc_rmse_projected_points(clean_points_names, annotated_pixels_path, projected)
    print("With scaling RMSE: " + str(error))

    # Find the error of Kabsch transformation - Euclidean distance.
    distances, average_dist = calc_euclidean_dist(vicon_csv_path=vicon_csv_path, points_deprojected=points_deprojected,
                                          clean_points_names=clean_points_names, scale=s, rotation_matrix=R,
                                          translate_vector=t)
    print("With scaling average euclidean distance: " + str(average_dist))
    print("Scale is {s}".format(s=s))
    print("--------------------------------------------------")

def test_with_improvements_2():
    """
    Project vicon points into the realsense pixels.

    Improvements:
       - Removing noisy depth points.
       - Averaging depth values over space (best kernel is 31).
       - Calculating Kabsch and finding error for each point, removing points with high error rates and re-calculating
         Kabsch.
    """
    # Some consts.
    bag_path = '../../data/Sub007_Left_Front.bag'
    annotated_pixels_path = '../../annotations_data/Sub007/Front/annotations_all.json'
    vicon_csv_path = '../../annotations_data/Sub007/Front/vicon_points.csv'
    rgb_frame_path = '../../annotations_data/Sub007/Front/rgb_frame.png'

    print("Finding Kabsch with following improvements: Removing noisy depth points, averaging depth values over space,"
          " removing points with high error rates and re-calulcating Kabsch...")
    # Get points.

    # Get annotated points names.
    reader = CVATReader(annotated_pixels_path)
    annotated_points_names = reader.get_points().keys()

    # 'clean_points_names' are the names of the remaining points after removing points with noisy depth values.
    points_deprojected, clean_points_names = read_points_from_deprojection(bag_file_path=bag_path,
                                                                           annotated_pixels_path=annotated_pixels_path,
                                                                           kernel_size=31, # Best kernel size is 31
                                                                           remove_noisy_depth_points=True)
    vicon_points = read_vicon_points(vicon_path=vicon_csv_path, points_names=clean_points_names)

    # --------------------------------------------- Without scaling ---------------------------------------------------
    # Calculate transformation with Kabsch.
    s, R, t = find_rotation_matrix(realsense_3d_points=points_deprojected, vicon_3d_points=vicon_points, scale=False)

    # Find the error of Kabsch transformation - Euclidean distance.
    distances, average_dist = calc_euclidean_dist(vicon_csv_path=vicon_csv_path, points_deprojected=points_deprojected,
                                          clean_points_names=clean_points_names, scale=s, rotation_matrix=R,
                                          translate_vector=t)

    # Find 3 points with biggest distance, remove them and re-run Kabsch.
    sorted_points = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
    p1_name = sorted_points.popitem()[0]
    p2_name = sorted_points.popitem()[0]
    p3_name = sorted_points.popitem()[0]

    # Start all over again with new points...
    points_deprojected, clean_points_names = read_points_from_deprojection(bag_file_path=bag_path,
                                                                           annotated_pixels_path=annotated_pixels_path,
                                                                           kernel_size=31, # Best kernel size is 31
                                                                           remove_noisy_depth_points=True,
                                                                           points_to_remove=[p1_name, p2_name, p3_name])
    vicon_points = read_vicon_points(vicon_path=vicon_csv_path, points_names=clean_points_names)

    # Calculate transformation with Kabsch.
    s, R, t = find_rotation_matrix(realsense_3d_points=points_deprojected, vicon_3d_points=vicon_points, scale=False)

    # Project the points.
    vicon_reader = VICONReader(vicon_csv_path)
    all_vicon_points = vicon_reader.get_points()
    all_vicon_points = list(all_vicon_points.items())[0][1]  # Get first frame's points

    projected = project_rs(vicon_3d_points=all_vicon_points, scale=s, rotation_matrix=R, translation_vector=t,
                           bag_file_path=bag_path)

    # Draw points
    image = cv2.imread(rgb_frame_path)
    points_indices = [i for i, keypoint in enumerate(KEYPOINTS_NAMES) if keypoint in annotated_points_names]

    for i, row in enumerate(projected):
        x = int(row[0])
        y = int(row[1])

        if i not in points_indices:
            continue

        if math.isnan(x) or math.isnan(y):
            continue

        image = cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=5)

    cv2.imwrite("projected_without_scale_with_improvements_2.png", image)

    # Find the error of the projection - RMSE
    error = calc_rmse_projected_points(clean_points_names, annotated_pixels_path, projected)
    print("Without scaling RMSE: " + str(error))

    # Find the error of Kabsch transformation - Euclidean distance.
    distances, average_dist = calc_euclidean_dist(vicon_csv_path=vicon_csv_path, points_deprojected=points_deprojected,
                                                  clean_points_names=clean_points_names, scale=s, rotation_matrix=R,
                                                  translate_vector=t)
    print("Without scaling average euclidean distance: " + str(average_dist))

    # ---------------------------------------------- With scaling -----------------------------------------------------
    # Get points.
    # 'clean_points_names' are the names of the remaining points after removing points with noisy depth values.
    points_deprojected, clean_points_names = read_points_from_deprojection(bag_file_path=bag_path,
                                                                           annotated_pixels_path=annotated_pixels_path,
                                                                           kernel_size=31,  # Best kernel size is 31
                                                                           remove_noisy_depth_points=True)
    vicon_points = read_vicon_points(vicon_path=vicon_csv_path, points_names=clean_points_names)

    # Calculate transformation with Kabsch.
    s, R, t = find_rotation_matrix(realsense_3d_points=points_deprojected, vicon_3d_points=vicon_points, scale=True)

    # Find the error of Kabsch transformation - Euclidean distance.
    distances, average_dist = calc_euclidean_dist(vicon_csv_path=vicon_csv_path, points_deprojected=points_deprojected,
                                                  clean_points_names=clean_points_names, scale=s, rotation_matrix=R,
                                                  translate_vector=t)

    # Find 3 points with biggest distance, remove them and re-run Kabsch.
    sorted_points = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
    p1_name = sorted_points.popitem()[0]
    p2_name = sorted_points.popitem()[0]
    p3_name = sorted_points.popitem()[0]

    # Start all over again with new points...
    points_deprojected, clean_points_names = read_points_from_deprojection(bag_file_path=bag_path,
                                                                           annotated_pixels_path=annotated_pixels_path,
                                                                           kernel_size=31,  # Best kernel size is 31
                                                                           remove_noisy_depth_points=True,
                                                                           points_to_remove=[p1_name, p2_name, p3_name])
    vicon_points = read_vicon_points(vicon_path=vicon_csv_path, points_names=clean_points_names)

    # Calculate transformation with Kabsch.
    s, R, t = find_rotation_matrix(realsense_3d_points=points_deprojected, vicon_3d_points=vicon_points, scale=True)

    # Project the points.
    vicon_reader = VICONReader(vicon_csv_path)
    all_vicon_points = vicon_reader.get_points()
    all_vicon_points = list(all_vicon_points.items())[0][1]  # Get first frame's points

    projected = project_rs(vicon_3d_points=all_vicon_points, scale=s, rotation_matrix=R, translation_vector=t,
                           bag_file_path=bag_path)

    # Draw points
    image = cv2.imread(rgb_frame_path)
    points_indices = [i for i, keypoint in enumerate(KEYPOINTS_NAMES) if keypoint in annotated_points_names]

    for i, row in enumerate(projected):
        x = int(row[0])
        y = int(row[1])

        if i not in points_indices:
            continue

        if math.isnan(x) or math.isnan(y):
            continue

        image = cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=5)

    cv2.imwrite("projected_with_scale_with_improvements_2.png", image)

    # Find the error of the projection - RMSE
    error = calc_rmse_projected_points(clean_points_names, annotated_pixels_path, projected)
    print("With scaling RMSE: " + str(error))

    # Find the error of Kabsch transformation - Euclidean distance.
    distances, average_dist = calc_euclidean_dist(vicon_csv_path=vicon_csv_path, points_deprojected=points_deprojected,
                                                  clean_points_names=clean_points_names, scale=s, rotation_matrix=R,
                                                  translate_vector=t)
    print("With scaling average euclidean distance: " + str(average_dist))
    print("Scale is {s}".format(s=s))
    print("--------------------------------------------------")

def test_with_improvements_3(average_frames=5):
    """
    Project vicon points into the realsense pixels.

    Improvements:
       - Removing noisy depth points.
       - Averaging depth values over space (best kernel is 31).
       - Calculating Kabsch and finding error for each point, removing points with high error rates and re-calculating
         Kabsch.
       - Averaging the vicon points over several frames.
    """
    # Some consts.
    bag_path = '../../data/Sub007_Left_Front.bag'
    annotated_pixels_path = '../../annotations_data/Sub007/Front/annotations_all.json'
    vicon_csv_path = '../../annotations_data/Sub007/Front/vicon_points.csv'
    rgb_frame_path = '../../annotations_data/Sub007/Front/rgb_frame.png'

    print("Finding Kabsch with following improvements: Removing noisy depth points, averaging depth values over space,"
          " removing points with high error rates and re-calulcating Kabsch, averaging vicon points...")
    # Get points.

    # Get annotated points names.
    reader = CVATReader(annotated_pixels_path)
    annotated_points_names = reader.get_points().keys()

    # 'clean_points_names' are the names of the remaining points after removing points with noisy depth values.
    points_deprojected, clean_points_names = read_points_from_deprojection(bag_file_path=bag_path,
                                                                           annotated_pixels_path=annotated_pixels_path,
                                                                           kernel_size=31, # Best kernel size is 31
                                                                           remove_noisy_depth_points=True)
    vicon_points = read_vicon_points(vicon_path=vicon_csv_path, points_names=clean_points_names, frames_to_average=average_frames)

    # --------------------------------------------- Without scaling ---------------------------------------------------
    # Calculate transformation with Kabsch.
    s, R, t = find_rotation_matrix(realsense_3d_points=points_deprojected, vicon_3d_points=vicon_points, scale=False)

    # Find the error of Kabsch transformation - Euclidean distance.
    distances, average_dist = calc_euclidean_dist(vicon_csv_path=vicon_csv_path, points_deprojected=points_deprojected,
                                          clean_points_names=clean_points_names, scale=s, rotation_matrix=R,
                                          translate_vector=t)

    # Find 3 points with biggest distance, remove them and re-run Kabsch.
    sorted_points = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
    p1_name = sorted_points.popitem()[0]
    p2_name = sorted_points.popitem()[0]
    p3_name = sorted_points.popitem()[0]

    # Start all over again with new points...
    points_deprojected, clean_points_names = read_points_from_deprojection(bag_file_path=bag_path,
                                                                           annotated_pixels_path=annotated_pixels_path,
                                                                           kernel_size=31, # Best kernel size is 31
                                                                           remove_noisy_depth_points=True,
                                                                           points_to_remove=[p1_name, p2_name, p3_name])
    vicon_points = read_vicon_points(vicon_path=vicon_csv_path, points_names=clean_points_names, frames_to_average=average_frames)

    # Calculate transformation with Kabsch.
    s, R, t = find_rotation_matrix(realsense_3d_points=points_deprojected, vicon_3d_points=vicon_points, scale=False)

    # Project the points.
    vicon_reader = VICONReader(vicon_csv_path)
    all_vicon_points = vicon_reader.get_points()
    all_vicon_points = list(all_vicon_points.items())[0][1]  # Get first frame's points

    projected = project_rs(vicon_3d_points=all_vicon_points, scale=s, rotation_matrix=R, translation_vector=t,
                           bag_file_path=bag_path)

    # Draw points
    image = cv2.imread(rgb_frame_path)
    points_indices = [i for i, keypoint in enumerate(KEYPOINTS_NAMES) if keypoint in annotated_points_names]

    for i, row in enumerate(projected):
        x = int(row[0])
        y = int(row[1])

        if i not in points_indices:
            continue

        if math.isnan(x) or math.isnan(y):
            continue

        image = cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=5)

    cv2.imwrite("projected_without_scale_with_improvements_3.png", image)

    # Find the error of the projection - RMSE
    error = calc_rmse_projected_points(clean_points_names, annotated_pixels_path, projected)
    print("Without scaling RMSE: " + str(error))

    # Find the error of Kabsch transformation - Euclidean distance.
    distances, average_dist = calc_euclidean_dist(vicon_csv_path=vicon_csv_path, points_deprojected=points_deprojected,
                                                  clean_points_names=clean_points_names, scale=s, rotation_matrix=R,
                                                  translate_vector=t)
    print("Without scaling average euclidean distance: " + str(average_dist))

    # ---------------------------------------------- With scaling -----------------------------------------------------
    # Get points.
    # 'clean_points_names' are the names of the remaining points after removing points with noisy depth values.
    points_deprojected, clean_points_names = read_points_from_deprojection(bag_file_path=bag_path,
                                                                           annotated_pixels_path=annotated_pixels_path,
                                                                           kernel_size=31,  # Best kernel size is 31
                                                                           remove_noisy_depth_points=True)
    vicon_points = read_vicon_points(vicon_path=vicon_csv_path, points_names=clean_points_names, frames_to_average=average_frames)

    # Calculate transformation with Kabsch.
    s, R, t = find_rotation_matrix(realsense_3d_points=points_deprojected, vicon_3d_points=vicon_points, scale=True)

    # Find the error of Kabsch transformation - Euclidean distance.
    distances, average_dist = calc_euclidean_dist(vicon_csv_path=vicon_csv_path, points_deprojected=points_deprojected,
                                                  clean_points_names=clean_points_names, scale=s, rotation_matrix=R,
                                                  translate_vector=t)

    # Find 3 points with biggest distance, remove them and re-run Kabsch.
    sorted_points = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
    p1_name = sorted_points.popitem()[0]
    p2_name = sorted_points.popitem()[0]
    p3_name = sorted_points.popitem()[0]

    # Start all over again with new points...
    points_deprojected, clean_points_names = read_points_from_deprojection(bag_file_path=bag_path,
                                                                           annotated_pixels_path=annotated_pixels_path,
                                                                           kernel_size=31,  # Best kernel size is 31
                                                                           remove_noisy_depth_points=True,
                                                                           points_to_remove=[p1_name, p2_name, p3_name])
    vicon_points = read_vicon_points(vicon_path=vicon_csv_path, points_names=clean_points_names, frames_to_average=average_frames)

    # Calculate transformation with Kabsch.
    s, R, t = find_rotation_matrix(realsense_3d_points=points_deprojected, vicon_3d_points=vicon_points, scale=True)

    # Project the points.
    vicon_reader = VICONReader(vicon_csv_path)
    all_vicon_points = vicon_reader.get_points()
    all_vicon_points = list(all_vicon_points.items())[0][1]  # Get first frame's points

    projected = project_rs(vicon_3d_points=all_vicon_points, scale=s, rotation_matrix=R, translation_vector=t,
                           bag_file_path=bag_path)

    # Draw points
    image = cv2.imread(rgb_frame_path)
    points_indices = [i for i, keypoint in enumerate(KEYPOINTS_NAMES) if keypoint in annotated_points_names]

    for i, row in enumerate(projected):
        x = int(row[0])
        y = int(row[1])

        if i not in points_indices:
            continue

        if math.isnan(x) or math.isnan(y):
            continue

        image = cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=5)

    cv2.imwrite("projected_with_scale_with_improvements_3.png", image)

    # Find the error of the projection - RMSE
    error = calc_rmse_projected_points(clean_points_names, annotated_pixels_path, projected)
    print("With scaling RMSE: " + str(error))

    # Find the error of Kabsch transformation - Euclidean distance.
    distances, average_dist = calc_euclidean_dist(vicon_csv_path=vicon_csv_path, points_deprojected=points_deprojected,
                                                  clean_points_names=clean_points_names, scale=s, rotation_matrix=R,
                                                  translate_vector=t)
    print("With scaling average euclidean distance: " + str(average_dist))
    print("Scale is {s}".format(s=s))
    print("--------------------------------------------------")


if __name__ == "__main__":
    test_no_improvements()
    test_with_improvements_1()
    test_with_improvements_2()
    test_with_improvements_3(10)













