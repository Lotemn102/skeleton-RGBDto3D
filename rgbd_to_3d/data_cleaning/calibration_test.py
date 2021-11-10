"""
Checking if calculating the Kabsch transformation using the calibration device improves projection accuracy.
"""
import numpy as np
import cv2
import pyrealsense2 as rs
import math
import open3d as o3d

from data_cleaning.realsense_data_reader import RealSenseReader
from data_cleaning.vicon_data_reader import VICONReader
from data_cleaning.cvat_data_reader import CVATReader
from data_cleaning.projection import project_rs, find_rotation_matrix

FOLDER_PATH = 'F:/Kimmel-Agmon/data_calibration_test/'
DEVICE_BAG_PATH = FOLDER_PATH + 'Board_Front.bag'
DEVICE_CSV_PATH = FOLDER_PATH + 'Calibration_Board.csv'
SUB_BAG_PATH = FOLDER_PATH + 'Lotem_Front.bag'
SUB_CSV_PATH = FOLDER_PATH + 'Lotem.csv'
DEVICE_ANNOTATIONS = FOLDER_PATH + 'device_annotations.json'
SUBJECT_ANNOTATIONS = FOLDER_PATH + 'lotem_annotations.json'
DEVICE_RGB_IMAGE = FOLDER_PATH + "device.png"
DEVICE_POINTS_NAMES = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '110','111', '112', '113', '114',
                       '115', '116', '117', '118', '119', '120', '121', '122', '123', '124']


def extract_frames():
    """
    Extract a single frame from the rgb video (front), and the a single frame from the Vicon, for both the device and
    the person.

    Note that the order of the points of the calibration device is in:
        'data_calibration_test/calibration_device_points_names.png'

    :return: None.
    """
    # ----------------------------  Device RGB ---------------------------------
    reader = RealSenseReader(DEVICE_BAG_PATH, type='RGB')
    pipeline, _ = reader.setup_pipeline()

    # Stream only first frame and save it.
    frames = pipeline.wait_for_frames()

    # Get color frame
    color_frame = frames.get_color_frame()

    # Convert color_frame to numpy array to render image in opencv
    color_image = np.asanyarray(color_frame.get_data())
    color_image = np.rot90(color_image, k=3)

    # Render image in opencv window
    cv2.imwrite(FOLDER_PATH + "device.png", color_image)

    # ------------------------------ Subject RGB ----------------------------------
    reader = RealSenseReader(SUB_BAG_PATH, type='RGB')
    pipeline, _ = reader.setup_pipeline()

    # Stream only first frame and save it.
    frames = pipeline.wait_for_frames()

    # Get color frame
    color_frame = frames.get_color_frame()

    # Convert color_frame to numpy array to render image in opencv
    color_image = np.asanyarray(color_frame.get_data())
    color_image = np.rot90(color_image, k=3)

    # Render image in opencv window
    cv2.imwrite(FOLDER_PATH + "subject.png", color_image)

    # ------------------------------ Subject CSV -----------------------------------
    # Omer started recording the RealSense video 10 seconds after the Vicon. So take frame number 1200 (FPS is 120).
    reader = VICONReader(SUB_CSV_PATH)
    points = reader.get_points()
    first_frame_subject = points[1200]

    # ------------------------------ Device CSV -------------------------------------

def calc_euclidean_dist(A, B):
    if type(A) == list:
        A = np.array(A)

    if type(B) == list:
        B = np.array(B)

    distance = np.linalg.norm(A - B)
    mean = np.mean(distance)
    return mean

def read_points_from_deprojection(bag_file_path, annotated_pixels_path, kernel_size=31, remove_noisy_depth_points=True):
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

    while frames.frame_number < 10:
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

    # ----------------------------------- FOR DEBUGGING: Create point cloud --------------------------------------------
    colorizer = rs.colorizer()
    colorized = colorizer.process(frames)
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
    pcd = o3d.io.read_point_cloud("pc.ply")
    points = np.asarray(pcd.points)

    # Visualize pointcloud
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
    # ------------------------------------------------------------------------------------------------------------------

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
                depth_value = aligned_depth_image[x][y]

                if remove_noisy_depth_points and depth_value > clipping_distance:
                    continue

                if depth_value < 1:
                    continue

                depth_value_in_meters = depth_value * depth_scale  # source: https://dev.intelrealsense.com/docs/rs-align-advanced
                # the function rs2_deproject_pixel_to_point expect to get the depth value in meters.
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth_value_in_meters)

                # -------------------------- FOR DEBUGGING: Check re-construction works --------------------------------
                pixel = rs.rs2_project_point_to_pixel(depth_intrin, depth_point)
                pixel[0], pixel[1] = int(np.round(pixel[0])), int(np.round(pixel[1]))
                assert pixel == [x, y]
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

    #  ----------------------- FOR DEBUGGING: Show points after reconstruction to 3D -----------------------------------
    pcd = o3d.geometry.PointCloud()
    points = np.array([(point[0], point[1], point[2]) for point in points_3d.values()])
    pcd.points = o3d.utility.Vector3dVector(points)
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name='pixels -> realsense 3d')
    visualizer.add_geometry(pcd)
    visualizer.run()
    visualizer.close()
    # ------------------------------------------------------------------------------------------------------------------

    A = np.asmatrix(A)
    points_names = points_3d.keys()
    return A, points_names

def test():
    # ---------------------------- Calculate Kabsch on the calibration device ------------------------------------------
    # 'clean_points_names' are the names of the remaining points after removing points with noisy depth values.
    points_deprojected, clean_points_names = read_points_from_deprojection(bag_file_path=DEVICE_BAG_PATH,
                                                                           annotated_pixels_path=DEVICE_ANNOTATIONS,
                                                                           kernel_size=1,
                                                                           remove_noisy_depth_points=False)
    device_vicon_reader = VICONReader(DEVICE_CSV_PATH, num_points=24)
    points = device_vicon_reader.get_points()
    device_vicon_points = list(points.values())[0]

    # ---------------------------------------------- FOR DEBUGGING -----------------------------------------------------
    #  ----------------------------------------- DRAW ALL VICON POINTS -------------------------------------------------
    pcd = o3d.geometry.PointCloud()
    points = np.array([(point.x, point.y, point.z) for point in device_vicon_points])
    pcd.points = o3d.utility.Vector3dVector(points)
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name='all vicon points')
    visualizer.add_geometry(pcd)
    visualizer.run()
    visualizer.close()
    # ------------------------------------------------------------------------------------------------------------------

    # Remove all points that were removed in the de-projection
    clean_points = []

    for p, name in zip(device_vicon_points, DEVICE_POINTS_NAMES):
        if name in clean_points_names:
            clean_points.append(p)

    # ---------------------------------------------- FOR DEBUGGING -----------------------------------------------------
    #  ---------------------------------- DRAW VICON POINTS THAT WERE ANNOTATED ----------------------------------------
    pcd = o3d.geometry.PointCloud()
    points = np.array([(point.x, point.y, point.z) for point in clean_points])
    pcd.points = o3d.utility.Vector3dVector(points)
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name='vicon points that were annotated')
    visualizer.add_geometry(pcd)
    visualizer.run()
    visualizer.close()
    # ------------------------------------------------------------------------------------------------------------------

    # Convert into matrix
    clean_mat = np.zeros((len(clean_points), 3))

    for i, p in enumerate(clean_points):
        clean_mat[i][0] = p.x
        clean_mat[i][1] = p.y
        clean_mat[i][2] = p.z

    clean_mat = np.matrix(clean_mat)

    # Calculate transformation with Kabsch.
    s, R, t = find_rotation_matrix(realsense_3d_points=points_deprojected, vicon_3d_points=clean_mat, scale=False)

    # Find projection error on the calibration device
    # Get annotated points names.
    reader = CVATReader(DEVICE_ANNOTATIONS, is_calibration=True)
    device_annotated_points = reader.get_points()
    projected = project_rs(vicon_3d_points=device_vicon_points, scale=s, rotation_matrix=R, translation_vector=t,
                           bag_file_path=DEVICE_BAG_PATH, first_frame=1)

    # Remove points that were not annotated
    projected_sub = []
    for p, name in zip(projected, DEVICE_POINTS_NAMES):
        if name in device_annotated_points.keys():
            projected_sub.append(p)

    average_dist = calc_euclidean_dist(projected_sub, list(device_annotated_points.values()))
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

    # cv2.imshow("device projection", image)
    # cv2.waitKey()
    cv2.imwrite("projected.png", image)

    # ----------------------------- Apply the projection matrix on the subject points ----------------------------------
    # # Read Vicon points
    # subject_vicon_reader = VICONReader(SUB_CSV_PATH)
    # points = subject_vicon_reader.get_points()
    # # Omer started recording the RealSense video 10 seconds after the Vicon. So take frame number 1200 (FPS is 120).
    # subject_vicon_points = points[1200]
    # projected = project_rs(vicon_3d_points=subject_vicon_points, scale=s, rotation_matrix=R, translation_vector=t,
    #                        bag_file_path=SUB_BAG_PATH)
    #
    # # Get the annotated points for finding the error



if __name__ == "__main__":
    test()