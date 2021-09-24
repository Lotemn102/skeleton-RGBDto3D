import math
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2

from data_cleaning.kabsch import kabsch
from data_cleaning.realsense_data_reader import RealSenseReader
from data_cleaning.cvat_data_reader import CVATReader
from data_cleaning.vicon_data_reader import VICONReader, KEYPOINTS_NAMES

'''
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

def read_points_from_deprojection(bag_file_path, annotated_pixels_path):
    # Read annotated pixels
    # Load 2d points.
    cvat_reader = CVATReader(annotated_pixels_path)
    data_2d_dict = cvat_reader.get_points()

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
        kernel_size = 21
        for i in range(-1 * int(kernel_size / 2), int(kernel_size / 2) + 1):
            for j in range(-1 * int(kernel_size / 2), int(kernel_size / 2) + 1):
                x = int(np.round(coordinate[0])) + i
                y = int(np.round(coordinate[1])) + j
                depth_pixel = [x, y]
                depth_value = aligned_depth_image[depth_pixel[0]][depth_pixel[1]]

                if depth_value > clipping_distance:
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

    # Rotate points. The axes system of the vicon points and realsense 3d points are different. I've decided to rotate
    # them all according to the the open3d default axes system.
    rotated_points = rotate_vicon_points_90_degrees_counterclockwise(rotation_axis='x', points=points_3d)
    rotated_points = rotate_vicon_points_90_degrees_counterclockwise(rotation_axis='x', points=rotated_points)
    points_3d = rotated_points

    # Init matrix with the data. Order of points is according to KEYPOINTS_NAMES.
    A = np.zeros((len(points_3d), 3))

    for i, key in enumerate(list(points_3d.keys())):
        keypoint = points_3d[key]
        x = keypoint[0]
        y = keypoint[1]
        z = keypoint[1]
        # Convert from meters to mm, since the vicon points are in mm!
        A[i][0] = x * 1000
        A[i][1] = y * 1000
        A[i][2] = z * 1000

    #  -------------------------------------- FOR DEBUGGING: Show points -----------------------------------------------
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

def read_vicon_points(vicon_path, points_names):
    # Load 3d points.
    reader_3d = VICONReader(vicon_path) # Vicon points are in mm!
    points = reader_3d.get_points()
    points = list(points.items())[0][1] # Get first frame's points

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
    data_3d_dict = rotated_points

    B = np.zeros((len(data_3d_dict), 3))

    for i, key in enumerate(data_3d_dict):
        keypoint = data_3d_dict[key]
        B[i][0] = keypoint[0]
        B[i][1] = keypoint[1]
        B[i][2] = keypoint[2]

     #  -------------------------------------- FOR DEBUGGING: Show points -----------------------------------------------
    pcd = o3d.geometry.PointCloud()
    points = np.array([(point[0], point[1], point[2]) for point in data_3d_dict.values()])
    pcd.points = o3d.utility.Vector3dVector(points)
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name='vicon 3d')
    visualizer.add_geometry(pcd)
    visualizer.run()
    visualizer.close()
    # ------------------------------------------------------------------------------------------------------------------

    B = np.asmatrix(B)
    return B

def find_rotation_matrix(realsense_3d_points, vicon_3d_points):
    s, ret_R, ret_t = kabsch(A=realsense_3d_points, B=vicon_3d_points, scale=False)
    return s, ret_R, ret_t

def project(vicon_3d_points, scale, rotation_matrix, translation_vector):
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

def project_rs(vicon_3d_points, scale, rotation_matrix, translation_vector, bag_file_path):
    # Vicon 3d -> realsense 3d
    # Rotate points. The axes system of the vicon points and realsense 3d points are different. I've decided to rotate
    # them all according to the the open3d default axes system.
    dummy_dict = {}

    for i, p in enumerate(vicon_3d_points):
        dummy_dict[i] = p

    rotated_points = rotate_vicon_points_90_degrees_counterclockwise(rotation_axis='x', points=dummy_dict)
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
    pcd = o3d.geometry.PointCloud()
    points = np.array([(point[0], point[1], point[2]) for point in target_matrix])
    pcd.points = o3d.utility.Vector3dVector(points)
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name='vicon 3d -> realsense 3d')
    visualizer.add_geometry(pcd)
    visualizer.run()
    visualizer.close()
    # ------------------------------------------------------------------------------------------------------------------

    # Realsense 3d -> pixels
    depth_intrin = get_depth_intrinsics(bag_file_path=bag_file_path, first_frame=5842)
    pixels = []

    for i, row in enumerate(target_matrix):
        point = np.array(row)
        pixel = rs.rs2_project_point_to_pixel(depth_intrin, point)
        pixels.append(pixel)

    return pixels

def get_depth_intrinsics(bag_file_path, first_frame):
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
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

    return depth_intrin

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

def test():
    # Some consts.
    bag_path = '../../data/Sub007_Left_Front.bag'
    annotated_pixels_path = '../../annotations_data/Sub007/Front/annotations_all.json'
    vicon_csv_path = '../../annotations_data/Sub007/Front/vicon_points.csv'
    rgb_frame_path = '../../annotations_data/Sub007/Front/rgb_frame.png'

    # Get points.
    points_deprojected, points_names = read_points_from_deprojection(bag_file_path=bag_path,
                                                                     annotated_pixels_path=annotated_pixels_path)
    vicon_points = read_vicon_points(vicon_path=vicon_csv_path, points_names=points_names)

    # Calculate transformation with Kabsch.
    s, R, t = find_rotation_matrix(realsense_3d_points=points_deprojected, vicon_3d_points=vicon_points)

    # Project the points.
    vicon_reader = VICONReader(vicon_csv_path)
    all_vicon_points = vicon_reader.get_points()
    all_vicon_points = list(all_vicon_points.items())[0][1] # Get first frame's points
    projected = project_rs(vicon_3d_points=all_vicon_points, scale=s, rotation_matrix=R, translation_vector=t,
                           bag_file_path=bag_path)

    # Draw points
    image = cv2.imread(rgb_frame_path)

    for i, row in enumerate(projected):
        x = row[0]
        y = row[1]

        if math.isnan(x) or math.isnan(y):
            continue

        x = int(int(x) / 10) + 200
        y = int(int(y) / 10) + 200

        image = cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=5)

    cv2.imshow("Projected", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    test()












