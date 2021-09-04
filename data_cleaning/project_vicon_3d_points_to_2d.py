"""
We have decided to first try to train OpenPose with our data, but use 2d points instead of 3d points as labels.
This file contains scripts for projecting the 3d vicon points in the realsense rgb image pixels.
"""
import copy
import math
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
from typing import List

from data_cleaning.trim_data import rotate_vicon_points_90_degrees_counterclockwise

POSITION = 'Front'
BAG_PATH = '../../data/Sub007_Left_' + POSITION + '.bag'
CSV_PATH = '../../data/Sub007/Sub007/Left/' + POSITION + '/Sub007_Left_' + POSITION + '.csv'
RGB_IMAGES = '../../data/Sub007/Sub007/Left/' + POSITION + '/rgb_frames/'
DEPTH_IMAGES = '../../data/Sub007/Sub007/Left/' + POSITION + '/depth_frames/'

def world_coordinates_to_camera_coordinates(point: List, r1: float, r2: float,  r3: float, r4: float,  r5: float,
                                            r6: float, r7: float, r8: float, r9: float, t1: float, t2: float,
                                            t3: float) -> np.array:
    """
    Translate world coordinate to camera coordinate.
    Transformation matrix source: https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/OWENS/LECT9/node2.html

    :param point: List of 3 float.
    :return: Numpy array of size 4. The last element in the array is always 1.
    """
    point.append(1)
    point = np.array(point)

    transformation_matrix = np.zeros((4, 4))
    transformation_matrix[0][0] = r1
    transformation_matrix[0][1] = r2
    transformation_matrix[0][2] = r3
    transformation_matrix[1][0] = r4
    transformation_matrix[1][1] = r5
    transformation_matrix[1][2] = r6
    transformation_matrix[2][0] = r7
    transformation_matrix[2][1] = r8
    transformation_matrix[2][2] = r9
    transformation_matrix[0][3] = t1
    transformation_matrix[1][3] = t2
    transformation_matrix[2][3] = t3
    transformation_matrix[3][3] = 1

    xyz = np.dot(transformation_matrix, point)
    return xyz

def project_camera_coordinates_to_pixel_coordinates(point: np.ndarray, fx: float, fy: float, ppx: float, ppy: float,
                                                    shooting_angle: str) -> tuple:
    """
    Project camera coordinate to image pixel.
    Projection matrix source: https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/OWENS/LECT9/node2.html

    :param point: Numpy array of size 4. The last element in the array is 1.
    :param fx: x value of focal length.
    :param fy: y value of focal length.
    :param ppx: x value of projection center.
    :param ppy: y value of projection center.
    :param shooting_angle: "Back", "Front" or "Side".
    :return: Tuple of 2 ints, representing the pixel.
    """
    #if shooting_angle.lower() == 'back':
    #    fx = -fx  # Flip the image along x axis.

    projection_matrix = np.zeros((3, 4))
    projection_matrix[0][0] = fx
    projection_matrix[1][1] = fy
    projection_matrix[2][2] = 1
    projection_matrix[0][2] = ppx
    projection_matrix[1][2] = ppy

    #point.append(1)
    point = np.array(point)

    uvw = np.dot(projection_matrix, point)
    u = uvw[0] / point[2] # Scale by Z value
    v = uvw[1] / point[2] # Scale by Z value

    if math.isnan(u) or math.isnan(v):
        return -1, -1

    return int(np.round(u)), int(np.round(v))
'''
def deproject_2d_to_3d(pixel: List, fx: float, fy: float, ppx: float, ppy: float, scale: float) -> tuple:
    pixel.append(1)
    pixel = np.array(pixel)
    pixel = pixel * scale

    rotation_matrix = np.zeros((3, 4))
    rotation_matrix[0][0] = fx
    rotation_matrix[1][1] = fy
    rotation_matrix[2][2] = 1
    rotation_matrix[0][2] = ppx
    rotation_matrix[1][2] = ppy

    invered_rotation_matrix = np.linalg.pinv(rotation_matrix)
    xyz = np.dot(invered_rotation_matrix, pixel)
    return xyz[0], xyz[1], xyz[2]
'''

def save_calibration_parameters(bag_file_path: str):
    """
    Read the calibration parameters from the bag file, and save them to json file.

    :param bag_file_path: Path to the bag file.
    :return: None.
    """
    # Set the pipe.
    pipeline = rs.pipeline()
    config = rs.config()

    # Read the data.
    rs.config.enable_device_from_file(config, bag_file_path)
    config.enable_stream(stream_type=rs.stream.color, width=640, height=480, format=rs.format.rgb8, framerate=30)
    config.enable_stream(stream_type=rs.stream.depth, width=848, height=480, format=rs.format.z16, framerate=30)

    try:
        pipe_profile = pipeline.start(config)
    except:
        config.enable_stream(stream_type=rs.stream.color, width=640, height=480, format=rs.format.rgb8, framerate=15)
        config.enable_stream(stream_type=rs.stream.depth, width=640, height=480, format=rs.format.z16, framerate=15)
        pipe_profile = pipeline.start(config)

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    # Intrinsics & Extrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    color_to_depth_extrin = color_frame.profile.get_extrinsics_to(depth_frame.profile)

    # Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
    depth_sensor = pipe_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    fx, fy = color_intrin.fx, color_intrin.fy
    ppx, ppy = color_intrin.ppx, color_intrin.ppy
    rotation = color_to_depth_extrin.rotation
    translation = color_to_depth_extrin.translation

    splitted = bag_file_path.split('/')
    file_name = splitted[-1]
    file_name_splitted = file_name.split('_') # For example: 'Sub002_Left_Back'
    sub_name = file_name_splitted[0]
    sub_position = file_name_splitted[1]
    if sub_position == 'Tightstand':
        sub_position = 'Tight'
    sub_shooting_angle = file_name_splitted[2][:-4]

    # Read parameters.
    f = open('assets/realsense_calibration_parameters.json')
    data = json.load(f)

    # Iterate through all of the cameras, and check which one was used for this shooting.
    for camera in list(data.keys()):
        params = data[camera]

        if params['fx'] == fx and  params['fy'] == fy and  params['ppx'] == ppx and  params['ppy'] == ppy:
            camera_id = camera

    calib_parameters = {
        'camera_id' : int(camera_id),
        'fx' : fx,
        'fy' : fy,
        'ppx' : ppx,
        'ppy' : ppy,
        'depth_scale' : depth_scale,
        'r1' : rotation[0],
        'r2' : rotation[1],
        'r3' : rotation[2],
        'r4' : rotation[3],
        'r5' : rotation[4],
        'r6' : rotation[5],
        'r7' : rotation[6],
        'r8' : rotation[7],
        'r9' : rotation[8],
        't1' : translation[0],
        't2' : translation[1],
        't3' : translation[2]
    }

    saving_json_path = '../../data/' + sub_name + '/' + sub_name + '/' + sub_position + '/' + sub_shooting_angle +\
                       '/realsense_calibration_parameters.json'
    json_data = json.dumps(calib_parameters)

    try:
        f = open(saving_json_path, 'w')
    except FileNotFoundError:
        return

    f.write(json_data)
    f.close()

    return fx, fy, ppx, ppy, depth_scale

def aux_save_calibration_parameters():
    """
    Auxiliary function for saving calibration parameters from the bag files to json files.

    :return: None
    """
    for root, dirs, files in os.walk("H:/Movement Sense Research/Vicon Validation Study"):
        for file in files:
            if file.endswith(".bag"):

                if 'Sub001' in file:
                    continue

                if 'Extra' in file or 'Extra' in dirs or 'Extra' in root or 'original' in root:
                    continue

                if 'NOT' in file:
                    continue

                remove_extension = file[:-4]
                if 'withoutlight' in remove_extension:
                    continue

                splitted = remove_extension.split('_')
                subject_name = [e for e in splitted if 'Sub' in e][0]
                subject_number = int(subject_name[3:])
                shooting_angle = [e for e in splitted if e in ['Front', 'Back', 'Side']][0]

                for e in splitted:

                    if 'squat' in e.lower():
                        subject_position = e
                        break
                    elif 'stand' in e.lower():
                        subject_position = e
                        break
                    elif 'left' in e.lower():
                        subject_position = e
                        break
                    elif 'right' in e.lower():
                        subject_position = e
                        break
                    elif 'tight' in e.lower():
                        subject_position = e
                        break

                if subject_number > 15:
                    continue

                print("Working on " + subject_name + ", " + subject_position + ", " + shooting_angle)
                save_calibration_parameters(root + "/" +  file)

def find_rotation_angle(rgb_image_path : str):
    # Read all image path.
    frames_file_names = []
    for root, dirs, files in os.walk(rgb_image_path):
        for file in files:
            if file.endswith(".png"):
                frames_file_names.append(file)

    # Read only first image. We only need 1 image in order to find the angle.
    file_name = frames_file_names[0]

    # Read the image.
    image = cv2.imread(rgb_image_path + file_name)

    # Use the depth mask.
    CLIPPING_DIST = 2.3 / 0.001

    # Use the correlated depth frame as mask.
    depth_map = np.fromfile(DEPTH_IMAGES + file_name[:-4] + ".raw", dtype='int16', sep="")
    depth_map = depth_map.reshape([640, 480])

    # Threshold all pixels that are far from clipping distance.
    depth_mask = np.where((depth_map > CLIPPING_DIST) | (depth_map <= 0), 0, 255)
    depth_mask = np.stack((depth_mask,) * 3, axis=-1)
    depth_mask = depth_mask.astype('uint8')
    kernel = np.ones((10, 10), 'uint8')
    dilated_mask = cv2.dilate(depth_mask, kernel, iterations=2)

    # Combine the image and the mask
    masked = cv2.bitwise_and(image, dilated_mask)
    masked[np.all(masked == (0, 0, 0), axis=-1)] = (255, 255, 255)

    # Convert image to grayscale.
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # Threshold the image.
    _, thresh = cv2.threshold(gray, 250, 1, cv2.THRESH_BINARY_INV) # 0 for the background, 1 for the person

    mat = np.argwhere(thresh != 0)

    # Let's swap here... (e. g. [[row, col], ...] to [[col, row], ...])
    mat[:, [0, 1]] = mat[:, [1, 0]]
    mat = np.array(mat).astype(np.float32)  # have to convert type for PCA

    # Find center and eigenvectors.
    mean, eigenvectors = cv2.PCACompute(mat, mean=np.array([]))

    #  ---------------------------------- FOR DEBUGGING: Draw eigenvectors --------------------------------------------
    # a, b = mean[0]
    # a, b = int(a), int(b)
    # center = (a, b)
    # endpoint1 = tuple(center + eigenvectors[0] * 100)
    # c, d = endpoint1
    # c, d = int(c), int(d)
    # endpoint1 = (c, d)
    # endpoint2 = tuple(center + eigenvectors[1] * 50)
    # eigenvectors, f = endpoint2
    # eigenvectors, f = int(eigenvectors), int(f)
    # endpoint2 = (eigenvectors, f)
    #
    # red_color = (0, 0, 255)
    # cv2.circle(img=image, center=center, radius=5, color=red_color)
    # cv2.line(image, center, endpoint1, red_color)
    # cv2.line(image, center, endpoint2, red_color)
    # cv2.imshow("eigenvectors", image)
    # cv2.waitKey(0)

    # Calculate rotation angle.
    a, b = mean[0]
    a, b = int(a), int(b)
    center_pixel = (a, b)
    bottom_pixel = (a, image.shape[0])
    vector_1 = (center_pixel[0] - bottom_pixel[0], center_pixel[1] - bottom_pixel[1])

    end_pixel = tuple(center_pixel + eigenvectors[0] * 100)
    c, d = end_pixel
    c, d = int(c), int(d)
    end_pixel = (c, d)
    vector_2 = (end_pixel[0] - center_pixel[0], end_pixel[1] - center_pixel[1])

    angle = np.arccos((np.dot(vector_1, vector_2)) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))) # In radians.

    #  ---------------------------------- FOR DEBUGGING: Draw vectors --------------------------------------------
    # red_color = (0, 0, 255)
    # blue_color = (255, 0, 0)
    # green_color = (0, 255, 0)
    # cv2.circle(img=image, center=center_pixel, radius=5, color=green_color)
    # cv2.line(image, center_pixel, end_pixel, red_color)
    # cv2.line(image, center_pixel, bottom_pixel, blue_color)
    # cv2.imshow("vectors", image)
    # cv2.waitKey(0)

    angle = 180 - np.degrees(angle) # We need the complementary angle.
    angle = angle - 10 # This is manual fix, might require re-adjustment...

    return angle, center_pixel

def rotate_pixel(pixel, center_pixel, angle):
    angle = np.radians(angle)
    angle = -angle
    s = np.sin(angle)
    c = np.cos(angle)

    # Move pixel to origin.
    origin_x = pixel[0] - center_pixel[0]
    origin_y = pixel[1] - center_pixel[1]

    # Rotate.
    x_new = int(np.round(origin_x * c - origin_y * s))
    y_new = int(np.round(origin_x * s + origin_y * c))

    # Move the pixel back.
    x = x_new + center_pixel[0]
    y = y_new + center_pixel[1]

    return x, y

def rotate_image(image, angle):
    angle = -angle
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def realsense_project_camera_coordinates_to_pixel_coordinates(point: List, bag_file_path):
    """
    This is realsense api implementation for projecting camera coordinate to image pixel. I used it for testing my
    implementation.

    :param point: List of 3 floats.
    :param bag_file_path: Path to the bag file of the current session.
    :return: Camera pixel.
    """
    # Set the pipe.
    pipeline = rs.pipeline()
    config = rs.config()

    # Read the data.
    rs.config.enable_device_from_file(config, bag_file_path)
    config.enable_stream(stream_type=rs.stream.color, width=640, height=480, format=rs.format.rgb8, framerate=30)
    config.enable_stream(stream_type=rs.stream.depth, width=848, height=480, format=rs.format.z16, framerate=30)

    pipe_profile = pipeline.start(config)
    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    color_to_depth_extrin = color_frame.profile.get_extrinsics_to(depth_frame.profile)

    res = rs.rs2_project_point_to_pixel(color_intrin, point)
    return res

def test():
    # Read all image path.
    frames_file_names = []
    for root, dirs, files in os.walk(RGB_IMAGES):
        for file in files:
            if file.endswith(".png"):
                frames_file_names.append(file)

    # Get calibration parameters.
    json_path = RGB_IMAGES[:-11] + 'realsense_calibration_parameters.json'
    f = open(json_path)
    data = json.load(f)

    fx, fy, ppx, ppy, depth_scale = data['fx'], data['fy'], data['ppx'], data['ppy'], data['depth_scale']
    r1, r2, r3, r4, r5, r6, r7, r8, r9 = data['r1'], data['r2'], data['r3'], data['r4'], data['r5'], data['r6'], \
                                         data['r7'], data['r8'], data['r9']
    t1, t2, t3 = data['t1'], data['t2'], data['t3']

    # Read points
    if POSITION == 'Front':
        points = rotate_vicon_points_90_degrees_counterclockwise(csv_path=CSV_PATH, rotation_axis='x')
        points = rotate_vicon_points_90_degrees_counterclockwise(points=points, rotation_axis='z')
        points = rotate_vicon_points_90_degrees_counterclockwise(points=points, rotation_axis='z')
        points = rotate_vicon_points_90_degrees_counterclockwise(points=points, rotation_axis='y')
        points = rotate_vicon_points_90_degrees_counterclockwise(points=points, rotation_axis='y')

    if POSITION == 'Back':
        # Rotate the points, since the 'default' of the vicon point is Front.
        points = rotate_vicon_points_90_degrees_counterclockwise(csv_path=CSV_PATH, rotation_axis='x')
        points = rotate_vicon_points_90_degrees_counterclockwise(points=points, rotation_axis='z')
        points = rotate_vicon_points_90_degrees_counterclockwise(points=points, rotation_axis='z')

    if POSITION == 'Side':
        # Rotate the points, since the 'default' of the vicon point is Front.
        points = rotate_vicon_points_90_degrees_counterclockwise(csv_path=CSV_PATH, rotation_axis='x')
        points = rotate_vicon_points_90_degrees_counterclockwise(points=points, rotation_axis='z')
        points = rotate_vicon_points_90_degrees_counterclockwise(points=points, rotation_axis='z')
        points = rotate_vicon_points_90_degrees_counterclockwise(points=points, rotation_axis='y')
        points = rotate_vicon_points_90_degrees_counterclockwise(points=points, rotation_axis='y')
        points = rotate_vicon_points_90_degrees_counterclockwise(points=points, rotation_axis='y')

    # Find rotation angle
    angle, center_pixel = find_rotation_angle(RGB_IMAGES)

    for i, frame in enumerate(points.keys()):
        image = cv2.imread(RGB_IMAGES + frames_file_names[i])
        #blank = np.zeros(shape=(1080, 1024, 3), dtype=np.uint8)
        #image = cv2.cvtColor(blank, cv2.COLOR_RGB2BGR)

        first_frame_points = points[frame]
        for p in first_frame_points:
            if POSITION == 'Back':
                # Back 007:
                c = 2210
                b = 1200
                a = -190
            if POSITION == 'Front':
                # Front 007:
                c = 1950
                b = 1220
                a = -570
            if POSITION == 'Side':
                # Side 007:
                c = 1850
                b = 1100
                a = -150

            point = [p.x + a, p.y + b, p.z + c]
            xyz = world_coordinates_to_camera_coordinates(point=point, r1=r1, r2=r2, r3=r3, r4=r4, r5=r5, r6=r6, r7=r7,
                                                          r8=r8, r9=r9, t1=t1, t2=t2, t3=t3)
            u, v = project_camera_coordinates_to_pixel_coordinates(xyz, fx, fy, ppx, ppy, shooting_angle=POSITION)
            if u == -1 or v == -1:
                continue

            if angle < 0.1: # The angle finding script is not perfect, due to noise in the depth images. So this is a
                # for corner case, where the angle rotation adds more noise...
                x, y = u, v
            else:
                x, y = rotate_pixel(pixel=(u, v), center_pixel=center_pixel, angle=angle)
            image = cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=5)

        if i > 100:
            cv2.imwrite("projection.png", image)
            #cv2.imshow("image with points", image)
            #cv2.waitKey(0)

test()

