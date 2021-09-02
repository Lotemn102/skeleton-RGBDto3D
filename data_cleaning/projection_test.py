import math

import pyrealsense2 as rs
import numpy as np
import cv2
import os

from typing import List

from vicon_data_reader import VICONReader
from data_cleaning.trim_data import rotate_vicon_points_90_degree_counterclockwise


BAG_PATH = '../../data/Sub007_Left_Front.bag'
CSV_PATH = '../../data/Sub007/Sub007/Left/Front/Sub007_Left_Front.csv'
RGB_IMAGES = '../../data/Sub007/Sub007/Left/Front/rgb_frames/'


def project_3d_to_2d(point: List, fx: float, fy: float, ppx: float, ppy: float, is_front: bool):
    if is_front:
        fx = -fx  # "Change" the location of the camera

    rotation_matrix = np.zeros((3, 4))
    rotation_matrix[0][0] = fx
    rotation_matrix[1][1] = fy
    rotation_matrix[2][2] = 1
    rotation_matrix[0][2] = ppx
    rotation_matrix[1][2] = ppy

    point.append(1)
    point = np.array(point)

    uvw = np.dot(rotation_matrix, point)
    u = uvw[0] / uvw[2]
    v = uvw[1] / uvw[2]

    if math.isnan(u) or math.isnan(v):
        return -1, -1

    return int(np.round(u)), int(np.round(v))

def deproject_2d_to_3d(pixel: List, fx: float, fy: float, ppx: float, ppy: float, scale: float):
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

def save_calibration_parameters(bag_file_path: str):
    # Set the pipe.
    pipeline = rs.pipeline()
    config = rs.config()

    # Read the data.
    rs.config.enable_device_from_file(config, BAG_PATH)
    config.enable_stream(stream_type=rs.stream.color, width=640, height=480, format=rs.format.rgb8, framerate=30)
    config.enable_stream(stream_type=rs.stream.depth, width=848, height=480, format=rs.format.z16, framerate=30)
    pipe_profile = pipeline.start(config)

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    # Intrinsics & Extrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

    # Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
    depth_sensor = pipe_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    fx, fy = color_intrin.fx, color_intrin.fy
    ppx, ppy = color_intrin.ppx, color_intrin.ppy

    return fx, fy, ppx, ppy, depth_scale

def test():
    # Read all image path.
    frames_file_names = []
    for root, dirs, files in os.walk(RGB_IMAGES):
        for file in files:
            if file.endswith(".png"):
                frames_file_names.append(file)

    # Get calibration parameters
    fx, fy, ppx, ppy, depth_scale = save_calibration_parameters(BAG_PATH)

    # Read points
    points = rotate_vicon_points_90_degree_counterclockwise(csv_path=CSV_PATH, rotation_axis='x')
    #reader = VICONReader(CSV_PATH)
    #points = reader.get_points()

    for i, frame in enumerate(points.keys()):
        image = cv2.imread(RGB_IMAGES + frames_file_names[i])
        image = cv2.rotate(image, cv2.ROTATE_180)

        first_frame_points = points[frame]
        for p in first_frame_points:
            c = 2350
            b = -700
            a = 10
            point = [p.x + a, p.y + b, p.z + c]
            u, v = project_3d_to_2d(point, fx, fy, ppx, ppy, is_front=True)

            if u == -1 or v == -1:
                continue

            image = cv2.circle(image, (u, v), radius=1, color=(0, 0, 255), thickness=5)

        image = cv2.rotate(image, cv2.ROTATE_180)
        cv2.imshow("image with points", image)
        cv2.waitKey(100)


test()
'''
x, y, z = deproject_2d_to_3d([131, 320], fx, fy, ppx, ppy, depth_scale)
p = [x, y, z]
u, v = project_3d_to_2d(p, fx, fy, ppx, ppy)
print(u, v)
'''

