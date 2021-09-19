import math
import unittest
import os
import cv2
import json
import numpy as np

from data_cleaning.cvat_data_reader import CVATReader

ANGLE = 'Front'
POSITION = 'Left'
PATH = '../../data/Sub007/Sub007/'
CALIBRATION_PATH = PATH + 'calibration_sub007_{angle}_test.json'.format(angle=ANGLE.lower())
RGB_FRAMES = PATH + '{pos}/{angle}/rgb_frames/'.format(angle=ANGLE.lower(), pos=POSITION)
VICON_PATH = PATH + '{pos}/{angle}/Sub007_{pos}_{angle2}.csv'.format(angle=ANGLE.lower(), angle2=ANGLE, pos=POSITION)
ANNOTATION_PATH = '../../annotations_data/Sub007/{angle}/annotations.json'.format(angle=ANGLE)


from data_cleaning.vicon_data_reader import VICONReader, KEYPOINTS_NAMES
from data_cleaning.project_vicon_3d_points_to_2d import project


class TestProjection(unittest.TestCase):
    def test_all_points(self):
        # Read calibration data
        f = open(CALIBRATION_PATH)
        data = json.load(f)

        scale = data['Scale']
        rotation = data['Rotation']
        translation = data['Translation']
        rotation = np.asmatrix(rotation)
        translation = np.asmatrix(translation)

        # Read vicon points
        reader = VICONReader(VICON_PATH)
        points = reader.get_points()

        # Read all frame numbers in the image folder path.
        frame_numbers = []
        for root, dirs, files in os.walk(RGB_FRAMES):
            for file in files:
                if file.endswith(".png"):
                    frame_numbers.append(int(file[:-4]))

        # Sort frames.
        sorted_frame_numbers = sorted(frame_numbers)

        for frame in sorted_frame_numbers:
            # Read image
            image = cv2.imread(RGB_FRAMES + str(frame) + '.png')

            # Read points
            frame_points = points[frame]

            projected = project(vicon_3d_points=frame_points, rotation_matrix=rotation, translation_vector=translation,
                                scale=scale)

            for row in projected:
                x = row.T[0]
                y = row.T[1]

                if math.isnan(x) or math.isnan(y):
                    continue

                x = int(x)
                y = int(y)

                image = cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=5)

            cv2.imshow("Projected", image)
            cv2.waitKey(200)

    def test_partial_points(self):
        print("ghghgghg")
        # Read calibration data
        f = open(CALIBRATION_PATH)
        data = json.load(f)

        scale = data['Scale']
        rotation = data['Rotation']
        translation = data['Translation']
        rotation = np.asmatrix(rotation)
        translation = np.asmatrix(translation)

        # Read annotation points.
        cvat_reader = CVATReader(ANNOTATION_PATH)
        data_2d_dict = cvat_reader.get_points()
        all_points_names_2d = data_2d_dict.keys()
        points_indices = [i for i, keypoint in enumerate(KEYPOINTS_NAMES) if keypoint in all_points_names_2d]

        # Read vicon points
        reader = VICONReader(VICON_PATH)
        points = reader.get_points()

        # Read all frame numbers in the image folder path.
        frame_numbers = []
        for root, dirs, files in os.walk(RGB_FRAMES):
            for file in files:
                if file.endswith(".png"):
                    frame_numbers.append(int(file[:-4]))

        # Sort frames.
        sorted_frame_numbers = sorted(frame_numbers)

        for frame in sorted_frame_numbers:
            # Read image
            image = cv2.imread(RGB_FRAMES + str(frame) + '.png')

            # Read points
            frame_points = points[frame]

            projected = project(vicon_3d_points=frame_points, rotation_matrix=rotation, translation_vector=translation,
                                scale=scale)

            for i, row in enumerate(projected):

                if i not in points_indices:
                    continue

                x = row.T[0]
                y = row.T[1]

                if math.isnan(x) or math.isnan(y):
                    continue

                x = int(x)
                y = int(y)

                image = cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=5)

            cv2.imshow("Projected", image)
            cv2.waitKey(0)







