"""
Script for visualizing the recordings after trimming them, to manually validate the trimming.
"""

import cv2
import numpy as np
import os
import json
import math

from data_cleaning.vicon_data_reader import VICONReader

SUBJECT_NUMBER = 3

def validate_equal_number_of_frames():
    """
    Make sure the all pair of realsense video and corresponding vicon recording has the same frame numbers.

    :return: None.
    """
    for i in range(SUBJECT_NUMBER, SUBJECT_NUMBER+1):
        subject_name = 'Sub00' + str(i) if i < 10 else 'Sub0' + str(i)
        for position in ['Stand', 'Squat', 'Tight', 'Left', 'Right']:

            if position != 'Squat':
                continue

            for angle in ['Front', 'Back', 'Side']:

                LOG_PATH = '/media/lotemn/Other/project-data/trimmed/' + subject_name + '/' + position + '/' + angle + '/log.json'

                f = open(LOG_PATH)
                data = json.load(f)
                n1 = data['Number of frames Realsense RGB']
                n2 = data['Number of frames depth video']
                n3 = data['Number of frames Vicon']

                if n1 != n2 or n2 != n3:
                    print("Mismatch in " + subject_name + ', ' + position + ', ' + angle)
                f.close()

    print("Finished checking number of frames.")

def visualize_all():
    for i in range(SUBJECT_NUMBER, SUBJECT_NUMBER+1):
        subject_name = 'Sub00' + str(i) if i < 10 else 'Sub0' + str(i)

        for position in ['Stand', 'Squat', 'Tight', 'Left', 'Right']:

            if position != 'Squat':
                continue

            for angle in ['Front', 'Back', 'Side']:

                print(subject_name + ", " + position + ", " + angle)

                RGB_PATH = '/media/lotemn/Other/project-data/trimmed/' + subject_name + '/' + position + '/' + angle + '/rgb_frames/'
                DEPTH_PATH = '/media/lotemn/Other/project-data/trimmed/' + subject_name + '/' + position + '/' + angle + '/depth_frames/'
                CSV_PATH = '/media/lotemn/Other/project-data/trimmed/' + subject_name + '/' + position + '/' + angle + '/' \
                          + subject_name + '_' + position + '_' + angle + '.csv'

                log_file_path = '/media/lotemn/Other/project-data/trimmed/' + subject_name + '/' + position + '/' + angle + '/log.json'
                f = open(log_file_path)
                data = json.load(f)
                REALSENSE_FPS = data['RGB FPS']

                if data['RGB FPS'] != data['Depth FPS']:
                    print("FPS mismatch.")

                # Init the VICON reader and read the points.
                vicon_reader = VICONReader(vicon_file_path=CSV_PATH)
                vicon_points = vicon_reader.get_points()  # Dictionary of <frame_id, List<Point>>
                index = 0

                # Set-up two windows.
                cv2.namedWindow("RGB", cv2.WINDOW_AUTOSIZE)
                cv2.namedWindow("VICON", cv2.WINDOW_AUTOSIZE)
                cv2.namedWindow("DEPTH", cv2.WINDOW_AUTOSIZE)
                cv2.moveWindow("RGB", 0, 0, )
                cv2.moveWindow("VICON", 1100, 0, )
                cv2.moveWindow("DEPTH", 600, 0, )

                counter = 0

                # All rgb frames
                all_frames_files_realsense_rgb = os.listdir(RGB_PATH)

                if 'log.json' in all_frames_files_realsense_rgb:
                    all_frames_files_realsense_rgb.remove('log.json')

                all_frames_files_realsense_rgb = sorted(all_frames_files_realsense_rgb, key=lambda x: int(x[:-4]))

                # All depth frames
                all_frames_files_realsense_depth = os.listdir(DEPTH_PATH)

                if 'log.json' in all_frames_files_realsense_depth:
                    all_frames_files_realsense_depth.remove('log.json')

                all_frames_files_realsense_depth = sorted(all_frames_files_realsense_depth, key=lambda x: int(x[:-4]))

                while 1:

                    if index >= len(list(vicon_points.keys())):
                        break

                    # Create an empty image to write the vicon points on in later.
                    current_frame = list(vicon_points.keys())[index]
                    blank = np.zeros(shape=(640, 480, 3), dtype=np.uint8)
                    vicon_image = cv2.cvtColor(blank, cv2.COLOR_RGB2BGR)
                    current_frame_points = vicon_points[current_frame]

                    for i, point in enumerate(current_frame_points):
                        x = point.x
                        y = point.y
                        z = point.z

                        if math.isnan(x) or math.isnan(y) or math.isnan(z):
                            # Skip this point for the moment
                            continue

                        # Scale the coordinates so they will fit the image.
                        x = x / 5
                        z = z / 5
                        # Draw the point on the blank image (orthographic projection).
                        vicon_image = cv2.circle(vicon_image, ((int(x) + 170), (int(z) + 120)), radius=0,
                                                 color=(0, 0, 255),
                                                 thickness=10)  # Coordinates offsets are manually selected to center the object.

                    vicon_image = cv2.rotate(vicon_image, cv2.cv2.ROTATE_180)  # OpenCV origin is TOP-LEFT, so image
                    rgb_image = cv2.imread(RGB_PATH + '/' + all_frames_files_realsense_rgb[index])
                    depth_image = np.fromfile(DEPTH_PATH + '/' + all_frames_files_realsense_depth[index], dtype='int16',
                                              sep="")
                    depth_image = depth_image.reshape([640, 480])
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),
                                                       cv2.COLORMAP_JET)

                    cv2.imshow('RGB', rgb_image)
                    cv2.imshow('DEPTH', depth_colormap)
                    cv2.imshow('VICON', vicon_image)
                    cv2.waitKey(REALSENSE_FPS)
                    counter = counter + 1
                    index += 1

if __name__ == "__main__":
    validate_equal_number_of_frames()
    visualize_all()