"""
List of function for manually synchronizing RealSense videos and Vicon points.

Pipeline
=======
    1. Extract all frames of RealSense video as images using 'generate_realsense_frames()'.
    2. Extract all frames of Vicon points as images (orthographic projected) using 'generate_vicon_frames()'.
    3. MANUALLY find the first frame in the Vicon frames, and it's correlated frame in RealSense data. Save the
        data in 'assets/frames_sync.json'.
    4. Cut the videos by calling 'generate_synchronized_videos_for_all()'.
"""

import cv2
import numpy as np
import os
import math
import json
import pyrealsense2 as rs

from realsense_data_reader import RealSenseReader
from vicon_data_reader import VICONReader

# ---------------------- Generate frames for manually detecting T-pose -------------------------------------------------
"""
For manually detecting the T-pose frames.
"""
def generate_realsense_frames_rgb_and_depth(bag_path: str, bag_shoot_angle: str, sub_name: str, sub_position: str):
    try:
        REALSENSE_FPS = 30
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, bag_path)
        config.enable_stream(stream_type=rs.stream.color, width=640, height=480, format=rs.format.rgb8, framerate=REALSENSE_FPS)
        config.enable_stream(stream_type=rs.stream.depth, width=848, height=480, format=rs.format.z16, framerate=REALSENSE_FPS)
        pipeline.start(config)
    except:
        REALSENSE_FPS = 15
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, bag_path)
        config.enable_stream(stream_type=rs.stream.color, width=640, height=480, format=rs.format.rgb8,
                             framerate=REALSENSE_FPS)
        config.enable_stream(stream_type=rs.stream.depth, width=640, height=480, format=rs.format.z16,
                             framerate=REALSENSE_FPS)
        pipeline.start(config)

    # Create save path.
    save_path_rgb = '/media/lotemn/Other/project-data/frames/' + sub_name + '/RealSense/' + sub_position + '/' + bag_shoot_angle + '/'
    save_path_depth = '/media/lotemn/Other/project-data/frames/' + sub_name + '/RealSenseDepth/' + sub_position + '/' + bag_shoot_angle + '/'

    # Create align object to align depth frames to RGB frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    for i in range(10):
        frames = pipeline.wait_for_frames() # Not sure why, but the reading only 1-2 frames causes infinite loop in
        # following lines.

    first_frame = frames.frame_number
    current_frame = 0

    if not os.path.isdir(save_path_rgb):
        os.makedirs(save_path_rgb)

    if not os.path.isdir(save_path_depth):
        os.makedirs(save_path_depth)

    if sub_position == 'Tightstand':
        sub_position = 'Tight'

    while current_frame != first_frame:
        # Get frameset.
        frames = pipeline.wait_for_frames()
        current_frame = frames.frame_number

        # Align depth frames to the color ones. The RealSense has 2 sensors: RGB & depth. They are not 100% aligned
        # by default.
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data(), dtype=np.uint16)
        aligned_depth_image = np.rot90(aligned_depth_image, k=3)

        # Get color image
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        color_image = np.rot90(color_image,
                               k=3)  # OpenCV origin is TOP-LEFT, so we to rotate the image 180 degrees.

        # Save color image.
        cv2.imwrite(save_path_rgb + "/" + str(frames.frame_number) + '.png', color_image)

        # Save depth image.
        file_path = save_path_depth + "/" + str(frames.frame_number) + '.raw'
        aligned_depth_image.astype('int16').tofile(file_path)

        # -------------------------------------------------- FOR DEBUGGING ---------------------------------------------
        '''
        cv2.imshow("rgb", color_image)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_image, alpha=0.03),
                                           cv2.COLORMAP_JET)
        cv2.imshow("depth", depth_colormap)
        cv2.waitKey(1)
        '''

        # --------------------------------------------------------------------------------------------------------------

    # Read all frames to find first one, last one and number of frames.
    all_frames = os.listdir(save_path_rgb)

    if 'log.json' in all_frames:
        all_frames.remove('log.json')

    all_frames = sorted(all_frames, key=lambda x: int(x[:-4]))
    first_frame = all_frames[0][:-4]
    last_frame = all_frames[-1][:-4]
    number_of_frames = len(all_frames)

    # Write metadata to json file
    metadata = {
        'sub_name': sub_name,
        'sub_position': sub_position,
        'shooting_angle': bag_shoot_angle,
        'first_frame': int(first_frame),
        'last_frame': int(last_frame),
        'total_frames_number': number_of_frames,
        'width': 640,
        'height': 480,
        'FPS': REALSENSE_FPS
    }

    json_data = json.dumps(metadata)
    json_path_rgb = save_path_rgb + 'log.json'
    json_path_depth = save_path_depth + 'log.json'
    json_file_rgb = open(json_path_rgb, "w")
    json_file_depth = open(json_path_depth, "w")
    json_file_rgb.write(json_data)
    json_file_depth.write(json_data)
    json_file_rgb.close()
    json_file_depth.close()

def generate_vicon_frames(csv_path: str):
    print(csv_path)

    vicon_reader = VICONReader(vicon_file_path=csv_path)
    vicon_points = vicon_reader.get_points()  # Dictionary of <frame_id, List<Point>>

    print('Number of vicon points: ' + str(len(vicon_points)))
    print('First vicon frame: ' + str(list(vicon_points.keys())[0]))
    print('Last vicon frame: ' + str(list(vicon_points.keys())[-1]))

    # Create save path.
    csv_file_name = csv_path.split("/")[-1]
    file_name = csv_file_name[:-4]
    splitted = file_name.split("_")
    if len(splitted) == 1:
        splitted = file_name.split(" ")
    sub_name = splitted[0]
    sub_position = splitted[1]

    if sub_position == 'Standwithlight01':
        sub_position = 'Stand'

    save_path = '/media/lotemn/Other/project-data/frames/' + sub_name + "/Vicon/" + sub_position

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for frame in list(vicon_points.keys()):
        # Get 39 Vicon points.
        current_frame_points = vicon_points[frame]

        # Create an empty image to write the vicon points on in later.
        blank = np.zeros(shape=(640, 480, 3), dtype=np.uint8)
        vicon_image = cv2.cvtColor(blank, cv2.COLOR_RGB2BGR)

        for i, point in enumerate(current_frame_points):
            x = point.x
            y = point.y
            z = point.z

            if math.isnan(x) or math.isnan(y) or math.isnan(z):
                # Skip this point for the moment
                continue

            # Scale the coordinates so they will fit the image.
            x = x / 4.5
            z = z / 4.5
            # Draw the point on the blank image (orthographic projection).
            vicon_image = cv2.circle(vicon_image, ((int(z) + 50), (int(x) + 250)), radius=0, color=(0, 0, 255),
                                     thickness=10)  # Coordinates offsets are manually selected to center the object.

        # Rotate the image, since the vicon points are also rotated by default.
        vicon_image = cv2.rotate(vicon_image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Save image.
        cv2.imwrite(save_path + "/" + str(frame) + '.png', vicon_image)

def aux_generate_realsense_frames():
    """
    Read all bag files and generate their frames.

    :return: None.
    """

    for root, dirs, files in os.walk("/media/lotemn/Transcend/Movement Sense Research/Vicon Validation Study"):
        for file in files:
            if file.endswith(".bag"):

                if 'Sub014' not in file and 'Sub015' not in file:
                    continue

                if 'Extra' in file or 'Extra' in dirs or 'Extra' in root:
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
                generate_realsense_frames_rgb_and_depth(bag_path=root + "/" + file, sub_name=subject_name,
                                          bag_shoot_angle=shooting_angle, sub_position=subject_position)

def aux_generate_vicon_frames():
    for root, dirs, files in os.walk("/media/lotemn/Transcend/Movement Sense Research/Vicon Validation Study/"):
        for file in files:
            if file.endswith(".csv"):

                if 'Sub014' not in file:
                    continue

                if 'Tight' not in file:
                    continue

                if 'without' in file or 'Cal' in file:
                    continue

                remove_extension = file[:-4]
                splitted = remove_extension.split('_')

                if len(splitted) == 1:
                    splitted = remove_extension.split(' ')

                subject_name = [e for e in splitted if 'Sub' in e][0]
                subject_number = int(subject_name[3:])

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


                print("Working on " + subject_name + ", " + subject_position)
                generate_vicon_frames(csv_path=root + "/" + file)


if __name__ == "__main__":
    aux_generate_vicon_frames()

