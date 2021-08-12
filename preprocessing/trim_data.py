"""
Trim the realsense video and vicon data, based on the synchronized frames manually found beforehand.
All missing frames (both on realsense and vicon) are removed.
"""

import json
import os
import cv2
import numpy as np
from typing import List, Dict
import math

from vicon_data_reader import VICONReader


def find_clean_frames_indices_on_realsense_and_vicon(bag_shoot_angle: str, sub_name: str, sub_position: str,
                                                     first_frame_number_realsense: int,
                                                     first_frame_number_vicon: int) -> (List[int], Dict):
    # -------------------------------------------- Find realsense clean frames -----------------------------------------
    # Get to the folder of all frames.
    folder_path_realsense = 'frames/' + sub_name + '/RealSense/' + sub_position + '/' + bag_shoot_angle + '/'

    # Starting from the given first_frame_number, create a new video.
    all_frames_files_realsense = os.listdir(folder_path_realsense)
    all_frames_files_realsense.remove('log.json')
    all_frames_files_realsense = sorted(all_frames_files_realsense, key=lambda x: int(x[:-4]))

    # Find first_frame_number index in the sorted list.
    first_frame_index_realsense = [i for i in range(len(all_frames_files_realsense)) if
                                   str(first_frame_number_realsense) in all_frames_files_realsense[i]][0]

    # Remove all frames before first_frame_number.
    trimmed_frames_files_realsense = all_frames_files_realsense[first_frame_index_realsense:]
    total_frames_number = len(trimmed_frames_files_realsense)

    frames_numbers_realsense = []  # Saving the frame numbers, so i would be able to calculate the difference between each 2 frames
    # and "skip" those frames in the vicon data.

    # Read all frames.
    for index, file in enumerate(trimmed_frames_files_realsense):
        current_frame_number = int(file[:-4])

        if index > total_frames_number:
            break

        frames_numbers_realsense.append(current_frame_number)

    differences_list_realsense = [j - i for i, j in zip(frames_numbers_realsense[:-1], frames_numbers_realsense[1:])]

    # ---------------------------------------------- Find vicon clean frames -------------------------------------------
    # Get to the folder of vicon csv file.
    csv_path = '/media/lotemn/Transcend/Movement Sense Research/Vicon Validation Study/' + sub_name + "/" + \
               sub_name + "_" + "Vicon/"  + sub_name + " " + sub_position + ".csv"

    vicon_reader = VICONReader(vicon_file_path=csv_path)
    vicon_points = vicon_reader.get_points()  # Dictionary of <frame_id, List<Point>>

    # Find first_frame_number index in the dict.
    all_frames_vicon = list(vicon_points.keys())
    first_frame_index_vicon = [i for i in range(len(all_frames_vicon)) if
                                   first_frame_number_vicon == all_frames_vicon[i]][0]

    # Remove all frames before first_frame_number.
    trimmed_frames_vicon = {k: vicon_points[k] for k in list(vicon_points.keys())[first_frame_index_vicon:]}

    # ONLY FOR NOW - TAKE EVERY 4TH FRAME FROM THE VICON DATA
    # TODO: Make this work better with 120 FPS
    temp_dict = {}
    counter = 1

    for i in trimmed_frames_vicon.keys():
        if counter == 1:
            temp_dict[i] = trimmed_frames_vicon[i]

        counter = counter + 1

        if counter == 5:
            # Reset
            counter = 1

    trimmed_frames_vicon = temp_dict

    # Calculate the indices of the frames we need to take from vicon, based on the clean realsense frames.
    vicon_indices_list = [0]

    for diff in differences_list_realsense:
        vicon_indices_list.append(vicon_indices_list[-1] + diff)

    # Take all vicon points based on the clean realsense frames, and remove the NAN frames.
    vicon_frames_clean = {}
    there_is_missing_point = False
    vicon_skipped_frames_numbers = []

    for idx, index in enumerate(vicon_indices_list):
        vicon_frame = list(trimmed_frames_vicon.keys())[index]
        points = trimmed_frames_vicon[vicon_frame]

        for point in points:
            if math.isnan(point.x) or math.isnan(point.y) or math.isnan(point.z):
                there_is_missing_point = True
                break

        if there_is_missing_point:
            # Reset the flag, and add the frame to the skipped frames list.
            vicon_skipped_frames_numbers.append(idx)
            there_is_missing_point = False
        else:
            # Add the frame to the clean frames.
            vicon_frames_clean[vicon_frame] = trimmed_frames_vicon[vicon_frame]

    # Clean the frames in realsense that have missing vicon correlated frames.
    final_frames_numbers_realsense = []

    for idx, frame in enumerate(frames_numbers_realsense):
        if idx in vicon_skipped_frames_numbers:
            continue

        final_frames_numbers_realsense.append(frame)

    return final_frames_numbers_realsense, vicon_frames_clean


def trim_single_realsense_video(bag_shoot_angle: str, sub_name: str, sub_position: str,
                                realsense_frames_numbers: List[int]) -> List[int]:
    """
    Trim a single realsense video. Ignore missing frames.

    :param bag_shoot_angle: 'Back', 'Front' or 'Side'.
    :param sub_name: e.g 'Sub004'.
    :param sub_position: 'Stand', 'Squat', 'Tight', 'Left' or 'Right'.
    :param first_frame_number: The number manually picked.
    :param total_frames_number:
    :return: List of differences between every 2 frames.
    """

    print("Starting trimming video " + sub_name + ", " + sub_position + ", " + bag_shoot_angle + "...")

    f = open('frames/' + sub_name + '/RealSense/' + sub_position + '/' + bag_shoot_angle + '/log.json')
    data = json.load(f)
    REALSENSE_FPS = data['FPS']

    # Get to the folder of all frames.
    folder_path = 'frames/' + sub_name + '/RealSense/' + sub_position + '/' + bag_shoot_angle + '/'

    # Starting from the given first_frame_number, create a new video.
    all_frames_files = os.listdir(folder_path)
    all_frames_files.remove('log.json')
    all_frames_files = sorted(all_frames_files, key=lambda x: int(x[:-4]))

    # Read all images.
    img_array = []
    for file in all_frames_files:
        current_frame_number = int(file[:-4])

        if current_frame_number not in realsense_frames_numbers:
            continue

        img = cv2.imread(folder_path + "/" + file)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    if not os.path.isdir('trimmed/' + sub_name + "/"):
        os.makedirs('trimmed/' + sub_name + "/")

    # Create the video
    out = cv2.VideoWriter("trimmed/" + sub_name + "/" + sub_name + "_" + sub_position + '_' + bag_shoot_angle + '.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'), REALSENSE_FPS, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("Finished.")


def trim_single_vicon_video(bag_shoot_angle: str, sub_name: str, sub_position: str, vicon_points: Dict):
    """
    Create video of the vicon points.

    :param sub_name: Sub name, e.g 'Sub005'.
    :param bag_shoot_angle: 'Front' or 'Back' or 'Side'
    :param sub_position: 'Squat' or 'Stand' or 'Left' or 'Right' or 'Tight'.
    :param vicon_points: Dictionary of <int, List<Point>> where the key is frame number, and the value is a list
    of 39 points.
    :return: None.
    """
    VICON_FPS = 30
    img_array_120 = []
    img_array_30 = []
    counter = 0

    for i, frame in enumerate(list(vicon_points.keys())):
        # Get 39 Vicon points.
        current_frame_points = vicon_points[frame]

        # Create an empty image to write the vicon points on in later.
        blank = np.zeros(shape=(640, 480, 3), dtype=np.uint8)
        vicon_image = cv2.cvtColor(blank, cv2.COLOR_RGB2BGR)

        for i, point in enumerate(current_frame_points):
            x = point.x
            z = point.z

            # Scale the coordinates so they will fit the image.
            x = x / 4.5
            z = z / 4.5
            # Draw the point on the blank image (orthographic projection).
            vicon_image = cv2.circle(vicon_image, ((int(z) + 50), (int(x) + 250)), radius=0, color=(0, 0, 255),
                                     thickness=10)  # Coordinates offsets are manually selected to center the object.

        # Rotate the image, since the vicon points are also rotated by default.
        vicon_image = cv2.rotate(vicon_image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        height, width, layers = vicon_image.shape
        size = (width, height)
        img_array_120.append(vicon_image)

        if (counter % 4) == 0:
            img_array_30.append(vicon_image)

        counter = counter + 1

    # Create the video
    out_120 = cv2.VideoWriter(
        "trimmed/" + sub_name + "/" + sub_name + "_" + sub_position + '_' + bag_shoot_angle + "_Vicon_120" + '.avi',
        cv2.VideoWriter_fourcc(*'DIVX'), VICON_FPS, size)
    for i in range(len(img_array_120)):
        out_120.write(img_array_120[i])
    out_120.release()

    # Create the video
    ''''
    out_30 = cv2.VideoWriter(
        "trimmed/" + sub_name + "/" + sub_name + "_" + sub_position + '_' + bag_shoot_angle + "_Vicon_30" + '.avi',
        cv2.VideoWriter_fourcc(*'DIVX'), int(VICON_FPS / 4), size)
    for i in range(len(img_array_30)):
        out_30.write(img_array_30[i])
    out_30.release()
    '''

    print("Finished.")


if __name__ == "__main__":
    sub_name = 'Sub005'
    sub_num = 5
    position = 'Right'

    f = open('frames_sync.json')
    data = json.load(f)

    for angle in ['Front', 'Back', 'Side']:
        first_frame_number_realsense = data[sub_num-1][position][angle]
        first_frame_number_vicon = data[sub_num-1][position]['Vicon']

        realsense_frames_numbers, vicon_points = find_clean_frames_indices_on_realsense_and_vicon(bag_shoot_angle=angle, sub_name=sub_name, sub_position=position,
                                                         first_frame_number_realsense=first_frame_number_realsense,
                                                         first_frame_number_vicon=first_frame_number_vicon)
        print(sub_name + ", " + position + ", " + angle + ", " + "realsense frames len: " + str(len(realsense_frames_numbers))
             + " vicon frames len: " + str(len(vicon_points.keys())))

        trim_single_realsense_video(bag_shoot_angle=angle, sub_name=sub_name, sub_position=position,
                                    realsense_frames_numbers=realsense_frames_numbers)
        trim_single_vicon_video(bag_shoot_angle=angle, sub_name=sub_name, sub_position=position, vicon_points=vicon_points)