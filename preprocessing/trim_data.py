"""
Trim the realsense video and vicon data, based on the synchronized frames manually found beforehand.
All missing frames (both on realsense and vicon) are removed.
"""
import copy
import json
import os
import cv2
import numpy as np
from typing import List, Dict
import math
import shutil
import pandas as pd
import csv

from preprocessing.vicon_data_reader import VICONReader
from preprocessing.structs import Point


def sync_30_fps(bag_shoot_angle: str, sub_name: str, sub_position: str,
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
    csv_path = 'D:/Movement Sense Research/Vicon Validation Study/' + sub_name + "/" + \
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
        try:
            vicon_frame = list(trimmed_frames_vicon.keys())[index]
            points = trimmed_frames_vicon[vicon_frame]
        except:
            continue

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

    if len(final_frames_numbers_realsense) > len(vicon_frames_clean):
        length = min(len(final_frames_numbers_realsense), len(vicon_frames_clean))
        final_frames_numbers_realsense = final_frames_numbers_realsense[:length]

    return final_frames_numbers_realsense, vicon_frames_clean

def sync_120_fps(bag_shoot_angle: str, sub_name: str, sub_position: str,
                 first_frame_number_realsense: int,
                 first_frame_number_vicon: int, test_realsense: List[str] = None,
                 test_csv: str = None) -> (List[int], Dict):
    # -------------------------------------------- Find realsense clean frames -----------------------------------------
    f = open('frames/' + sub_name + '/RealSense/' + sub_position + '/' + bag_shoot_angle + '/log.json')
    data = json.load(f)
    REALSENSE_FPS = data['FPS'] # 15 or 30
    fps_diff = int(120 / REALSENSE_FPS) # Vicon FPS is always 120
    print("FPS diff: " + str(fps_diff))

    if test_realsense is None:
        # Get to the folder of all frames.
        folder_path_realsense = 'frames/' + sub_name + '/RealSense/' + sub_position + '/' + bag_shoot_angle + '/'

        # Starting from the given first_frame_number, create a new video.all_frames_files_realsense
        all_frames_files_realsense = os.listdir(folder_path_realsense)
        all_frames_files_realsense.remove('log.json')
        all_frames_files_realsense = sorted(all_frames_files_realsense, key=lambda x: int(x[:-4]))
    else:
        all_frames_files_realsense = test_realsense

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
    if test_csv is None:
        csv_path = '/media/lotemn/Transcend/Movement Sense Research/Vicon Validation Study/' + sub_name + "/" + \
                   sub_name + "_" + "Vicon/"  + sub_name + " " + sub_position + ".csv"
    else:
        csv_path = test_csv

    vicon_reader = VICONReader(vicon_file_path=csv_path)
    vicon_points = vicon_reader.get_points()  # Dictionary of <frame_id, List<Point>>

    # Find first_frame_number index in the dict.
    all_frames_vicon = list(vicon_points.keys())
    first_frame_index_vicon = [i for i in range(len(all_frames_vicon)) if
                                   first_frame_number_vicon == all_frames_vicon[i]][0]

    # Remove all frames before first_frame_number.
    trimmed_frames_vicon = {k: vicon_points[k] for k in list(vicon_points.keys())[first_frame_index_vicon:]}

    # Calculate the indices of the frames we need to take from vicon, based on the clean realsense frames.
    if fps_diff == 4: # Realsense FPS is 30
        vicon_indices_list = [0, 1, 2, 3]
    elif fps_diff == 8: # Realsense FPS is 15
        vicon_indices_list = [0, 1, 2, 3, 4, 5, 6, 7]
    else:
        print('Error in fps diff.')
        return

    if fps_diff == 4:
        for diff in differences_list_realsense:
            # Each realsense frame has 4 correlated vicon frames.
            vicon_indices_list.append(vicon_indices_list[-4] + diff*4)
            vicon_indices_list.append(vicon_indices_list[-1] + 1)
            vicon_indices_list.append(vicon_indices_list[-1] + 1)
            vicon_indices_list.append(vicon_indices_list[-1] + 1)
    elif fps_diff == 8:
        for diff in differences_list_realsense:
            # Each realsense frame has 8 correlated vicon frames.
            vicon_indices_list.append(vicon_indices_list[-8] + diff*8)
            vicon_indices_list.append(vicon_indices_list[-1] + 1)
            vicon_indices_list.append(vicon_indices_list[-1] + 1)
            vicon_indices_list.append(vicon_indices_list[-1] + 1)
            vicon_indices_list.append(vicon_indices_list[-1] + 1)
            vicon_indices_list.append(vicon_indices_list[-1] + 1)
            vicon_indices_list.append(vicon_indices_list[-1] + 1)
            vicon_indices_list.append(vicon_indices_list[-1] + 1)

    # Take all vicon points based on the clean realsense frames, and remove the NAN frames.
    vicon_frames_clean = {}
    there_is_missing_point = False
    vicon_skipped_frames_numbers = []

    for idx in range(0, len(vicon_indices_list), fps_diff):
        for i in range(0, fps_diff): # Check if there is NAN point in all 4 frames correlated to the single realsense frame
            vicon_frame = list(trimmed_frames_vicon.keys())[idx+i]
            points = trimmed_frames_vicon[vicon_frame]

            for point in points:
                if math.isnan(point.x) or math.isnan(point.y) or math.isnan(point.z):
                    there_is_missing_point = True
                    break

        if there_is_missing_point:
            # Reset the flag, and add the frame to the skipped frames list.
            vicon_skipped_frames_numbers.append(int(idx / fps_diff))
            there_is_missing_point = False
        else:
            # Add the frames to the clean frames.
            for i in range(fps_diff): # Add 4 frames!
                vicon_frame = list(trimmed_frames_vicon.keys())[idx + i]
                vicon_frames_clean[vicon_frame] = trimmed_frames_vicon[vicon_frame]

    # Clean the frames in realsense that have missing vicon correlated frames.
    final_frames_numbers_realsense = []

    for idx, frame in enumerate(frames_numbers_realsense):
        if idx in vicon_skipped_frames_numbers:
            continue

        final_frames_numbers_realsense.append(frame)

    if len(final_frames_numbers_realsense)*fps_diff != len(vicon_frames_clean):
        print("Frames mismatch!")
        print("realsense frames: " + str(len(final_frames_numbers_realsense)) + ", vicon frames: " + \
              str(len(vicon_frames_clean)))

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
    save_folder = 'trimmed/' + sub_name + '/' + sub_position + '/' + bag_shoot_angle + '/'

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    out = cv2.VideoWriter(save_folder + sub_name + "_" + sub_position + '_' + bag_shoot_angle + '.avi',
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
    VICON_FPS = 120
    img_array_120 = []

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

def trim_single_csv_file(bag_shoot_angle: str, sub_name: str, sub_position: str, vicon_points: Dict):
    print("Starting trimming csv " + sub_name + ", " + sub_position + ", " + bag_shoot_angle + "...")

    # Write the points to a new csv file.
    csv_template_path = 'assets/csv_template.csv'
    trimmed_csv_folder = 'trimmed/' + sub_name + '/' + sub_position + '/' + bag_shoot_angle + '/'

    if not os.path.isdir(trimmed_csv_folder):
        os.makedirs(trimmed_csv_folder)

    trimmed_csv_path = 'trimmed/' + sub_name + '/' + sub_position + '/' + bag_shoot_angle + '/' + sub_name \
                        + '_' + sub_position + '_' + bag_shoot_angle + '.csv'
    shutil.copy2(csv_template_path, trimmed_csv_path)

    # Change 'SubXXX' to subject's name.
    file = pd.read_csv(trimmed_csv_path)
    file.replace(to_replace='SubXXX', value=sub_name)
    file.to_csv(trimmed_csv_path, header=False, index=False)

    rows = []

    for frame in vicon_points.keys():
        current = []
        current.append(frame) # Frame
        current.append(0) # SubFrame

        for point in vicon_points[frame]:
            current.append(point.x)
            current.append(point.y)
            current.append(point.z)

        rows.append(current)

    with open(trimmed_csv_path, 'a') as file:
        writer = csv.writer(file)
        for row in rows:
            writer.writerow(row)

    print("Finished.")

def rotate_vicon_points_90_degree_counterclockwise(rotation_axis: str, csv_path: str = None, points: Dict = None):
    if csv_path is not None:
        vicon_reader = VICONReader(vicon_file_path=csv_path)
        vicon_points = vicon_reader.get_points()  # Dictionary of <frame_id, List<Point>>
    elif points is not None:
        vicon_points = points
    else:
        print('Error with vicon points input.')
        return

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

    for frame, points in vicon_points.items():
        rotated_points = []

        for point in points:
            transformed = rotation_matix.dot(np.array([point.x, point.y, point.z]))
            rotated_point = Point(transformed[0], transformed[1], transformed[2])
            rotated_points.append(rotated_point)

        final_points[frame] = rotated_points

    return final_points

'''
def rotate_vicon_points_90_degree_clockwise(rotation_axis: str, csv_path: str = None, points: Dict = None):
    if csv_path is not None:
        vicon_reader = VICONReader(vicon_file_path=csv_path)
        vicon_points = vicon_reader.get_points()  # Dictionary of <frame_id, List<Point>>
    elif points is not None:
        vicon_points = points
    else:
        print('Error with vicon points input.')
        return

    if rotation_axis.lower() == 'x':
        rotation_matix = np.ndarray((3, 3), dtype=float)
        rotation_matix[0] = [1, 0, 0]
        rotation_matix[1] = [0, 0, -1]
        rotation_matix[2] = [0, 1, 0]
    elif rotation_axis.lower() == 'y':
        rotation_matix = np.ndarray((3, 3), dtype=float)
        rotation_matix[0] = [0, 0, 1]
        rotation_matix[1] = [0, 1, 0]
        rotation_matix[2] = [-1, 0, 0]
    elif rotation_axis.lower() == 'z':
        rotation_matix = np.ndarray((3, 3), dtype=float)
        rotation_matix[0] = [0, -1, 0]
        rotation_matix[1] = [1, 0, 0]
        rotation_matix[2] = [0, 0, 1]
    else:
        print('Wrong axis. Use \"x\",  \"y\" or \"z\".')
        return

    final_points = {}

    for frame, points in vicon_points.items():
        rotated_points = []

        for point in points:
            transformed = rotation_matix.dot(np.array([point.x, point.y, point.z]))
            rotated_point = Point(transformed[0], transformed[1], transformed[2])
            rotated_points.append(rotated_point)

        final_points[frame] = rotated_points

    return final_points
'''

def trim_all():
    f1 = open('assets/frames_sync.json')
    frames_sync = json.load(f1)

    for i in range(4, 5):
        subject_name = 'Sub00' + str(i) if i < 10 else 'Sub0' + str(i)
        subject_num = i

        for position in ['Stand', 'Squat', 'Tight', 'Left', 'Right']:
            for angle in ['Front', 'Back', 'Side']:
                first_frame_number_realsense = frames_sync[subject_num - 1][position][angle]
                first_frame_number_vicon = frames_sync[subject_num - 1][position]['Vicon']

                realsense_frames_numbers, vicon_points = sync_30_fps(bag_shoot_angle=angle,
                                                                      sub_name=subject_name,
                                                                      sub_position=position,
                                                                      first_frame_number_realsense=first_frame_number_realsense,
                                                                      first_frame_number_vicon=first_frame_number_vicon)

                # TODO: Ask Ron about the rotation
                #rotated_vicon_points = rotate_vicon_points_90_degree_counterclockwise(rotation_axis='x', points=vicon_points)

                trim_single_realsense_video(bag_shoot_angle=angle, sub_position=position, sub_name=subject_name,
                                            realsense_frames_numbers=realsense_frames_numbers)
                trim_single_csv_file(bag_shoot_angle=angle, sub_name=subject_name, sub_position=position,
                                     vicon_points=vicon_points)


if __name__ == "__main__":
    trim_all()