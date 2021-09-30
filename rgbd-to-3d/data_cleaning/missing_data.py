"""
Plot some graphs about missing data in RealSense and vicon, for data quality evaluation.
"""

import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2

from vicon_data_reader import VICONReader
from realsense_data_reader import RealSenseReader


def generate_max_missing_points_vicon(csv_path: str, filename: str):
    vicon_reader = VICONReader(vicon_file_path=csv_path)
    vicon_points = vicon_reader.get_points()  # Dictionary of <frame_id, List<Point>>

    max_missing_frames_per_point = {} # Dictionary of <point_index, max_number_of_repeated_frames_the_point_is_missing>
    missing_counter = {}

    for i in range(0, 39):
        missing_counter[i] = 0
        max_missing_frames_per_point[i] = 0

    for frame_idx, frame_points in enumerate(vicon_points.values()):
        for point_idx, point in enumerate(frame_points):
            if math.isnan(point.x) or math.isnan(point.y) or math.isnan(point.z):
                missing_counter[point_idx] += 1

                if missing_counter[point_idx] > max_missing_frames_per_point[point_idx]:
                    max_missing_frames_per_point[point_idx] = missing_counter[point_idx]
            else:
                missing_counter[point_idx] = 0

    x = list(max_missing_frames_per_point.keys())
    y = list(max_missing_frames_per_point.values())
    sns.set_style("darkgrid")
    plt.rcParams['xtick.labelsize'] = 7
    plt.scatter(x, y)
    plt.xticks(x)
    plt.xlabel("Point index")
    plt.ylabel("Max number of following frames the point is missing")
    plt.title("Max number of following frames points are missing in " + filename)
    plt.savefig('plots/missing-vicon-points/' + filename + '.png')
    plt.close()

def generate_max_missing_frames_realsense(bag_path: str, filename: str):
    realsense_reader = RealSenseReader(bag_file_path=bag_path, type='RGB', frame_rate=30)
    pipeline = realsense_reader.setup_pipeline()
    cv2.namedWindow("RGB Stream", cv2.WINDOW_AUTOSIZE)

    # Get frameset
    frames = pipeline.wait_for_frames()
    first_frame = frames.frame_number
    current_frame_number = 0
    last_frame_number = first_frame
    diff_list = []

    while current_frame_number != first_frame:
        # Get frameset.
        frames = pipeline.wait_for_frames()
        current_frame_number = frames.frame_number
        diff = current_frame_number - last_frame_number
        diff_list.append(diff)
        last_frame_number = current_frame_number

    x = list(range(len(diff_list)))
    x = x[:-10] # Not sure why but getting weird values on last frames, pretty sure it has nothing to do with realsense
    y = diff_list
    y = y[:-10] # Not sure why but getting weird values on last frames, pretty sure it has nothing to do with realsense

    sns.set_style("darkgrid")
    plt.xlabel("Frame")
    plt.ylabel("Difference")
    plt.title("Differences between current frame number to previous frame number in " + filename, fontsize=8)
    plt.plot(x, y)
    plt.savefig('plots/missing-realsense-frames/' + filename + '.png')
    plt.close()

def aux_missing_vicon():
    for root, dirs, files in os.walk("D:\\Movement Sense Research\\Vicon Validation Study"):
        for file in files:
            if file.endswith(".csv"):
                if "Sub001" in file or "Sub002" in file:
                    continue
                else:
                    generate_max_missing_points_vicon(root + "\\" + file, file)

def aux_missing_realsense():
    for root, dirs, files in os.walk("/media/lotemn/Transcend/Movement Sense Research/Vicon Validation Study"):
        for file in files:
            if file.endswith(".bag"):
                if "Sub006" in file:
                    print(file)
                    generate_max_missing_frames_realsense(root + "/" + file, file)

                '''
                if "Sub001" in file or "Sub002" in file or "Sub003" in file or "Sub004" in file or "20210216" in file \
                        or "sub02" in file or "Sub005" in file or "006" in file or '2021' in file:
                    continue
                else:
                    print(file)
                    generate_max_missing_frames_realsense(root + "/" + file, file)
                '''''


if __name__ == "__main__":
    pass
    #aux_missing_realsense()
