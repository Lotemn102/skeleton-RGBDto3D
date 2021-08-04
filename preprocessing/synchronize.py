"""
Export all frames of RealSense video as images, and all frames of Vicon points as images (orthographic projected),
so a manually sync can be done based on this images.
"""

import cv2
import numpy as np
import os
import math

from realsense_data_reader import RealSenseReader
from vicon_data_reader import VICONReader

CSV_PATH = '../../Data/Sub007/Sub007_Vicon/Sub007 Squat.csv'
BAG_PATH = '../../Data/Sub007/Sub007_Front/Sub007_Squat_Front.bag'
FPS = 30


def generate_realsense_frames(bag_path: str):
    realsense_reader = RealSenseReader(bag_file_path=bag_path, type='RGB', frame_rate=30)
    pipeline = realsense_reader.setup_pipeline()

    cv2.namedWindow("RGB Stream", cv2.WINDOW_AUTOSIZE)

    # Get frameset
    frames = pipeline.wait_for_frames()
    first_frame = frames.frame_number
    current_frame = 0

    # Create save path.
    bag_file_name = bag_path.split("/")[-1]
    file_name = bag_file_name[:-4]
    save_path = 'frames/' + file_name

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    while current_frame != first_frame:
        # Get frameset.
        frames = pipeline.wait_for_frames()
        current_frame = frames.frame_number

        # Get color frame.
        color_frame = frames.get_color_frame()

        # Convert color_frame to numpy array to render image in opencv.
        color_image = np.asanyarray(color_frame.get_data())
        color_image = np.rot90(color_image, k=3)

        # Save image.
        cv2.imwrite(save_path + "/" + str(frames.frame_number) + '.png', color_image)

def generate_vicon_frames(csv_path: str):
    vicon_reader = VICONReader(vicon_file_path=CSV_PATH)
    vicon_points = vicon_reader.get_points()  # Dictionary of <frame_id, List<Point>>

    # Create save path.
    csv_file_name = csv_path.split("/")[-1]
    file_name = csv_file_name[:-4]
    save_path = 'frames/' + file_name + "_Vicon"

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for frame in list(vicon_points.keys()):
        print(frame)
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

        # Render realsense image and vicon image.
        #cv2.imshow("Vicon Stream", vicon_image)
        #cv2.waitKey(1)

        # Save image.
        cv2.imwrite(save_path + "/" + str(frame) + '.png', vicon_image)

def create_realsense_synchronized_video(bag_path: str, first_frame_number: int):
    # Get to the folder of all frames.
    bag_file_name = bag_path.split("/")[-1]
    file_name = bag_file_name[:-4]
    folder_path = 'frames/' + file_name

    # Starting from the given first_frame_number, create a new video.
    all_frames_files = os.listdir(folder_path)
    all_frames_files = sorted(all_frames_files, key=lambda x: int(x[:-4]))

    # Find first_frame_number index in the sorted list.
    first_frame_index = [i for i in range(len(all_frames_files)) if str(first_frame_number) in all_frames_files[i]][0]

    # Remove all frames before first_frame_number.
    cut_frames_files = all_frames_files[first_frame_index:]

    # Add padding of the same images for missing frames. This is due to a known bug in RealSense, that drops the FPS
    # down, hence resulting in missing frames. PLease refer to:
    # - https://github.com/IntelRealSense/librealsense/issues/8288
    # - https://github.com/IntelRealSense/librealsense/issues/2102
    # Iterate through frames, find frames that are missing.
    last_frame_number = int(cut_frames_files[0][:-4])
    imputed_cut_frames_files = []
    index = 0
    max_diff = 0

    for file in cut_frames_files:
        current_frame_number = int(file[:-4])
        diff = current_frame_number - last_frame_number

        if max_diff < diff:
            max_diff = diff

        if diff <= 1:
            imputed_cut_frames_files.append(file)
            index = index + 1
        else:
            # Append the file.
            imputed_cut_frames_files.append(file)

            # Add "padding" of the same file.
            for j in range(1, diff):
                imputed_cut_frames_files.insert(index+j, file)

            index = index + diff

        last_frame_number = current_frame_number

    print(max_diff)
    # Read all images.
    img_array = []
    for file in imputed_cut_frames_files:
        img = cv2.imread(folder_path + "/" + file)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    # Create the video
    out = cv2.VideoWriter(file_name + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), FPS, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == "__main__":
    #generate_realsense_frames(bag_path=BAG_PATH)
    # generate_vicon_frames(csv_path=CSV_PATH)
    create_realsense_synchronized_video(bag_path=BAG_PATH, first_frame_number=613)