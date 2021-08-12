"""
List of function for manually synchronizing RealSense videos and Vicon points.

Pipeline
=======
    1. Extract all frames of RealSense video as images using 'generate_realsense_frames()'.
    2. Extract all frames of Vicon points as images (orthographic projected) using 'generate_vicon_frames()'.
    3. MANUALLY find the first frame in the Vicon frames, and it's correlated frame in RealSense data. Save the
        data in 'frames_sync.json'.
    4. Cut the videos by calling 'generate_synchronized_videos_for_all()'.
"""

import cv2
import numpy as np
import os
import math
import json

from realsense_data_reader import RealSenseReader
from vicon_data_reader import VICONReader

# ---------------------- Generate frames for manually detecting T-pose -------------------------------------------------
"""
For manually detecting the T-pose frames.
"""
def generate_realsense_frames(bag_path: str, bag_shoot_angle: str, sub_name: str, sub_position: str):
    """
    Extract frames from .bag file, and save them as images.

    :param bag_path: Path to the bag file.
    :param sub_name: Sub name, e.g 'Sub005'.
    :param bag_shoot_angle: 'Front' or 'Back' or 'Side'
    :param sub_position: 'Squat' or 'Stand' or 'Left' or 'Right' or 'Tight'.
    :return: None.
    """
    try:
        # Most of the videos were recorded with FPS of 30.
        REALSENSE_FPS = 30
        realsense_reader = RealSenseReader(bag_file_path=bag_path, type='RGB', frame_rate=REALSENSE_FPS)
        pipeline = realsense_reader.setup_pipeline()
        print('Skipped.')
        return
    except:
        # Some videos were recorded with FPS of 15.
        REALSENSE_FPS = 15
        realsense_reader = RealSenseReader(bag_file_path=bag_path, type='RGB', frame_rate=REALSENSE_FPS)
        pipeline = realsense_reader.setup_pipeline()

    cv2.namedWindow("RGB Stream", cv2.WINDOW_AUTOSIZE)


    # Get frameset
    frames = pipeline.wait_for_frames()
    first_frame = frames.frame_number
    current_frame = 0

    # Create save path.
    save_path = 'frames/' + sub_name + '/RealSense/' + sub_position + '/' + bag_shoot_angle + '/'

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

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

    # Read all frames to find first one, last one and number of frames.
    all_frames = os.listdir(save_path)
    all_frames = sorted(all_frames, key=lambda x: int(x[:-4]))
    first_frame = all_frames[0][:-4]
    last_frame = all_frames[-1][:-4]
    number_of_frames = len(all_frames)

    # Write metadata to json file
    metadata = {
        'sub_name' : sub_name,
        'sub_position' : sub_position,
        'shooting_angle' : bag_shoot_angle,
        'first_frame' : int(first_frame),
        'last_frame' : int(last_frame),
        'number_of_padding_frames' : -1,
        'total_frames_without_padding' : number_of_frames,
        'total_frames_with_padding' : -1,
        'width' : 480,
        'height' : 640,
        'FPS' : REALSENSE_FPS
    }

    json_data = json.dumps(metadata)
    json_path = save_path + 'log.json'
    json_file = open(json_path, "w")
    json_file.write(json_data)
    json_file.close()

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

    save_path = 'frames/' + sub_name + "/Vicon/" + sub_position

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
                if 'Sub001' in file or 'Sub002' in file:
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

                if subject_name in ['Sub003', 'Sub004', 'Sub005', 'Sub006', 'Sub007', 'Sub008']:
                    continue

                if subject_name == 'Sub009' and subject_position != 'Back':
                    continue

                print("Working on " + subject_name + ", " + subject_position + ", " + shooting_angle)

                try:
                    generate_realsense_frames(bag_path=root + "/" + file, sub_name=subject_name,
                                              bag_shoot_angle=shooting_angle, sub_position=subject_position)
                except:
                    print("Couldn't read " + file)

def aux_generate_vicon_frames():
    for root, dirs, files in os.walk("/media/lotemn/Transcend/Movement Sense Research/Vicon Validation Study"):
        for file in files:
            if file.endswith(".csv"):
                if 'Sub001' in file or 'Sub002' in file or 'Sub003' in file:
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

# --------------------------------------------- Create trimmed videos -------------------------------------------------
"""
For evaluating the synchronizing accuracy.
"""

def create_realsense_synchronized_video(bag_shoot_angle: str, sub_name: str, sub_position: str, first_frame_number: int,
                                        total_frames_number: int):
    """
    Create video from the frames. This implementation currently add padding of frames for the frames gaps!

    :param bag_shoot_angle: 'Back', 'Front' or 'Side'.
    :param sub_name: e.g 'Sub004'.
    :param sub_position: 'Stand', 'Squat', 'Tight', 'Left' or 'Right'.
    :param first_frame_number: The number manually picked.
    :param total_frames_number: Number of frames this video should have.
    :return: None.
    """
    print("Starting creating video " + sub_name + ", " + sub_position + ", " + bag_shoot_angle + "...")

    f = open('frames/' + sub_name + '/RealSense/' + sub_position + '/' + bag_shoot_angle + '/log.json')
    data = json.load(f)
    REALSENSE_FPS = data['FPS']

    # Get to the folder of all frames.
    folder_path = 'frames/' + sub_name + '/RealSense/' + sub_position + '/' + bag_shoot_angle + '/'

    # Starting from the given first_frame_number, create a new video.
    all_frames_files = os.listdir(folder_path)
    all_frames_files.remove('log.json')
    all_frames_files = sorted(all_frames_files, key=lambda x: int(x[:-4]))

    # Find first_frame_number index in the sorted list.
    first_frame_index = [i for i in range(len(all_frames_files)) if str(first_frame_number) in all_frames_files[i]][0]

    # Remove all frames before first_frame_number.
    cut_frames_files = all_frames_files[first_frame_index:]

    # Add padding of the same images for missing frames. This is due to a known bug in RealSense, resulting in missing
    # frames. Please refer to:
    # - https://github.com/IntelRealSense/librealsense/issues/8288
    # - https://github.com/IntelRealSense/librealsense/issues/2102
    # PADDING IS ONLY FOR DEBUGGING THE SYNCHRONIZING PROCESS.
    # Iterate through frames, find frames that are missing.
    previous_frame_number = int(cut_frames_files[0][:-4])
    padded_cut_frames_files = []
    index = 0
    max_diff = 0

    last_frame_number = first_frame_number + total_frames_number # For trimmimg the video to fit the shortest recording.

    for file in cut_frames_files:
        current_frame_number = int(file[:-4])

        if current_frame_number >= last_frame_number:
            break

        diff = current_frame_number - previous_frame_number

        if max_diff < diff:
            max_diff = diff

        if diff <= 1:
            padded_cut_frames_files.append(file)
            index = index + 1
        else:
            # Append the file.
            padded_cut_frames_files.append(file)

            # Add "padding" of the same file.
            for j in range(1, diff):
                padded_cut_frames_files.insert(index+j, file)

            index = index + diff

        previous_frame_number = current_frame_number

    print("Number of frames: " + str(len(padded_cut_frames_files)))

    # Read all images.
    img_array = []
    for file in padded_cut_frames_files:
        img = cv2.imread(folder_path + "/" + file)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    if not os.path.isdir('clean/' + sub_name + "/"):
        os.makedirs('clean/' + sub_name + "/")

    # Create the video
    out = cv2.VideoWriter("clean/" + sub_name + "/" + sub_name + "_" + sub_position + '_' + bag_shoot_angle + '.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'), REALSENSE_FPS, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("Finished.")

def create_vicon_synchronized_video(sub_name: str, sub_position: str, first_frame_number: int, total_frames_number: int):
    """
    Create video from the frames. This implementation currently doesn't remove Vicon frames!

    :param sub_name: e.g 'Sub004'.
    :param sub_position: 'Stand', 'Squat', 'Tight', 'Left' or 'Right'.
    :param first_frame_number:  The number manually picked.
    :return: None.
    """
    print("Starting creating Vicon video...")

    VICON_FPS = 120

    # Get to the folder of all frames.
    folder_path = 'frames/' + sub_name + '/Vicon/' + sub_position + '/'

    # Starting from the given first_frame_number, create a new video.
    all_frames_files = os.listdir(folder_path)
    all_frames_files = sorted(all_frames_files, key=lambda x: int(x[:-4]))

    # Find first_frame_number index in the sorted list.
    first_frame_index = [i for i in range(len(all_frames_files)) if str(first_frame_number) in all_frames_files[i]][0]

    # Remove all frames before first_frame_number.
    cut_frames_files = all_frames_files[first_frame_index:]

    # Trimming the video to fit the shortest recording.
    last_frame_number = first_frame_number + total_frames_number
    trimmed_frames_files = []

    for file in cut_frames_files:
        current_frame = int(file[:-4])
        if current_frame >= last_frame_number:
            break
        else:
            trimmed_frames_files.append(file)

    print("Number of frames: " + str(len(trimmed_frames_files)))

    # Read all images.
    img_array = []
    for file in trimmed_frames_files:
        img = cv2.imread(folder_path + "/" + file)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    if not os.path.isdir('clean/' + sub_name + "/"):
        os.makedirs('clean/' + sub_name + "/")

    # Create the video
    out = cv2.VideoWriter("clean/" + sub_name + "/" + sub_name + "_" + sub_position + '_Vicon.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'), VICON_FPS, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    print("Finished.")

def generate_synchronized_videos_for_all():
    f = open('frames_sync.json')
    data = json.load(f)

    for i in range(9, 10):
        subject_name = 'Sub00' + str(i) if i < 10 else 'Sub0' + str(i)

        for position in ['Stand', 'Squat', 'Tight', 'Left', 'Right']:
            # Calculate number of frames after trimming for each video & vicon, so we can know which one is the shortest.
            min_frames_number = np.Inf

            for angle in ['Back', 'Front', 'Side', 'Vicon']:
                first_frame_number = int(data[i - 1][position][angle])

                # Starting from the given first_frame_number, create a new video.
                if angle != 'Vicon':
                    path = 'frames/' + subject_name + '/RealSense/' + position + '/' + angle
                else:
                    path = 'frames/' + subject_name + '/Vicon/' + position + '/'

                frames = os.listdir(path)
                if 'log.json' in frames:
                    frames.remove('log.json')
                frames = sorted(frames, key=lambda x: int(x[:-4]))

                # Find number of frames.
                last_frame = int(frames[-1][:-4])
                diff = last_frame - first_frame_number

                if diff < min_frames_number:
                    min_frames_number = diff

            print(position + " " + str(min_frames_number))

            # Cut RealSense videos.
            for angle in ['Back', 'Front', 'Side']:
                first_frame_number = data[i-1][position][angle]
                create_realsense_synchronized_video(sub_name=subject_name, sub_position=position, bag_shoot_angle=angle,
                                                    first_frame_number=first_frame_number,
                                                    total_frames_number=min_frames_number)

            # Cut vicon video.
            first_frame_number = data[i - 1][position]['Vicon']
            create_vicon_synchronized_video(sub_name=subject_name, sub_position=position,
                                            first_frame_number=first_frame_number,
                                            total_frames_number=(min_frames_number * 4)) # Vicon FPS is 120


if __name__ == "__main__":
    generate_synchronized_videos_for_all()

