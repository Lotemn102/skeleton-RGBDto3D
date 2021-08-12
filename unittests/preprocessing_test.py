import unittest
import cv2
import numpy as np
import math


from preprocessing.vicon_data_reader import VICONReader
from preprocessing.realsense_data_reader import RealSenseReader

class TestPreprocessing(unittest.TestCase):
    def test_read_data(self):
        REALSENSE_PATH = '../../Data/Sub013_Left_Back.bag'
        VICON_PATH = '../../Data/Sub007 Stand.csv'
        REALSENSE_FRAME_RATE = 30
        VICON_FRAME_RATE = 120

        # Init the VICON reader and read the points.
        vicon_reader = VICONReader(vicon_file_path=VICON_PATH)
        vicon_points = vicon_reader.get_points() # Dictionary of <frame_id, List<Point>>
        current_frame = list(vicon_points.keys())[0] # Get the first frame

        print('Number of vicon points: ' + str(len(vicon_points)))
        print('First vicon frame: ' + str(list(vicon_points.keys())[0]))
        print('Last vicon frame: ' + str(list(vicon_points.keys())[-1]))

        # Init the realsense reader and get the pipeline.
        try:
            # Most of the videos were recorded with FPS of 30.
            realsense_reader = RealSenseReader(bag_file_path=REALSENSE_PATH, type='RGB', frame_rate=REALSENSE_FRAME_RATE)
            pipeline = realsense_reader.setup_pipeline()
            print(30)
        except:
            # Some videos were recorded with FPS of 15.
            REALSENSE_FRAME_RATE = 15
            realsense_reader = RealSenseReader(bag_file_path=REALSENSE_PATH, type='RGB', frame_rate=REALSENSE_FRAME_RATE)
            pipeline = realsense_reader.setup_pipeline()
            print(15)

        # Set-up two windows.
        cv2.namedWindow("RGB Stream", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Vicon Stream", cv2.WINDOW_AUTOSIZE)

        # Start playing the videos.
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            realsense_image = np.asanyarray(color_frame.get_data())
            realsense_image = np.rot90(realsense_image, k=3) # If image is rotated by default

            # Read Vicon points
            try:
                current_frame_points = vicon_points[current_frame]
            except KeyError:
                pass

            # Create an empty image to write the vicon points on in later.
            blank = np.zeros(shape=(800, 800, 3), dtype=np.uint8)
            vicon_image = cv2.cvtColor(blank, cv2.COLOR_RGB2BGR)
            #scale = get_scale(points)

            for i, point in enumerate(current_frame_points):
                x = point.x
                y = point.y
                z = point.z

                if math.isnan(x) or math.isnan(y) or math.isnan(z):
                    # Skip this point for the moment
                    print("empty point")
                    continue

                # Scale the coordinates so they will fit the image.
                x = x / 5
                z = z / 5
                # Draw the point on the blank image (orthographic projection).
                vicon_image = cv2.circle(vicon_image, ((int(z) + 300), (int(x) + 400)), radius=0, color=(0, 0, 255),
                                         thickness=10) # Coordinates offsets are manually selected to center the object.

            vicon_image = cv2.rotate(vicon_image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE) # The vicon points are also rotated
            # by default.

            # Render realsense image and vicon image.
            #cv2.imshow("RGB Stream", realsense_image)
            cv2.imshow("Vicon Stream", vicon_image)
            key = cv2.waitKey(1)
            t = (VICON_FRAME_RATE / REALSENSE_FRAME_RATE)
            current_frame = current_frame + (VICON_FRAME_RATE / REALSENSE_FRAME_RATE)

    def test_sync(self):
        FPS = 30
        SUB_NUMBER = '005'
        POSITIONS_LIST = ['Squat', 'Stand', 'Left', 'Right', 'Tight']
        POSITON = POSITIONS_LIST[3]
        FIRST_VIDEO_PATH = '../preprocessing/trimmed/Sub' + str(SUB_NUMBER) + '/Sub' + str(SUB_NUMBER) + '_' + POSITON + '_Back.avi'
        SECOND_VIDEO_PATH = '../preprocessing/trimmed/Sub' + str(SUB_NUMBER) + '/Sub' + str(SUB_NUMBER) + '_' + POSITON +  '_Front.avi'
        THIRD_VIDEO_PATH = '../preprocessing/trimmed/Sub' + str(SUB_NUMBER) + '/Sub' + str(SUB_NUMBER) + '_' + POSITON + '_Side.avi'
        VICON_VIDEO_PATH = '../preprocessing/trimmed/Sub' + str(SUB_NUMBER) + '/Sub' + str(SUB_NUMBER) + '_' + POSITON +  '_Vicon.avi'

        cap_1 = cv2.VideoCapture(FIRST_VIDEO_PATH)
        cap_2 = cv2.VideoCapture(SECOND_VIDEO_PATH)
        cap_3 = cv2.VideoCapture(THIRD_VIDEO_PATH)
        cap_4 = cv2.VideoCapture(VICON_VIDEO_PATH)

        # Set-up two windows.
        cv2.namedWindow("RGB 1", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("RGB 2", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("RGB 3", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("VICON", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("RGB 1", 0, 0,)
        cv2.moveWindow("RGB 2", 540, 0,)
        cv2.moveWindow("RGB 3", 1000, 0,)
        cv2.moveWindow("VICON", 1700, 0,)

        counter = 0
        first_iteration = True

        while cap_1.isOpened():
            ret_1, frame_1 = cap_1.read()
            ret_2, frame_2 = cap_2.read()
            ret_3, frame_3 = cap_3.read()
            ret_4, frame_4 = cap_4.read()

            while counter < 4: # Read every 4th frames from vicon, since vicon FPS is 120
                ret_4, frame_4 = cap_4.read()
                counter = counter + 1

            if not ret_1 or not ret_2 or not ret_3 or not ret_4 or frame_4 is None: # Loop
                cap_1.set(1, 0) # Set current frame number to 0
                cap_2.set(1, 0) # Set current frame number to 0
                cap_3.set(1, 0) # Set current frame number to 0
                cap_4.set(1, 0) # Set current frame number to 0
                continue

            if first_iteration:
                scale_percent = 90
                width_realsense = int(frame_1.shape[1] * scale_percent / 100)
                height_realsense = int(frame_1.shape[0] * scale_percent / 100)
                dims_realsense = (width_realsense, height_realsense)

                width_vicon = int(frame_4.shape[1] * scale_percent / 100)
                height_vicon = int(frame_4.shape[0] * scale_percent / 100)
                dims_vicon = (width_vicon, height_vicon)
                first_iteration = False

            counter = 0
            frame_1 = cv2.resize(frame_1, dims_realsense, interpolation=cv2.INTER_AREA)
            frame_2 = cv2.resize(frame_2, dims_realsense, interpolation=cv2.INTER_AREA)
            frame_3 = cv2.resize(frame_3, dims_realsense, interpolation=cv2.INTER_AREA)
            frame_4 = cv2.resize(frame_4, dims_vicon, interpolation=cv2.INTER_AREA)

            if not ret_1 or not ret_2 or not ret_3 or not ret_4: # Loop
                cap_1.set(1, 0) # Set current frame number to 0
                cap_2.set(1, 0) # Set current frame number to 0
                cap_3.set(1, 0) # Set current frame number to 0
                cap_4.set(1, 0) # Set current frame number to 0
            else:
                img_1 = cv2.cvtColor(frame_1, cv2.COLOR_RGB2BGR)
                img_2 = cv2.cvtColor(frame_2, cv2.COLOR_RGB2BGR)
                img_3 = cv2.cvtColor(frame_3, cv2.COLOR_RGB2BGR)
                img_4 = cv2.cvtColor(frame_4, cv2.COLOR_RGB2BGR)
                cv2.imshow('RGB 1', img_1)
                cv2.imshow('RGB 2', img_2)
                cv2.imshow('RGB 3', img_3)
                cv2.imshow('VICON', img_4)
                cv2.waitKey(FPS)

    def test_trim(self):
        FPS = 30
        SUB_NUMBER = '005'
        POSITIONS_LIST = ['Squat', 'Stand', 'Left', 'Right', 'Tight']
        ANGLE_LIST = ['Back', 'Front', 'Side']
        POSITON = POSITIONS_LIST[3]
        ANGLE = ANGLE_LIST[1]
        RGB_VIDEO_PATH = '../preprocessing/trimmed/Sub' + str(SUB_NUMBER) + '/Sub' + str(
            SUB_NUMBER) + '_' + POSITON + '_' + ANGLE + '.avi'
        VICON_VIDEO_PATH = '../preprocessing/trimmed/Sub' + str(SUB_NUMBER) + '/Sub' + str(
            SUB_NUMBER) + '_' + POSITON + '_' + ANGLE + '_Vicon_120.avi'

        cap_1 = cv2.VideoCapture(RGB_VIDEO_PATH)
        cap_2 = cv2.VideoCapture(VICON_VIDEO_PATH)

        # Set-up two windows.
        cv2.namedWindow("RGB", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("VICON", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("RGB", 0, 0, )
        cv2.moveWindow("VICON", 1700, 0, )

        counter = 0
        first_iteration = True

        while cap_1.isOpened():
            ret_2, frame_2 = cap_2.read()
            ret_1, frame_1 = cap_1.read()

            if not ret_1 or not ret_2: # Loop
                cap_1.set(1, 0) # Set current frame number to 0
                cap_2.set(1, 0) # Set current frame number to 0
                continue

            if first_iteration:
                scale_percent = 90
                width_realsense = int(frame_1.shape[1] * scale_percent / 100)
                height_realsense = int(frame_1.shape[0] * scale_percent / 100)
                dims_realsense = (width_realsense, height_realsense)

                width_vicon = int(frame_2.shape[1] * scale_percent / 100)
                height_vicon = int(frame_2.shape[0] * scale_percent / 100)
                dims_vicon = (width_vicon, height_vicon)
                first_iteration = False

            frame_1 = cv2.resize(frame_1, dims_realsense, interpolation=cv2.INTER_AREA)
            frame_2 = cv2.resize(frame_2, dims_vicon, interpolation=cv2.INTER_AREA)

            if not ret_1 or not ret_2:
                cap_1.set(1, 0)  # Set current frame number to 0
                cap_2.set(1, 0)  # Set current frame number to 0
            else:
                img_1 = cv2.cvtColor(frame_1, cv2.COLOR_RGB2BGR)
                img_2 = cv2.cvtColor(frame_2, cv2.COLOR_RGB2BGR)
                cv2.imshow('RGB', img_1)
                cv2.imshow('VICON', img_2)
                cv2.waitKey(FPS)



