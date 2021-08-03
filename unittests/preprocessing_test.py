import unittest
import cv2
import numpy as np
import math


from preprocessing.vicon_data_reader import VICONReader
from preprocessing.realsense_data_reader import RealSenseReader


class TestPreprocessing(unittest.TestCase):
    def test_read_data(self):
        REALSENSE_PATH = '../Sub003_Squat01_Front.bag'
        VICON_PATH = '../Sub003_Squat.csv'
        REALSENSE_FRAME_RATE = 30
        VICON_FRAME_RATE = 120

        # Init the VICON reader and read the points.
        vicon_reader = VICONReader(vicon_file_path=VICON_PATH)
        vicon_points = vicon_reader.get_points() # Dictnary of <frame_id, List<Point>>
        current_frame = list(vicon_points.keys())[0] # Get the first frame

        # Init the realsense reader and get the pipeline.
        realsense_reader = RealSenseReader(bag_file_path=REALSENSE_PATH, type='RGB')
        pipeline = realsense_reader.setup_pipeline()

        # Set-up two windows.
        cv2.namedWindow("RGB Stream", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Vicon Stream", cv2.WINDOW_AUTOSIZE)

        # Start playing the videos.
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            realsense_image = np.asanyarray(color_frame.get_data())
            realsense_image = np.rot90(realsense_image, k=1) # If image is rotated by default

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
            cv2.imshow("RGB Stream", realsense_image)
            cv2.imshow("Vicon Stream", vicon_image)
            key = cv2.waitKey(1)
            current_frame = current_frame + (VICON_FRAME_RATE / REALSENSE_FRAME_RATE)



