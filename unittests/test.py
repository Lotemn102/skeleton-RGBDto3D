import os
import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image


from preprocessing.realsense_data_reader import RealSenseReader

folder_path = '../../Data/Sub005_Left_Back.bag'

# Create align object to align depth frames to RGB frames.
align_to = rs.stream.color
align = rs.align(align_to)

save_path = "../../Data/my_test.erf"
i = 0
aligned_depth_image = None

# Read single depth image and save it.
while i < 1:
    # Get frameset.
    REALSENSE_FPS = 30
    realsense_reader = RealSenseReader(bag_file_path=folder_path, type='DEPTH', frame_rate=REALSENSE_FPS)
    pipeline = realsense_reader.setup_pipeline()
    frames = pipeline.wait_for_frames()

    # Align depth frames to the color ones. The RealSense has 2 sensors: RGB & depth. They are not 100% aligned
    # by default.
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
    aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data(), dtype=np.uint16)
    aligned_depth_image = np.rot90(aligned_depth_image, k=3)

    # Save image.
    aligned_depth_image.astype('int16').tofile('../../Data/my_test.raw')
    i += 1

# Load image and see if it's the same
A = np.fromfile('../../Data/my_test.raw', dtype='int16', sep="")
A = A.reshape([848, 480])
y = (A == aligned_depth_image).all()
print((A == aligned_depth_image).all())
cv2.namedWindow('bla')
cv2.imshow('bla', A)
cv2.waitKey()

x = 2

