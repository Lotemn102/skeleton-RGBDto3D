import pyrealsense2 as rs
import numpy as np
import cv2

MIN_DIST = 0
MAX_DIST = 2.6
thr_filter = rs.threshold_filter(min_dist=MIN_DIST, max_dist=MAX_DIST)
colorizer = rs.colorizer()
colorizer.set_option(rs.option.visual_preset, 0) # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
colorizer.set_option(rs.option.min_distance, MIN_DIST)
colorizer.set_option(rs.option.max_distance, MAX_DIST)

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
rs.config.enable_device_from_file(config, "../../Data/Sub013_Squat_Front.bag")
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# Start streaming
profile = pipeline.start(config)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        filtered = thr_filter.process(aligned_depth_frame)
        depth_colormap = np.asanyarray(colorizer.colorize(filtered).get_data())
        depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)
        result = depth_colormap.copy()
        result[np.where((result == [128, 0, 0]).all(axis=2))] = [0, 0, 0]

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', result)
        key = cv2.waitKey(1)
finally:
    pipeline.stop()