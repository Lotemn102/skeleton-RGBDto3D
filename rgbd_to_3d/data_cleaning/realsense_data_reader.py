"""
Read bag file. See usage examples in the comment in the bottom of this file.
"""

import pyrealsense2 as rs


class RealSenseReader:
    def __init__(self, bag_file_path: str, type: str, frame_rate: int = 30):
        """
        Initialize new real sense data object.

        :param bag_file_path: Path to the bag file.
        :param type: "DEPTH" or "RGB" or "BOTH"
        :param frame_rate: The frame rate of the realsense camera.
        """
        self.bag_file_path = bag_file_path
        self.type = type
        self.frame_rate = frame_rate

    def setup_pipeline(self):
        """
        Initialize the pipeline for reading real sense bag files.

        :return: Pipeline.
        """
        # Set the pipe.
        pipeline = rs.pipeline()
        config = rs.config()

        # Read the data.
        rs.config.enable_device_from_file(config, self.bag_file_path)

        # Set the format & type.
        if self.type != 'BOTH':
            format = rs.format.rgb8 if self.type == 'RGB' else rs.format.z16
            type = rs.stream.color if self.type == 'RGB' else rs.stream.depth
            width = 640 if self.type == 'RGB' else 848

            if type == rs.stream.depth and self.frame_rate == 15:
                width = 640

            config.enable_stream(stream_type=type, width=width, height=480, format=format,
                                 framerate=self.frame_rate)
        else:
            format_1 = rs.format.rgb8
            type_1 = rs.stream.color
            width_1 = 640

            format_2 = rs.format.z16
            type_2 = rs.stream.depth
            width_2 = 848

            config.enable_stream(stream_type=type_1, width=width_1, height=480, format=format_1,
                                 framerate=self.frame_rate)
            config.enable_stream(stream_type=type_2, width=width_2, height=480, format=format_2,
                                 framerate=self.frame_rate)

        pipeline.start(config)
        return pipeline, config

"""
Usage examples
==============

1. RGB
=======================================================================

# Init the class
realsense_data = RealSenseData(PATH, 'RGB')

# Get pipeline
pipeline = realsense_data.setup_pipeline()

# Create opencv window to render image in
cv2.namedWindow("RGB Stream", cv2.WINDOW_AUTOSIZE)

# Streaming loop
while True:
    # Get frameset
    frames = pipeline.wait_for_frames()

    # Get color frame
    color_frame = frames.get_color_frame()

    # Convert color_frame to numpy array to render image in opencv
    color_image = np.asanyarray(color_frame.get_data())
    color_image = np.rot90(color_image, k=3)

    color_colormap_dim = color_image.shape

    # Render image in opencv window
    cv2.imshow("RGB Stream", color_image)
    key = cv2.waitKey(1)        
        
2. DEPTH
============================================================================
# Init the class
realsense_data = RealSenseData(PATH, 'DEPTH')

# Get pipeline
pipeline = realsense_data.setup_pipeline()

# Create opencv window to render image in
cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)

# Create colorizer object
colorizer = rs.colorizer()

# Streaming loop
while True:
    # Get frameset of depth
    frames = pipeline.wait_for_frames()

    # Get depth frame
    depth_frame = frames.get_depth_frame()

    # Colorize depth frame to jet colormap
    depth_color_frame = colorizer.colorize(depth_frame)

    # Convert depth_frame to numpy array to render image in opencv
    depth_color_image = np.asanyarray(depth_color_frame.get_data())

    # Render image in opencv window
    cv2.imshow("Depth Stream", depth_color_image)
    key = cv2.waitKey(1)
"""
