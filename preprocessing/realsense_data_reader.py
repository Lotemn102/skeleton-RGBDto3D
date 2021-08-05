import pyrealsense2 as rs


class RealSenseReader:
    def __init__(self, bag_file_path: str, type: str, frame_rate: int = 30):
        """
        Initialize new real sense data object.

        :param bag_file_path: Path to the bag file.
        :param type: "DEPTH" or "RGB"
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
        format = rs.format.rgb8 if self.type == 'RGB' else rs.format.z16
        type = rs.stream.color if self.type == 'RGB' else rs.stream.depth

        config.enable_stream(stream_type=type, format=format, framerate=self.frame_rate)
        pipeline.start(config)
        return pipeline

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
