import unittest
import cv2
import numpy as np
import math
import os


from data_cleaning.vicon_data_reader import VICONReader
from data_cleaning.realsense_data_reader import RealSenseReader
from data_cleaning.trim_data import sync_30_fps


class TestPreprocessing(unittest.TestCase):
    def test_RGB_and_Vicon_data(self):
        """
       For visual manual evaluation.

       :return: None.
       """
        #REALSENSE_PATH = '/media/lotemn/Other/project-data/frames/Sub002/RealSense/Stand'
        VICON_PATH = '../../annotations_data/Sub008/Front/vicon_points.csv'
        REALSENSE_FRAME_RATE = 30
        VICON_FRAME_RATE = 30

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
            #realsense_reader = RealSenseReader(bag_file_path=REALSENSE_PATH, type='RGB', frame_rate=REALSENSE_FRAME_RATE)
            #pipeline = realsense_reader.setup_pipeline()
            print(30)
        except:
            # Some videos were recorded with FPS of 15.
            #REALSENSE_FRAME_RATE = 15
            #realsense_reader = RealSenseReader(bag_file_path=REALSENSE_PATH, type='RGB', frame_rate=REALSENSE_FRAME_RATE)
            #pipeline = realsense_reader.setup_pipeline()
            print(15)

        # Set-up two windows.
        #cv2.namedWindow("RGB Stream", cv2.WINDOW_AUTOSIZE)
        #cv2.namedWindow("Vicon Stream", cv2.WINDOW_AUTOSIZE)

        # Start playing the videos.
        while True:
            #frames = pipeline.wait_for_frames()
            #color_frame = frames.get_color_frame()
            #realsense_image = np.asanyarray(color_frame.get_data())
            #realsense_image = np.rot90(realsense_image, k=3)
            print(current_frame)

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
            cv2.waitKey(30)
            current_frame = current_frame + (VICON_FRAME_RATE / REALSENSE_FRAME_RATE)

    def test_read_align_data(self):
        # First import the library
        import pyrealsense2 as rs

        # Create a pipeline
        pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        config = rs.config()

        rs.config.enable_device_from_file(config, '/media/lotemn/Transcend/Movement Sense Research/Vicon Validation Study/Sub004/Sub004_Back/Sub004_Left_Back.bag')

        # Set the format & type.
        config.enable_stream(stream_type=rs.stream.color, width=640, height=480, format=rs.format.rgb8, framerate=30)
        config.enable_stream(stream_type=rs.stream.depth, width=848, height=480, format=rs.format.z16, framerate=30)
        profile = pipeline.start(config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        clipping_distance_in_meters = 5  # 1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale

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
                aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
                color_frame = aligned_frames.get_color_frame()

                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                color_image = np.rot90(color_image, k=3)
                depth_image = np.rot90(depth_image, k=3)

                # Remove background - Set pixels further than clipping_distance to grey
                grey_color = 153
                depth_image_3d = np.dstack(
                    (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
                bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color,
                                      color_image)

                # Render images:
                #   depth align to color on left
                #   depth on right
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                images = np.hstack((bg_removed, depth_colormap))

                cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Align Example', images)
                cv2.waitKey(1)
        finally:
            pipeline.stop()

    def test_depth_data(self):
        """
       For visual manual evaluation.

       :return: None.
       """
        REALSENSE_PATH = 'D:/Movement Sense Research/Vicon Validation Study/Sub013/Sub013_Back/Sub013_Left_Back.bag'
        REALSENSE_FRAME_RATE = 30

        # Init the realsense reader and get the pipeline.
        try:
            # Most of the videos were recorded with FPS of 30.
            realsense_reader = RealSenseReader(bag_file_path=REALSENSE_PATH, type='DEPTH', frame_rate=REALSENSE_FRAME_RATE)
            pipeline = realsense_reader.setup_pipeline()
            print(30)
        except:
            # Some videos were recorded with FPS of 15.
            REALSENSE_FRAME_RATE = 15
            realsense_reader = RealSenseReader(bag_file_path=REALSENSE_PATH, type='DEPTH', frame_rate=REALSENSE_FRAME_RATE)
            pipeline = realsense_reader.setup_pipeline()
            print(15)

        # Set-up two windows.
        cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)

        # Start playing the videos.
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_image = np.rot90(depth_image, k=3)
            #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET)
            # Render realsense image and vicon image.
            cv2.imshow("Depth Stream", depth_image)
            cv2.waitKey(2)

    def test_sync(self):
        """
        For visual manual evaluation.

        :return: None.
        """
        FPS = 30
        SUB_NUMBER = '005'
        POSITIONS_LIST = ['Squat', 'Stand', 'Left', 'Right', 'Tight']
        POSITON = POSITIONS_LIST[3]
        FIRST_VIDEO_PATH = '../data_cleaning/trimmed/Sub' + str(SUB_NUMBER) + '/Sub' + str(SUB_NUMBER) + '_' + POSITON + '_Back.avi'
        SECOND_VIDEO_PATH = '../data_cleaning/trimmed/Sub' + str(SUB_NUMBER) + '/Sub' + str(SUB_NUMBER) + '_' + POSITON +  '_Front.avi'
        THIRD_VIDEO_PATH = '../data_cleaning/trimmed/Sub' + str(SUB_NUMBER) + '/Sub' + str(SUB_NUMBER) + '_' + POSITON + '_Side.avi'
        VICON_VIDEO_PATH = '../data_cleaning/trimmed/Sub' + str(SUB_NUMBER) + '/Sub' + str(SUB_NUMBER) + '_' + POSITON +  '_Vicon.avi'

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
        """
        For visual manual evaluation.

        :return: None.
        """
        FPS = 30
        SUB_NUMBER = '005'
        POSITIONS_LIST = ['Squat', 'Stand', 'Left', 'Right', 'Tight']
        ANGLE_LIST = ['Back', 'Front', 'Side']
        POSITON = POSITIONS_LIST[3]
        ANGLE = ANGLE_LIST[0]
        RGB_VIDEO_PATH = '../data_cleaning/trimmed/Sub' + str(SUB_NUMBER) + '/Sub' + str(
            SUB_NUMBER) + '_' + POSITON + '_' + ANGLE + '.avi'
        VICON_VIDEO_PATH = '../data_cleaning/trimmed/Sub' + str(SUB_NUMBER) + '/Sub' + str(
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

            while counter < 4: # Read every 4th frames from vicon, since vicon FPS is 120
                ret_2, frame_2 = cap_2.read()
                counter = counter + 1

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

            counter = 0

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

    def test_trim_perfect_data(self):
        """
        Perfect data.

        :return: None.
        """
        frames_numbers = []

        # Generate 10 fake images.
        for i in range(1, 5):
            frames_numbers.append(str(i) + '.png')

        frames_realsense, frames_vicon = sync_30_fps(bag_shoot_angle='random', sub_name='random',
                                                      sub_position='random', first_frame_number_realsense=1,
                                                      first_frame_number_vicon=1, test_realsense=frames_numbers,
                                                      test_csv='../unittests/assets/test1.csv')
        self.assertEqual(len(frames_realsense), 4)
        self.assertEqual(len(frames_vicon), 16)

    def test_trim_missing_realsense(self):
        """
        Missing realsense frames.

        :return: None.
        """
        frames_numbers = []

        # Generate 10 fake images.
        for i in range(1, 5):
            frames_numbers.append(str(i) + '.png')

        frames_numbers.remove('3.png')

        frames_realsense, frames_vicon = sync_30_fps(bag_shoot_angle='random', sub_name='random',
                                                      sub_position='random', first_frame_number_realsense=1,
                                                      first_frame_number_vicon=1, test_realsense=frames_numbers,
                                                      test_csv='../unittests/assets/test1.csv')
        self.assertEqual(len(frames_realsense), 3)
        self.assertEqual(len(frames_vicon), 12)

    def test_trim_missing_vicon(self):
        """
        Missing vicon frames.

        :return: None.
        """

        frames_numbers = []

        # Generate 10 fake images.
        for i in range(1, 5):
            frames_numbers.append(str(i) + '.png')

        frames_realsense, frames_vicon = sync_30_fps(
            bag_shoot_angle='random', sub_name='random',
            sub_position='random', first_frame_number_realsense=1,
            first_frame_number_vicon=1, test_realsense=frames_numbers,
            test_csv='../unittests/assets/test2.csv')
        self.assertEqual(len(frames_realsense), 3)
        self.assertEqual(len(frames_vicon), 12)

    def test_trim_missing_realsense_and_vicon(self):
        """
        Missing both realsense and vicon frames.

        :return: None.
        """

        frames_numbers = []

        # Generate 10 fake images.
        for i in range(1, 5):
            frames_numbers.append(str(i) + '.png')

        frames_numbers.remove('4.png')

        frames_realsense, frames_vicon = sync_30_fps(
            bag_shoot_angle='random', sub_name='random',
            sub_position='random', first_frame_number_realsense=1,
            first_frame_number_vicon=1, test_realsense=frames_numbers,
            test_csv='../unittests/assets/test2.csv')
        self.assertEqual(len(frames_realsense), 2)
        self.assertEqual(len(frames_vicon), 8)

    def test_depth_vs_aligned_depth(self):
        # First import the library
        import pyrealsense2 as rs

        # Create a pipeline
        pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        config = rs.config()

        rs.config.enable_device_from_file(config,
                                          '/media/lotemn/Transcend/Movement Sense Research/Vicon Validation Study/Sub004/Sub004_Back/Sub004_Left_Back.bag')

        # Set the format & type.
        config.enable_stream(stream_type=rs.stream.color, width=640, height=480, format=rs.format.rgb8, framerate=30)
        config.enable_stream(stream_type=rs.stream.depth, width=848, height=480, format=rs.format.z16, framerate=30)
        profile = pipeline.start(config)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)

        # Streaming loop
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Read color frames.
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            color_image = np.rot90(color_image, k=3)

            # Read depth frames.
            '''
            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_image = np.rot90(depth_image, k=3)
            depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),
                                                    cv2.COLORMAP_JET)
            '''

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
            aligned_depth_image = np.rot90(aligned_depth_image, k=3)

            print(color_frame.frame_number, aligned_depth_frame.frame_number)

            # Render images:
            #   depth align to color on left
            #   depth on right
            aligned_depth_image = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_image, alpha=0.03), cv2.COLORMAP_JET)

            cv2.namedWindow('Align', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Color', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Align', aligned_depth_image)
            #cv2.imshow('Depth', depth_image)
            cv2.imshow('Color', color_image)
            cv2.waitKey(1)

    def test_sync_all_videos(self):
        REALSENSE_FPS = 30
        VICON_FPS = 30

        for i in range(4, 5):
            subject_name = 'Sub00' + str(i) if i < 10 else 'Sub0' + str(i)

            for position in ['Stand', 'Squat', 'Tight', 'Left', 'Right']:

                for angle in ['Front', 'Back', 'Side']:

                    if subject_name != 'Sub004' or position != 'Stand' or angle != 'Front':
                        continue

                    print(subject_name + ", " + position + ", " + angle)

                    RGB_PATH = '../data_cleaning/trimmed/' + subject_name + '/' + position + '/' + angle + '/' +  \
                               subject_name + '_' + position + '_' + angle +'_RGB.avi'
                    DEPTH_PATH = '../data_cleaning/trimmed/' + subject_name + '/' + position + '/' + angle + '/' +  \
                               subject_name + '_' + position + '_' + angle +'_Depth.avi'
                    CSV_PATH =  '../data_cleaning/trimmed/' + subject_name + '/' + position + '/' + angle + '/' \
                               + subject_name + '_' + position + '_' + angle +'.csv'

                    # Init the VICON reader and read the points.
                    try:
                        vicon_reader = VICONReader(vicon_file_path=CSV_PATH)
                        vicon_points = vicon_reader.get_points()  # Dictionary of <frame_id, List<Point>>
                    except:
                        continue
                    index = 0

                    cap_1 = cv2.VideoCapture(RGB_PATH)
                    cap_2 = cv2.VideoCapture(DEPTH_PATH)

                    # Set-up two windows.
                    cv2.namedWindow("RGB", cv2.WINDOW_AUTOSIZE)
                    cv2.namedWindow("VICON", cv2.WINDOW_AUTOSIZE)
                    cv2.namedWindow("DEPTH", cv2.WINDOW_AUTOSIZE)
                    cv2.moveWindow("RGB", 0, 0, )
                    cv2.moveWindow("VICON", 700, 0, )
                    cv2.moveWindow("DEPTH", 0, 700, )

                    first_iteration = True
                    counter = 0

                    while cap_1.isOpened():
                        ret_1, frame_1 = cap_1.read()
                        ret_2, frame_2 = cap_2.read()
                        frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY) # The depth video is in grayscale

                        if index >= len(list(vicon_points.keys())):
                            break

                        # Create an empty image to write the vicon points on in later.
                        current_frame = list(vicon_points.keys())[index]
                        blank = np.zeros(shape=(640, 480, 3), dtype=np.uint8)
                        vicon_image = cv2.cvtColor(blank, cv2.COLOR_RGB2BGR)
                        current_frame_points = vicon_points[current_frame]

                        for i, point in enumerate(current_frame_points):
                            x = point.x
                            y = point.y
                            z = point.z

                            if math.isnan(x) or math.isnan(y) or math.isnan(z):
                                # Skip this point for the moment
                                continue

                            # Scale the coordinates so they will fit the image.
                            x = x / 5
                            y = y / 5
                            z = z / 5
                            # Draw the point on the blank image (orthographic projection).
                            vicon_image = cv2.circle(vicon_image, ((int(x) + 170), (int(z) + 120)), radius=0,
                                                     color=(0, 0, 255),
                                                     thickness=10)  # Coordinates offsets are manually selected to center the object.
                        index += int(VICON_FPS / REALSENSE_FPS)

                        if first_iteration:
                            scale_percent = 90
                            width_realsense = int(frame_1.shape[1] * scale_percent / 100)
                            height_realsense = int(frame_1.shape[0] * scale_percent / 100)
                            dims_realsense = (width_realsense, height_realsense)
                            first_iteration = False

                        frame_1 = cv2.resize(frame_1, dims_realsense, interpolation=cv2.INTER_AREA)
                        frame_2 = cv2.resize(frame_2, dims_realsense, interpolation=cv2.INTER_AREA)
                        vicon_image = cv2.rotate(vicon_image, cv2.cv2.ROTATE_180) # OpenCV origin is TOP-LEFT, so image
                        # needs to be rotated 180 degrees.
                        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(frame_2, alpha=0.03),
                                                           cv2.COLORMAP_JET)

                        img_1 = cv2.cvtColor(frame_1, cv2.COLOR_RGB2BGR)
                        img_2 = cv2.cvtColor(depth_colormap, cv2.COLOR_RGB2BGR)
                        cv2.imshow('RGB', img_1)
                        cv2.imshow('DEPTH', img_2)
                        cv2.imshow('VICON', vicon_image)
                        cv2.waitKey(REALSENSE_FPS)
                        counter = counter + 1

    def test_sync_all_frames(self):
        REALSENSE_FPS = 30
        VICON_FPS = 30

        for i in range(4, 5):
            subject_name = 'Sub00' + str(i) if i < 10 else 'Sub0' + str(i)

            for position in ['Stand', 'Squat', 'Tight', 'Left', 'Right']:

                for angle in ['Front', 'Back', 'Side']:

                    if subject_name != 'Sub004' or position != 'Stand' or angle != 'Front':
                        continue

                    print(subject_name + ", " + position + ", " + angle)

                    RGB_PATH = '../data_cleaning/trimmed/' + subject_name + '/' + position + '/' + angle + '/rgb_frames/'
                    DEPTH_PATH = '../data_cleaning/trimmed/' + subject_name + '/' + position + '/' + angle + '/depth_frames/'
                    CSV_PATH =  '../data_cleaning/trimmed/' + subject_name + '/' + position + '/' + angle + '/' \
                               + subject_name + '_' + position + '_' + angle +'.csv'

                    # Init the VICON reader and read the points.
                    try:
                        vicon_reader = VICONReader(vicon_file_path=CSV_PATH)
                        vicon_points = vicon_reader.get_points()  # Dictionary of <frame_id, List<Point>>
                    except:
                        continue
                    index = 0

                    # Set-up two windows.
                    cv2.namedWindow("RGB", cv2.WINDOW_AUTOSIZE)
                    cv2.namedWindow("VICON", cv2.WINDOW_AUTOSIZE)
                    cv2.namedWindow("DEPTH", cv2.WINDOW_AUTOSIZE)
                    cv2.moveWindow("RGB", 0, 0, )
                    cv2.moveWindow("VICON", 700, 0, )
                    cv2.moveWindow("DEPTH", 0, 700, )

                    counter = 0

                    # All rgb frames
                    all_frames_files_realsense_rgb = os.listdir(RGB_PATH)
                    all_frames_files_realsense_rgb.remove('log.json')
                    all_frames_files_realsense_rgb = sorted(all_frames_files_realsense_rgb, key=lambda x: int(x[:-4]))

                    # All depth frames
                    all_frames_files_realsense_depth = os.listdir(DEPTH_PATH)
                    all_frames_files_realsense_depth.remove('log.json')
                    all_frames_files_realsense_depth = sorted(all_frames_files_realsense_depth, key=lambda x: int(x[:-4]))

                    while 1:

                        if index >= len(list(vicon_points.keys())):
                            break

                        # Create an empty image to write the vicon points on in later.
                        current_frame = list(vicon_points.keys())[index]
                        blank = np.zeros(shape=(640, 480, 3), dtype=np.uint8)
                        vicon_image = cv2.cvtColor(blank, cv2.COLOR_RGB2BGR)
                        current_frame_points = vicon_points[current_frame]

                        for i, point in enumerate(current_frame_points):
                            x = point.x
                            y = point.y
                            z = point.z

                            if math.isnan(x) or math.isnan(y) or math.isnan(z):
                                # Skip this point for the moment
                                continue

                            # Scale the coordinates so they will fit the image.
                            x = x / 5
                            y = y / 5
                            z = z / 5
                            # Draw the point on the blank image (orthographic projection).
                            vicon_image = cv2.circle(vicon_image, ((int(x) + 170), (int(z) + 120)), radius=0,
                                                     color=(0, 0, 255),
                                                     thickness=10)  # Coordinates offsets are manually selected to center the object.

                        vicon_image = cv2.rotate(vicon_image, cv2.cv2.ROTATE_180) # OpenCV origin is TOP-LEFT, so image
                        rgb_image = cv2.imread(RGB_PATH + '/' + all_frames_files_realsense_rgb[index])
                        depth_image = np.fromfile(DEPTH_PATH + '/' + all_frames_files_realsense_depth[index], dtype='int16', sep="")
                        depth_image = depth_image.reshape([848, 480])
                        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),
                                                           cv2.COLORMAP_JET)

                        cv2.imshow('RGB', rgb_image)
                        cv2.imshow('DEPTH', depth_colormap)
                        cv2.imshow('VICON', vicon_image)
                        cv2.waitKey(REALSENSE_FPS)
                        counter = counter + 1
                        index += 1



