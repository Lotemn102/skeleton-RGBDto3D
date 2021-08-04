import pandas as pd
import json

from preprocessing.structs import Point


KEYPOINTS_NAMES = ['LFHD', 'RFHD', 'LBHD', 'RBHD', 'C7', 'T10', 'CLAV', 'STRN', 'RBAK', 'LSHO', 'LUPA', 'LELB',
                   'LFRM', 'LWRA', 'LWRB', 'LFIN', 'RSHO', 'RUPA', 'RELB', 'RFRM', 'RWRA', 'RWRB', 'RFIN', 'LASI',
                   'RASI', 'LPSI', 'RPSI', 'LTHI', 'LKNE', 'LTIB', 'LANK', 'LHEE', 'LTOE', 'RTHI', 'RKNE', 'RTIB',
                   'RANK', 'RHEE', 'RTOE']
KEYPOINTS_NUM = 39


class VICONReader:
    def __init__(self, vicon_file_path: str):
        """
        Read vicon points from csv file.

        Expecting the csv file structure is as following:
        - N rows, each row represents a single frame.
        - 119 columns. The first 2 columns are "Frame", which represents the frame id, and "Sub Frame", i don't know
         what it represents, but it's always 0. The rest of the 117 columns are the points. There are 39 skeleton-points,
         each has 3 coordinates (3*39=117). The coordinates are being read in the order of X, Y, Z.

        :param vicon_file_path: Path to the csv file.
        """

        self.vicon_file_path = vicon_file_path
        self.csv_data = pd.read_csv(vicon_file_path, skiprows=3)
        self.points_per_frame_map = {}

        for index, row in self.csv_data.iterrows():
            if index < 2: # First 2 rows are empty
                continue

            frame = row[0] # First column holds the frame's id.
            points = []

            for i in range(2, 119, 3): # Skip the 2 first columns, and then read each point. See my comment above about
                # the csv file structure.
                point = Point(float(row[i]), float(row[i+1]), float(row[i+2])) # x, y, z
                points.append(point)
                self.points_per_frame_map[int(frame)] = points

    def get_points(self):
        return self.points_per_frame_map

    def to_json(self):
        filename = self.vicon_file_path[:-3] # Remove the 'csv' ending.
        filename = filename + "json"

        with open(filename, 'w') as f:
            data = []

            for _, v in self.points_per_frame_map.items():
                data.append(v)

            json.dump(data, f, indent=4)

"""
Usage example

# Init the reader.
vicon_reader = VICONReader(vicon_file_path=VICON_PATH)

# Get the points.
vicon_points = vicon_reader.get_points() # Dictionary of <frame_id, List<Point>>

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
        x = x / 5
        z = z / 5
        # Draw the point on the blank image (orthographic projection).
        vicon_image = cv2.circle(vicon_image, ((int(z) + 300), (int(x) + 400)), radius=0, color=(0, 0, 255),
                                 thickness=10) # Coordinates offsets are manually selected to center the object.
    
    # Rotate the image, since the vicon points are also rotated by default.
    vicon_image = cv2.rotate(vicon_image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE) 
    
    # Render realsense image and vicon image.
    cv2.imshow("Vicon Stream", vicon_image)
    cv2.waitKey(1)
"""