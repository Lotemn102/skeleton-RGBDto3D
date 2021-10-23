"""
I've used CVAT for annotating the points in the images. Link to the project: https://cvat.org/projects/14462
This code assumes you have downloaded the annotation in json format. This can be done by going to "Tasks" tab,
clicking on the "Actions" dropdown list, and then clicking on "Export task".
"""
import json

from data_cleaning.vicon_data_reader import KEYPOINTS_NAMES

class CVATReader:
    def __init__(self, json_path, is_calibration=False):
        self.path = json_path
        self.calibration = is_calibration

        # Read data.
        f = open(self.path)
        self.data = json.load(f)

    def get_points(self):
        points_raw = self.data[0]['shapes']
        points = {} # key=keypoint name, value=point.

        for p in points_raw:
            label = p['label']
            point = p['points']
            points[label] = point

        # Sort points according to KEYPOINTS_NAMES
        if not self.calibration: # Subject
            sorted_data = sorted(points.items(), key=lambda pair: KEYPOINTS_NAMES.index(pair[0]))
        else: # Calibration device
            sorted_data = sorted(points.items(), key=lambda pair: ['11', '12', '13', '14', '15', '16', '17', '18', '19', '110',
                                                                   '111', '112', '113', '114', '115', '116', '117', '118',
                                                                   '119', '120', '121', '122', '123', '124'].index(pair[0]))

        points = {}

        for e in sorted_data:
            points[e[0]] = e[1]

        return points

