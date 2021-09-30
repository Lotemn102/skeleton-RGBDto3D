"""
I've used CVAT for annotating the points in the images. Link to the project: https://cvat.org/projects/14462
This code assumes you have downloaded the annotation in json format. This can be done by going to "Tasks" tab,
clicking on the "Actions" dropdown list, and then clicking on "Export task".
"""
import json

from data_cleaning.vicon_data_reader import KEYPOINTS_NAMES

class CVATReader:
    def __init__(self, json_path):
        self.path = json_path

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
        sorted_data = sorted(points.items(), key=lambda pair: KEYPOINTS_NAMES.index(pair[0]))

        points = {}

        for e in sorted_data:
            points[e[0]] = e[1]

        return points

