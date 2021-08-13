import math
import unittest
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import cv2


from preprocessing.vicon_data_reader import VICONReader
from preprocessing.trim_data import rotate_vicon_points_90_degree_counterclockwise


class TestRotation(unittest.TestCase):
    def test_data_without_rotation(self):
        #reader = VICONReader('../preprocessing/trimmed/Sub005/Stand/Front/Sub005_Stand_Front.csv')
        reader = VICONReader( '../../Data/Sub007/Sub007_Vicon/Sub007 Left.csv')
        vicon_points = reader.get_points()

        first_frame_points = list(vicon_points.values())[0]
        self.assertEqual(len(first_frame_points), 39)

        # Open3D
        pcd = o3d.geometry.PointCloud()
        points = np.array([(point.x, point.y, point.z) for point in first_frame_points])
        pcd.points = o3d.utility.Vector3dVector(points)
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()
        visualizer.add_geometry(pcd)
        visualizer.run()
        visualizer.close()

        # OpenCV
        for key in vicon_points.keys():
            # Create an empty image to write the vicon points on in later.
            blank = np.zeros(shape=(800, 800, 3), dtype=np.uint8)
            vicon_image = cv2.cvtColor(blank, cv2.COLOR_RGB2BGR)

            current_points = vicon_points[key]
            for i, point in enumerate(current_points):
                x = point.x
                y = point.y
                z = point.z

                if math.isnan(x) or math.isnan(y) or math.isnan(z):
                    continue

                # Scale the coordinates so they will fit the image.
                x = x / 5
                y = y / 5
                z = z / 5
                # Draw the point on the blank image (orthographic projection).
                vicon_image = cv2.circle(vicon_image, ((int(x) + 300), (int(y) + 300)), radius=0, color=(0, 0, 255),
                                         thickness=10)  # Coordinates offsets are manually selected to center the object.

            cv2.imshow("Vicon Stream", vicon_image)
            cv2.waitKey(1)

    def test_data_with_rotation(self):
        #path = '../preprocessing/trimmed/Sub005/Stand/Front/Sub005_Stand_Front.csv'
        path =  '../../Data/Sub007/Sub007_Vicon/Sub007 Left.csv'
        vicon_points = rotate_vicon_points_90_degree_counterclockwise(csv_path=path, rotation_axis='x')
        #vicon_points = rotate_vicon_points_90_degree_counterclockwise(points=vicon_points, rotation_axis='z')
        #vicon_points = rotate_vicon_points_90_degree_counterclockwise(points=vicon_points, rotation_axis='z')
        first_frame_points = list(vicon_points.values())[0]
        self.assertEqual(len(first_frame_points), 39)

        # Open3D
        pcd = o3d.geometry.PointCloud()
        points = np.array([(point.x, point.y, point.z) for point in first_frame_points])
        pcd.points = o3d.utility.Vector3dVector(points)
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()
        visualizer.add_geometry(pcd)
        visualizer.run()

        # OpenCV
        for key in vicon_points.keys():
            current_points = vicon_points[key]

            # Create an empty image to write the vicon points on in later.
            blank = np.zeros(shape=(800, 800, 3), dtype=np.uint8)
            vicon_image = cv2.cvtColor(blank, cv2.COLOR_RGB2BGR)

            for i, point in enumerate(current_points):
                x = point.x
                y = point.y
                z = point.z

                if math.isnan(x) or math.isnan(y) or math.isnan(z):
                    continue

                # Scale the coordinates so they will fit the image.
                x = x / 5
                y = y / 5
                z = z / 5
                # Draw the point on the blank image (orthographic projection).
                vicon_image = cv2.circle(vicon_image, ((int(x) + 300), (int(y) + 300)), radius=0, color=(0, 0, 255),
                                         thickness=10)  # Coordinates offsets are manually selected to center the object.

            vicon_image = cv2.rotate(vicon_image, cv2.cv2.ROTATE_180)  # OpenCV origin is TOP-LEFT, so image
            # needs to be rotated 180 degrees.

            cv2.imshow("Vicon Stream", vicon_image)
            cv2.waitKey(1)




