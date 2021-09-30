import open3d as o3d
import numpy as np
import cv2


def visualize_vicon_points(points):
    # Open3D
    pcd = o3d.geometry.PointCloud()
    try:
        points = np.array([(point.x, point.y, point.z) for point in points])
    except:
        points = np.array([(point[0], point[1], point[2]) for point in points.values()])
    pcd.points = o3d.utility.Vector3dVector(points)
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(pcd)
    visualizer.run()
    visualizer.close()


def visualize_2d_points_on_frame(frame_image, points, scale=1, transform_x=0, transform_y=0):
    for p in points:
        x = int(int(p[0]) / scale) + int(transform_x)
        y = int(int(p[1]) / scale) + int(transform_y)
        frame_image = cv2.circle(frame_image, (x, y), radius=1, color=(0, 255, 0), thickness=5)

    cv2.imshow("Projected", frame_image)
    cv2.waitKey(0)