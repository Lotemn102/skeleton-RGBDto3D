"""
Check if we can use the vicon data in 30 FPS instead of 120 FPS.
"""
import math
import time
from typing import List
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os


from preprocessing.structs import Point
from preprocessing.vicon_data_reader import VICONReader
from preprocessing.trim_data import rotate_vicon_points_90_degree_counterclockwise


def calc_angle(p1: Point, p2: Point, p3: Point) -> float: # p1 is RFHD, pt2 is C7, p3 is RSHO
    # point2-point1
    vector1 = np.asarray([p2.x-p1.x, p2.y-p1.y, p2.z-p1.z])

    # point2-point3
    vector2 = np.asarray([p2.x-p3.x, p2.y-p3.y, p2.z-p3.z])

    angle = np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
    return float(np.degrees(angle))

def calc_average_angle_per_second(vicon_csv_path: str) -> (List, List):
    RFHD_index = 1
    C7_index = 4
    RSHO_index = 16

    reader = VICONReader(vicon_file_path=vicon_csv_path)
    vicon_points = reader.get_points()

    angles_120 = []
    angles_30 = []
    nans_counter = 0

    # Calculate each angle per frame, 120 FPS
    for frame in vicon_points.keys():
        points = vicon_points[frame]
        RFHD = points[RFHD_index]
        C7 = points[C7_index]
        RSHO = points[RSHO_index]

        angle = calc_angle(RFHD, C7, RSHO)

        if not math.isnan(angle):
            angles_120.append(angle)
        else:
            nans_counter += 1

    # Calculate each angle per frame, 30 FPS
    nans_counter = 0

    counter = 0
    for frame in vicon_points.keys():

        if counter % 4 == 0:
            points = vicon_points[frame]
            RFHD = points[RFHD_index]
            C7 = points[C7_index]
            RSHO = points[RSHO_index]

            angle = calc_angle(RFHD, C7, RSHO)

            if not math.isnan(angle):
                angles_30.append(angle)
            else:
                nans_counter += 1

        counter = counter + 1

    #print("Average 120: " + str(np.mean(angles_120)))
    #print("Average 30: " + str(np.mean(angles_30)))

    # Calculate average angle per second, 120 FPS
    average_angle_per_second_120 = []

    for i in range(len(angles_120) // 120):
        sub_list = []

        for j in range(120):
            idx = i*120+j
            sub_list.append(angles_120[i*120+j])

        assert len(sub_list) == 120
        average_angle_per_second_120.append(np.mean(sub_list))

    # Calculate average angle per second, 30 FPS
    average_angle_per_second_30 = []

    for i in range(len(angles_30) // 30):
        sub_list = []

        for j in range(30):
            sub_list.append(angles_30[i*30 + j])

        assert len(sub_list) == 30
        average_angle_per_second_30.append(np.mean(sub_list))

    return average_angle_per_second_30, average_angle_per_second_120

def visualize(vicon_csv_path: str, average_angle_per_second_120: List, average_angle_per_second_30: List):
    RFHD_index = 1
    C7_index = 4
    RSHO_index = 16
    WINDOW_WIDTH = 1920  # change this if needed
    WINDOW_HEIGHT = 1080  # change this if needed

    average_120_iter = iter(average_angle_per_second_120)
    average_30_iter = iter(average_angle_per_second_30)

    reader = VICONReader(vicon_file_path=vicon_csv_path)
    vicon_points = reader.get_points()

    # Rotate points for better visualizing.
    rotated_points = rotate_vicon_points_90_degree_counterclockwise(points=vicon_points, rotation_axis='x')

    # Init OpenCV window for the angles.
    #cv2.namedWindow("Angles", cv2.WINDOW_AUTOSIZE)
    #blank = np.zeros(shape=(100, 100, 3), dtype=np.uint8)
    #angle_image = cv2.cvtColor(blank, cv2.COLOR_RGB2BGR)

    # Init visualizer.
    frame_points = rotated_points[list(rotated_points.keys())[0]]
    pcd = o3d.geometry.PointCloud()
    points = np.array([(point.x, point.y, point.z) for point in frame_points])
    pcd.points = o3d.utility.Vector3dVector(points)

    # Color all points in black, except the 3 points used for angle calc.
    points_mask = np.zeros(shape=(len(points), 3))
    colors = np.zeros_like(points_mask)
    colors[RSHO_index] = [1, 0, 0]
    colors[C7_index] = [1, 0, 0]
    colors[RFHD_index] = [1, 0, 0]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Add points
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=WINDOW_WIDTH, height=WINDOW_HEIGHT, window_name='Vicon Points')
    visualizer.add_geometry(pcd)

    # Add vectors.
    lines = [[C7_index, RSHO_index], [C7_index, RFHD_index]]
    line_set = o3d.geometry.LineSet()
    p1 = frame_points[C7_index]
    p2 = frame_points[RFHD_index]
    p3 = frame_points[RSHO_index]
    points = np.array([(p1.x, p1.y, p1.z), (p2.x, p2.y, p2.z), (p3.x, p3.y, p3.z)])
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0], [0, 1, 0]])
    visualizer.add_geometry(line_set)
    #cv2.imshow('Angles', angle_image)

    start = time.time()
    rotation_angle = 0

    try:
        for index, frame in enumerate(rotated_points.keys()):

            if index == 0 or (index % 2 == 0 and index % 150 != 0):
                continue

            # Clear all previous points.
            pcd.clear()
            line_set.clear()

            # Read new points.
            frame_points = rotated_points[frame]
            points = np.array([(point.x, point.y, point.z) for point in frame_points])
            pcd.points = o3d.utility.Vector3dVector(points)

            # Color all points in black, except the 3 points used for angle calc.
            points_mask = np.zeros(shape=(len(points), 3))
            colors = np.zeros_like(points_mask)
            colors[RSHO_index] = [1, 0, 0]
            colors[C7_index] = [1, 0, 0]
            colors[RFHD_index] = [1, 0, 0]
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Add new points.
            visualizer.add_geometry(pcd)

            # Add vectors.
            line_set = o3d.geometry.LineSet()
            p1 = frame_points[C7_index]
            p2 = frame_points[RFHD_index]
            p3 = frame_points[RSHO_index]
            points = np.array([(p1.x, p1.y, p1.z), (p2.x, p2.y, p2.z), (p3.x, p3.y, p3.z)])
            lines = [[0, 1], [0, 2]]
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0], [0, 1, 0]])
            visualizer.add_geometry(line_set)

            # Rotate the object.
            ctr = visualizer.get_view_control()
            rotation_angle += 1
            ctr.rotate(x=rotation_angle, y=0)

            # Update the angles text.
            if index % 150 == 0:
                blank = np.zeros(shape=(100, 300, 3), dtype=np.uint8)
                angle_image = cv2.cvtColor(blank, cv2.COLOR_RGB2BGR)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.7
                fontColor = (255, 255, 0)
                lineType = 1
                cv2.putText(angle_image, "120 FPS angle: " + str(next(average_120_iter)),
                            (0, 30),
                            font,
                            fontScale,
                            fontColor,
                            lineType)
                cv2.putText(angle_image, "30 FPS angle: " + str(next(average_30_iter)),
                            (0, 60),
                            font,
                            fontScale,
                            fontColor,
                            lineType)

            visualizer.run()
            #cv2.imshow('Angles', angle_image)
            #cv2.waitKey(1)
    except StopIteration:
        pass

    end = time.time()
    print(end - start)

def plot(average_angle_per_second_120: List, average_angle_per_second_30: List, position: str, sub_name: str):
    if len(average_angle_per_second_30) != len(average_angle_per_second_120):
        #print("oh no")
        return

    # Create average angle plot
    x = list(range(1, len(average_angle_per_second_30)+1))
    sns.set_style("darkgrid")
    plt.plot(x, average_angle_per_second_30, label="30 FPS")
    plt.plot(x, average_angle_per_second_120, label="120 FPS")
    plt.title("Average angle per second in 120 FPS and 30 FPS (" + position + ")")
    plt.xlabel("Second")
    plt.ylabel("Angle")
    plt.legend()
    save_path = 'plots/angles_diff/' + sub_name + "/"

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    #plt.savefig(save_path + position + ".png")
    plt.close()

    # Create diff plot
    x = list(range(1, len(average_angle_per_second_30) + 1))
    y = [np.abs(average_angle_per_second_30[i]-average_angle_per_second_120[i]) for i in range(len(average_angle_per_second_30))]
    sns.set_style("darkgrid")
    plt.plot(x, y)
    plt.title("Difference in average angle per second in 120 FPS and 30 FPS (" + position + ")")
    plt.xlabel("Second")
    plt.ylabel("Difference")
    save_path = 'plots/angles_diff/' + sub_name + "/"

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    #plt.savefig(save_path + position + "_diff.png")
    plt.close()

    max_diff = max(y)
    mean_diff = np.mean(y)

    return (max_diff, mean_diff)

def generate_plots_for_all():
    max_diff_list = []
    mean_diff_list = []

    for root, dirs, files in os.walk("/media/lotemn/Transcend/Movement Sense Research/Vicon Validation Study"):
        for file in files:
            if file.endswith(".csv"):
                if 'Sub001' in file or 'Sub002' in file or 'Sub003' in file or 'Sub006' in file:
                    continue

                if 'without' in file or 'Cal' in file:
                    continue

                remove_extension = file[:-4]
                splitted = remove_extension.split('_')

                if len(splitted) == 1:
                    splitted = remove_extension.split(' ')

                subject_name = [e for e in splitted if 'Sub' in e][0]

                for e in splitted:
                    if 'squat' in e.lower():
                        subject_position = e
                        break
                    elif 'stand' in e.lower():
                        subject_position = e
                        break
                    elif 'left' in e.lower():
                        subject_position = e
                        break
                    elif 'right' in e.lower():
                        subject_position = e
                        break
                    elif 'tight' in e.lower():
                        subject_position = e
                        break

                #print("Working on " + subject_name + ", " + subject_position)
                path = root + "/" + file
                average_angle_per_second_30, average_angle_per_second_120 = calc_average_angle_per_second(path)

                res = plot(average_angle_per_second_120=average_angle_per_second_120, average_angle_per_second_30=average_angle_per_second_30,
                        position=subject_position, sub_name=subject_name)

                if res is None:
                    continue

                max_diff, mean_diff = res

                if max_diff is None:
                    continue
                elif math.isnan(max_diff):
                    continue
                else:
                    max_diff_list.append(max_diff)

                if mean_diff is None:
                    continue
                elif math.isnan(mean_diff):
                    continue
                else:
                    mean_diff_list.append(mean_diff)

    print(len(max_diff_list))
    print(len(mean_diff_list))

    sns.set_style("darkgrid")
    counts, bins = np.histogram(max_diff_list)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.title("Max difference in average angle per second between 120 FPS and 30 FPS")
    plt.xlabel("Max angle difference")
    plt.ylabel("Sessions count")
    #plt.savefig("max_diff_dist.png")
    plt.close()

    sns.set_style("darkgrid")
    counts, bins = np.histogram(mean_diff_list)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.title("Mean difference in average angle per second between 120 FPS and 30 FPS")
    plt.xlabel("Mean angle difference")
    plt.ylabel("Sessions count")
    #plt.savefig("mean_diff_dist.png")

if __name__ == "__main__":
    vicon_csv_path = '../../Data/Sub004 Left Leg.csv'
    #average_angle_per_second_30, average_angle_per_second_120 = calc_average_angle_per_second(vicon_csv_path)

    #visualize(vicon_csv_path=vicon_csv_path, average_angle_per_second_120=average_angle_per_second_120,
    #          average_angle_per_second_30=average_angle_per_second_30)
    generate_plots_for_all()

