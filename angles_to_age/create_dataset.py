import os
import numpy as np
import random
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import open3d as o3d
import cv2
from matplotlib.ticker import MaxNLocator

from rgbd_to_3d.data_cleaning.vicon_data_reader import VICONReader, KEYPOINTS_NAMES

SUBJECT_AGES = [24, 26, 27, 27, 27, 25, 27, 30, 23, 27, 26, 27, 23, 26, 26, 27, 85, 72, 77, 72, 66, 74, 67]

def visualize(data, file_name, age):
    vid_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    output = cv2.VideoWriter(file_name + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 120, (500, 500))

    for frame_number, points in enumerate(data):
        points = points[0]
        # Create an empty image to write the vicon points on in later.
        blank = np.zeros(shape=(500, 500, 3), dtype=np.uint8)
        vicon_image = cv2.cvtColor(blank, cv2.COLOR_RGB2BGR)

        for i, point in enumerate(points):
            x = point[0]
            y = point[1]
            z = point[2]

            if math.isnan(x) or math.isnan(y) or math.isnan(z):
                # Skip this point for the moment
                continue

            # Scale the coordinates so they will fit the image.
            SCALE_FACTOR = 0.2
            x = SCALE_FACTOR * x
            z = SCALE_FACTOR * z
            # Draw the point on the blank image.
            vicon_image = cv2.circle(vicon_image, ((int(z) + 100), (int(x) + 200)), radius=0, color=(0, 0, 255),
                                     thickness=10)  # Coordinates offsets are manually selected to center the object.

        # Draw center of mass
        center = find_center_of_mass(points)
        x = SCALE_FACTOR * center[0]
        z = SCALE_FACTOR * center[2]

        if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
            vicon_image = cv2.circle(vicon_image, ((int(z) + 100), (int(x) + 200)), radius=0, color=(0, 255, 0),
                                     thickness=10)  # Coordinates offsets are manually selected to center the object.

        vicon_image = cv2.rotate(vicon_image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)  # The vicon points are also rotated
        # by default.

        org_1 = (20, 20)
        org_2 = (20, 50)
        fontScale = 0.7
        color = (255, 255, 255)
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        vicon_image = cv2.putText(vicon_image, file_name, org_1, font,
                            fontScale, color, thickness, cv2.LINE_AA)
        vicon_image = cv2.putText(vicon_image, "Age: " + str(age), org_2, font,
                            fontScale, color, thickness, cv2.LINE_AA)
        output.write(vicon_image)
        cv2.imshow("Vicon Stream", vicon_image)
        cv2.waitKey(1)

    vid_capture.release()
    output.release()

def plot_center_of_mass_diff(data):
    # Split points into files
    points_files_map = {}

    for i, p in enumerate(data):
        file_name = p[2]

        if file_name not in points_files_map.keys():
            points_files_map[file_name] = []

        points_files_map[file_name].append(p)

    x = []
    y = []

    for i in range(4, 22):
        x.append([])
        y.append([])

    sns.set_style("darkgrid")
    color_list = ['blue', 'black', 'olive', 'chocolate', 'm', 'gold', 'c', 'red', 'darkblue', 'orange', 'pink',
                  'crimson', 'lime', 'gray', 'purple', 'g', 'saddlebrown', 'deepskyblue']
    colors = plt.cycler('color', color_list)
    plt.rc('axes', prop_cycle=colors)
    plt.title("Difference in center of mass from first frame")
    plt.xlabel("frame number")
    plt.ylabel("difference (in mm)")

    # For each file, plot center of mass diff.
    for j, (k, v) in enumerate(points_files_map.items()): # v is (N, 39, 3), N is the number of frames in this file
        # visualize(v, k, v[0][1])

        centroid_points_of_current_file = []

        for points in v: # points is (39, 3)
            center_of_mass = find_center_of_mass(points[0])
            centroid_points_of_current_file.append(center_of_mass)

        frames_gap = 1
        length = len(centroid_points_of_current_file)

        START = int(length / 3)
        END = length - int(length / 3)
        first_center_of_mass = centroid_points_of_current_file[START]
        i = START + 1

        while math.isnan(first_center_of_mass[0]):
            first_center_of_mass = centroid_points_of_current_file[i]
            i = i + 1

        for i in range(i, END, frames_gap):
            current_point = np.array(centroid_points_of_current_file[i])
            dist = np.linalg.norm(current_point - first_center_of_mass)
            x[j].append(i)
            y[j].append(dist)

    # Set all graph to start from origin
    origin_x = []

    for my_list in x:
        my_list = [e - min(my_list) for e in my_list]
        origin_x.append(my_list)

    # First plot - all diffs per frame
    for i, my_list in enumerate(origin_x):
        plt.plot(my_list, y[i])
        text = plt.text(my_list[-1], y[i][-1], f'{SUBJECT_AGES[i + 4]}')
        text.set_color(plt.gca().lines[-1].get_color())

    plt.show()
    plt.close()

    # Second plot - average diff per age
    means = []
    sub_ages = []

    for i, my_list in enumerate(origin_x):
        mean = np.nanmean(y[i])
        sub_age = SUBJECT_AGES[i + 4]
        means.append(mean)
        sub_ages.append(sub_age)

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    sns.set_style("darkgrid")
    plt.scatter(sub_ages, means)
    plt.title("Average difference in center of mass per age")
    plt.xlabel("age")
    plt.ylabel("average difference (in mm)")
    plt.show()

def find_center_of_mass(frame_points):
    """
    The function's name is misleading - it actually calculates the centroid of the 4 points Omer mentioned (RASI, LASI,
     LPSI and RPSI).

    :param points: 39 points of a single frame, matrix of shape (39, 3).
    :return: The centroid of RASI, LASI, LPSI and RPSI.
    """
    # Find RASI, LASI, LPSI and RPSI
    RASI = []
    LASI = []
    LPSI = []
    RPSI = []

    rasi_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'RASI'][0]
    lasi_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'LASI'][0]
    rpsi_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'RPSI'][0]
    lpsi_index = [i for i, e in enumerate(KEYPOINTS_NAMES) if e == 'LPSI'][0]

    for index, point in enumerate(frame_points):
        if index == rasi_index:
            RASI = point
        elif index == lasi_index:
            LASI = point
        elif index == rpsi_index:
            RPSI = point
        elif index == lpsi_index:
            LPSI = point

    sx = sy = sz = 0

    for p in [RASI, LASI, RPSI, LPSI]:
        sx += p[0]
        sy += p[1]
        sz += p[2]

    cx = sx / 4
    cy = sy / 4
    cz = sz / 4

    centroid = [cx, cy, cz]

    # ------------------------------------ For debugging: Draw the points ----------------------------------------------
    # points = np.copy(frame_points)
    # points = np.vstack([points, centroid])
    # pcd = o3d.geometry.PointCloud()
    # points = np.array([(point[0], point[1], point[2]) for point in points])
    # pcd.points = o3d.utility.Vector3dVector(points)
    #
    # # Color all points in black, except the 3 points used for angle calc.
    # points_mask = np.zeros(shape=(len(points), 3))
    # colors = np.zeros_like(points_mask)
    # colors[rasi_index] = [1, 0, 0]
    # colors[lasi_index] = [1, 0, 0]
    # colors[rpsi_index] = [1, 0, 0]
    # colors[lpsi_index] = [1, 0, 0]
    # colors[-1] = [0, 1, 0] # Centroid point
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    #
    # # Add points
    # visualizer = o3d.visualization.Visualizer()
    # visualizer.create_window(window_name='Vicon Points')
    # visualizer.add_geometry(pcd)
    # visualizer.run()
    # -----------------------------------------------------------------------------------------------------------------

    return centroid

def calculate_dataset_diversity(dataset):
    """
    Calculate how diverse a dataset is. "Diverse" is defined to be the average euclidean distance between consecutive
    frames. High diversity = large average euclidean distance = good dataset!

    :param dataset: List of all frames. Each frame is a tuple of [39_points_matrix, age, recording_file_name].
            frames are already sorted by recording_file_name, since i have not shuffled the dataset yet.
    :return: Diversity measurement (in mm).
    """

    average_distances = []

    for sample_index in range(len(dataset)-1):
        current_sample = dataset[sample_index]
        next_sample = dataset[sample_index + 1]

        if current_sample[2] == next_sample[2]: # Make sure they were taken from the same video.
            dists = []
            current_points = current_sample[0]
            next_points = next_sample[0]

            # Calculate euclidean distance between the objects.
            for point_index in range(39):

                # Ignore nans. In the Vicon dataset, if points was bad sampled, all coordinates are nan, so it's OK to
                # check only the `x` one.
                if math.isnan(current_points[point_index][0]) or math.isnan(next_points[point_index][0]):
                    continue

                p1 = current_points[point_index]
                p2 = next_points[point_index]
                d = np.linalg.norm(p1 - p2) # In mm.
                dists.append(d)

            average_distance = np.mean(dists)
            average_distances.append(average_distance)

    diversity = np.nanmean(average_distances) # In mm.
    return diversity

def increase_dataset_variance(dataset, threshold):
    """
    For each frame, check if the average euclidean distance to the previous object (frame) in the dataset is smaller than
    threshold. If it is - don't keep the object to the dataset. This is better than just "keeping" every Nth frame,
    since by "keeping" every Nth frame we ignore the fact that in some parts of the video the object is standing still
    and in other parts he is moving.

    :param dataset: List of all frames. Each frame is a tuple of [39_points_matrix, age, recording_file_name].
            frames are already sorted by recording_file_name, since i have not shuffled the dataset yet.
    :param threshold: What is the minimum average distance between consecutive frames that is allowed.
    :return: New dataset.
    """
    new_dataset = [dataset[0]] # Always add the first frame of the first file to the new dataset.

    for sample_index in range(1, len(dataset)):
        current_sample = dataset[sample_index]
        last_sample_added_to_new_dataset = new_dataset[-1]
        age = current_sample[1]

        if current_sample[2] == last_sample_added_to_new_dataset[2]: # Make sure they were taken from the same video,
            # if not, we can not compare them.
            dists = []
            current_points = current_sample[0]
            last_added_points = last_sample_added_to_new_dataset[0]

            # Calculate euclidean distance between the objects.
            for point_index in range(39):
                # Ignore nans. In the Vicon dataset, if points was bad sampled, all coordinates are nan, so it's OK to
                # check only the `x` one.
                if math.isnan(current_points[point_index][0]) or math.isnan(last_added_points[point_index][0]):
                    continue

                p1 = current_points[point_index]
                p2 = last_added_points[point_index]
                d = np.linalg.norm(p1 - p2)  # In mm.
                dists.append(d)

            average_distance = np.mean(dists)

            t = threshold[0] if age <= 30 else threshold[1]

            if average_distance >= t:
                new_dataset.append(current_sample)
        else: # This means we have finished reading all frames from the file, and we need to move to the next file.
            new_dataset.append(current_sample)

    return new_dataset

def create_splitted_dataset():
    DATA_PATH = '../../data_angles_to_age/'

    if not os.path.isdir(DATA_PATH + 'splitted/'):
        os.makedirs(DATA_PATH + 'splitted/')

    # Decide which subjects will be used for training and which for testing (~80%-~20%), make sure there will be similar
    # percentage of old in training and testing and young in training and testing.
    old_subjects_numbers = np.array(range(17, 24))
    young_subjects_numbers = np.array(range(1, 17))

    old_train_subject_numbers = random.sample(list(old_subjects_numbers), int(len(old_subjects_numbers) * 0.70))
    old_test_subject_numbers = list(set(old_subjects_numbers) - set(old_train_subject_numbers))
    young_train_subject_numbers = random.sample(list(young_subjects_numbers), int(len(young_subjects_numbers) * 0.70))
    young_test_subject_numbers = list(set(young_subjects_numbers) - set(young_train_subject_numbers))
    train_subject_numbers = old_train_subject_numbers + young_train_subject_numbers
    test_subjects_numbers = old_test_subject_numbers + young_test_subject_numbers

    print("Train numbers: ")
    print(train_subject_numbers)
    print("Test numbers: ")
    print(test_subjects_numbers)

    # Read all data.
    # Read ages.
    ages = {}
    data = pd.read_csv(DATA_PATH + "Subjects Ages.csv")

    for i in range(len(data['AGE'])):
        sub_name = 'Sub00' + str(i+1) if i+1 < 10 else 'Sub0' + str(i+1)
        ages[sub_name] = data['AGE'][i]

    train = [] # <matrix of shape (N, 39, 3), age> N is the size of the training set
    test = [] # <matrix of shape (n, 39, 3), age> n is the size of the testing set

    # Read pointclouds.
    for root, dirs, files in os.walk(DATA_PATH):

        for file in files:

            if file.endswith(".csv"):

                if "Sub001" in root: # The recordings in Sub001 were different than the others, so i'm not using it.
                    continue

                # ------------------------------------- Just for debugging ---------------------------------------------
                # if "Sub006" not in root and "Sub018" not in root and "Sub009" not in root and "Sub019" not in root:
                #     continue
                #
                # if "Left" not in file:
                #     continue
                # ------------------------------------------------------------------------------------------------------

                if file == 'Subjects Ages.csv':
                    continue

                # # TODO: Remove this
                # if 'Squat' in file:
                #     continue

                print("Adding points from {file}".format(file=file))

                remove_extension = file[:-4]
                splitted = remove_extension.split('_')

                if len(splitted) == 1:
                    splitted = remove_extension.split(' ')

                subject_name = [e for e in splitted if 'Sub' in e][0]
                subject_num = int(subject_name[3:])
                age = ages[subject_name]

                # Read all frames in that file.
                reader = VICONReader(root + '/' + file)
                frames = reader.get_points()

                # For each frame, add it as a separate pointcloud object of 39 points.
                for _, points in frames.items():
                    point_as_matrix = np.zeros((39, 3))

                    for i, p in enumerate(points):
                        point_as_matrix[i][0] = p.x
                        point_as_matrix[i][1] = p.y
                        point_as_matrix[i][2] = p.z

                    center_of_mass = find_center_of_mass(point_as_matrix)
                    pair = (point_as_matrix, age, file, center_of_mass)

                    # Add to all data
                    if subject_num in train_subject_numbers:
                        train.append(pair)
                    elif subject_num in test_subjects_numbers:
                        test.append(pair)
                    else:
                        print("Wrong subject number.")
                        return

    #plot_center_of_mass_diff(train)

    # Calculate how diverse the dataset is.
    train_diversity = calculate_dataset_diversity(train)
    print("Train size, with out removing frames, is {s}".format(s=len(train)))
    print("Train diversity, with out removing frames, is {d}".format(d=train_diversity))
    print("--------------------------------------------------------")

    # ------------------ Just for testing! Remove every N frames, to increase data variance. ---------------------------
    # N = 60
    # train_1 = [e for i, e in enumerate(train) if i % N == 0]
    # train_diversity = calculate_dataset_diversity(train_1)
    # print("Train size, after saving every Nth frame, is {s}".format(s=len(train_1)))
    # print("Train diversity, after saving every Nth frame, is {d}".format(d=train_diversity))
    # print("--------------------------------------------------------")
    # ------------------------------------------------------------------------------------------------------------------

    # Increase dataset variance.
    MINIMUM_AVERAGE_DIST_BETWEEN_FRAMES_OLD = 60 # In mm
    MINIMUM_AVERAGE_DIST_BETWEEN_FRAMES_YOUNG = 90 # In mm. Different threshold is used to increase number of samples of
    # old people, since dataset is not balanced.
    thresh = (MINIMUM_AVERAGE_DIST_BETWEEN_FRAMES_YOUNG, MINIMUM_AVERAGE_DIST_BETWEEN_FRAMES_OLD)
    train = increase_dataset_variance(dataset=train, threshold=thresh)
    test = increase_dataset_variance(dataset=test, threshold=thresh)
    train_diversity = calculate_dataset_diversity(train)
    print("Train size, after saving only diverse frames, is {s}".format(s=len(train)))
    print("Train diversity, after saving only diverse frames, is {d}".format(d=train_diversity))
    print("--------------------------------------------------------")

    # Split to samples and labels.
    x_train = [e[0] for e in train]
    y_train = [(e[1], e[2]) for e in train]
    x_test = [e[0] for e in test]
    y_test = [(e[1], e[2]) for e in test]

    # Save to npy files
    np.save('../../data_angles_to_age/splitted/x_train.npy', x_train, allow_pickle=True)
    np.save('../../data_angles_to_age/splitted/x_test.npy', x_test, allow_pickle=True)
    np.save('../../data_angles_to_age/splitted/y_train.npy', y_train, allow_pickle=True)
    np.save('../../data_angles_to_age/splitted/y_test.npy', y_test, allow_pickle=True)


if __name__ == "__main__":
    create_splitted_dataset()