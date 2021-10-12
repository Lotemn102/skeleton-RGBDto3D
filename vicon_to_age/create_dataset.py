"""
Copy all csv files from the disk-on-key. Data is raw, no modifications were added / no frames were trimmed.
"""
import math
import os
from shutil import copyfile
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import random
import json
import matplotlib.pyplot as plt
import seaborn as sns

from rgbd_to_3d.data_cleaning.vicon_data_reader import VICONReader

def copy_csv_files():
    SRC_PATH = 'H:/Movement Sense Research/Vicon Validation Study/'
    DST_PATH = '../../data_3d_to_age/'

    # Iterate through all sub folders in src, and copy all csv files to dst.
    for root, dirs, files in os.walk(SRC_PATH):
        for file in files:
            if file.endswith(".csv"):
                if 'without' in file or 'Cal' in file or 'original' in root:
                    continue
                remove_extension = file[:-4]
                splitted = remove_extension.split('_')

                if len(splitted) == 1:
                    splitted = remove_extension.split(' ')

                subject_name = [e for e in splitted if 'Sub' in e][0]
                path = DST_PATH + subject_name

                if not os.path.isdir(path):
                    os.makedirs(path)

                # In some sessions, they did multiple recordings. I want to use all of these files, so avoid overriding...
                if os.path.isfile(path + '/' + file):
                    save_file = file[:-4]
                    save_file = save_file + "_"
                    save_file = save_file + ".csv"
                else:
                    save_file = file

                copyfile(root + '/' + file, path + '/' + save_file)
                print("Saved {file}.".format(file=save_file))

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

            if average_distance >= threshold:
                new_dataset.append(current_sample)
        else: # This means we have finished reading all frames from the file, and we need to move to the next file.
            new_dataset.append(current_sample)

    return new_dataset

def plot_distances_between_frames_in_file(dataset, file_name, savefig_name):
    """
    Plot the distance between every consecutive frames in a "Squat" recording.

    :param dataset:
    :return:
    """
    average_dists = []

    for sample_index in range(len(dataset)-1):
        current_sample = dataset[sample_index]
        next_sample = dataset[sample_index + 1]

        if current_sample[2] == file_name and next_sample[2] == file_name: # Make sure they were taken from the same video.
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

            average_dists.append(np.mean(dists))

    sns.set_style("darkgrid")
    plt.plot(range(len(average_dists)), average_dists)
    plt.title("Average euclidean distance between objects in consecutive frames in {file}".format(file=file_name), fontsize=10)
    plt.xlabel("Object index")
    plt.ylabel("Euclidean distance (mm)")
    plt.savefig(savefig_name + '.png')
    plt.close()


def create_splitted_dataset():
    DATA_PATH = '../../data_3d_to_age/'

    if not os.path.isdir(DATA_PATH + 'splitted/'):
        os.makedirs(DATA_PATH + 'splitted/')

    # Decide which subjects will be used for training and which for testing (~80%-~20%), make sure there will be similar
    # percentage of old in training and testing and young in training and testing.
    old_subjects_numbers = np.array(range(17, 22))
    young_subjects_numbers = np.array(range(1, 17))

    old_train_subject_numbers = random.sample(list(old_subjects_numbers), int(len(old_subjects_numbers) * 0.8))
    old_test_subject_numbers = list(set(old_subjects_numbers) - set(old_train_subject_numbers))
    young_train_subject_numbers = random.sample(list(young_subjects_numbers), int(len(young_subjects_numbers) * 0.8))
    young_test_subject_numbers = list(set(young_subjects_numbers) - set(young_train_subject_numbers))
    train_subject_numbers = old_train_subject_numbers + young_train_subject_numbers
    test_subjects_numbers = old_test_subject_numbers + young_test_subject_numbers

    # Read all data.
    # Read ages.
    ages = {}
    data = pd.read_csv(DATA_PATH + "Subjects Ages.csv")

    for i in range(len(data['AGE'])):
        sub_name = 'Sub00' + str(i+1) if i+1 < 10 else 'Sub0' + str(i+1)
        ages[sub_name] = data['AGE'][i]

    train = [] # <matrix of shape (N, 39, 3), age> N is the size of the training set
    test = [] # <matrix of shape (n, 39, 3), age> n is the size of the testing set

    counter = 0

    # Read pointclouds.
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".csv"):

                if "Sub001" in root:
                    continue

                if file == 'Subjects Ages.csv':
                    continue

                if counter > 40:
                    break

                print("Adding points from {file}".format(file=file))

                counter += 1

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

                    pair = (point_as_matrix, age, file)

                    # Add to all data
                    if subject_num in train_subject_numbers:
                        train.append(pair)
                    elif subject_num in test_subjects_numbers:
                        test.append(pair)
                    else:
                        print("Wrong subject number.")
                        return

    FILE_NAME = 'Sub007 Squat.csv'
    # Calculate how diverse the dataset is.
    train_diversity = calculate_dataset_diversity(train)
    print("Train size, with out removing frames, is {s}".format(s=len(train)))
    print("Train diversity, with out removing frames, is {d}".format(d=train_diversity))
    print("--------------------------------------------------------")
    plot_distances_between_frames_in_file(train, FILE_NAME, "without_removing")

    # ------------------ Just for testing! Remove every N frames, to increase data variance. ---------------------------
    N = 60
    train_1 = [e for i, e in enumerate(train) if i % N == 0]
    train_diversity = calculate_dataset_diversity(train_1)
    print("Train size, after saving every Nth frame, is {s}".format(s=len(train_1)))
    print("Train diversity, after saving every Nth frame, is {d}".format(d=train_diversity))
    print("--------------------------------------------------------")
    plot_distances_between_frames_in_file(train_1, FILE_NAME, "remove_60")
    # ------------------------------------------------------------------------------------------------------------------

    # Increase dataset variance.
    MINIMUM_AVERAGE_DIST_BETWEEN_FRAMES = 80 # In mm
    train = increase_dataset_variance(dataset=train, threshold=MINIMUM_AVERAGE_DIST_BETWEEN_FRAMES)
    test = increase_dataset_variance(dataset=test, threshold=MINIMUM_AVERAGE_DIST_BETWEEN_FRAMES)
    train_diversity = calculate_dataset_diversity(train)
    print("Train size, after saving only diverse frames, is {s}".format(s=len(train)))
    print("Train diversity, after saving only diverse frames, is {d}".format(d=train_diversity))
    print("--------------------------------------------------------")
    plot_distances_between_frames_in_file(train, FILE_NAME, "diverse_frames")

    # Shuffle
    random.shuffle(train)
    random.shuffle(test)

    # Split to samples and labels.
    x_train = [e[0] for e in train]
    y_train = [e[1] for e in train]
    x_test = [e[0] for e in test]
    y_test = [e[1] for e in test]

    # Save to npy files
    # np.save('../../data_3d_to_age/splitted/x_train.npy', x_train, allow_pickle=True)
    # np.save('../../data_3d_to_age/splitted/x_test.npy', x_test, allow_pickle=True)
    # np.save('../../data_3d_to_age/splitted/y_train.npy', y_train, allow_pickle=True)
    # np.save('../../data_3d_to_age/splitted/y_test.npy', y_test, allow_pickle=True)


if __name__ == "__main__":
    create_splitted_dataset()






    