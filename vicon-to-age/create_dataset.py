"""
Copy all csv files from the disk-on-key. Data is raw, no modifications were added / no frames were trimmed.
"""

import os
from shutil import copyfile
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import h5py
import json

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

def create_splitted_dataset():
    DATA_PATH = '../../data_3d_to_age/'

    # Read all data.
    # Read ages.
    ages = {}
    data = pd.read_csv(DATA_PATH + "Subjects Ages.csv")

    for i in range(len(data['AGE'])):
        sub_name = 'Sub00' + str(i+1) if i+1 < 10 else 'Sub0' + str(i+1)
        ages[sub_name] = data['AGE'][i]

    all_data = [] # <matrix of shape (39, 3), age>

    # Read pointclouds.
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".csv"):
                if "Sub001" in root:
                    continue

                if file == 'Subjects Ages.csv':
                    continue

                print("Adding points from {file}".format(file=file))

                remove_extension = file[:-4]
                splitted = remove_extension.split('_')

                if len(splitted) == 1:
                    splitted = remove_extension.split(' ')

                subject_name = [e for e in splitted if 'Sub' in e][0]
                age = ages[subject_name]

                # Read all frames in that file.
                reader = VICONReader(root + '/' + file)
                frames = reader.get_points()

                # For each frame, add it as a separate pointcloud object of 39 points.
                for _, points in frames.items():
                    points_as_matrix = np.zeros((39, 3))

                    for i, p in enumerate(points):
                        points_as_matrix[i][0] = p.x
                        points_as_matrix[i][1] = p.y
                        points_as_matrix[i][2] = p.z

                    pair = (points_as_matrix, age)

                    # Add to all data
                    all_data.append(pair)


    # Split to trainset and testset.
    X = [e[0] for e in all_data]
    y = [e[1] for e in all_data]

    x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)

    # Save to npy files
    np.save('../../data_3d_to_age/splitted/x_train.npy', x_train, allow_pickle=True)
    np.save('../../data_3d_to_age/splitted/x_test.npy', x_test, allow_pickle=True)
    np.save('../../data_3d_to_age/splitted/y_train.npy', y_train, allow_pickle=True)
    np.save('../../data_3d_to_age/splitted/y_test.npy', y_test, allow_pickle=True)

if __name__ == "__main__":
    pass






    