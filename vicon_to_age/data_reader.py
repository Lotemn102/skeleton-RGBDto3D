"""
Read vicon points and their age labels.
Save into dict of <'pcs', 'labels'>, where 'pcs' is a matrix of the points in shape (dataset_size, 39, 3) and labels is
a matrix of ages in shape (dataset_size, 1).
"""
import math
import random
import glob
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import open3d as o3d

#from rgbd_to_3d.data_cleaning.vicon_data_reader import VICONReader

def is_there_nan_coordinate(item):
    for coordinate in item:
        if math.isnan(coordinate):
            return True

    return False


def read(remove_nans=True):
    DATA_PATH = '../../data_3d_to_age/splitted/'

    for file in glob.glob(DATA_PATH + '*.npy'):
        if 'x_train' in file:
            x_train = np.load(file, allow_pickle=True)

        if 'x_test' in file:
            x_test = np.load(file, allow_pickle=True)

        if 'y_train' in file:
            y_train = np.load(file, allow_pickle=True)

        if 'y_test' in file:
            y_test = np.load(file, allow_pickle=True)

    # TODO: Just for now! Remove all frames that have at least one point with nan value.
    if remove_nans:
        counter = 0
        x_train_nans_indices = []
        x_test_nans_indices = []

        for i, row in enumerate(x_train):
            for item in row:
                if is_there_nan_coordinate(item):
                    counter += 1
                    x_train_nans_indices.append(i)
                    break

        print("Percentage of removed train frames with at least 1 nan value: {num}%".format(num=round(100*(counter / x_train.shape[0]), 3)))

        counter = 0

        for i, row in enumerate(x_test):
            for item in row:
                if is_there_nan_coordinate(item):
                    counter += 1
                    x_test_nans_indices.append(i)
                    break

        print("Percentage of removed test frames with at least 1 nan value: {num}%".format(num=round(100*(counter / x_test.shape[0]), 3)))

        clean_x_train = np.zeros((x_train.shape[0]-len(x_train_nans_indices), 39, 3))
        clean_y_train = np.zeros((x_train.shape[0]-len(x_train_nans_indices), 1))
        clean_x_test = np.zeros((x_test.shape[0] - len(x_test_nans_indices), 39, 3))
        clean_y_test = np.zeros((x_test.shape[0] - len(x_test_nans_indices), 1))
        current_row = 0

        for i, _ in enumerate(x_train):
            if i not in x_train_nans_indices:
                clean_x_train[current_row] = x_train[i]
                clean_y_train[current_row] = y_train[i]
                current_row += 1

        current_row = 0
        for i, _ in enumerate(x_test):
            if i not in x_test_nans_indices:
                clean_x_test[current_row] = x_test[i]
                clean_y_test[current_row] = y_test[i]
                current_row += 1

        x_train = clean_x_train
        x_test = clean_x_test
        y_train = clean_y_train
        y_test = clean_y_test

    # Convert labels into binary for first sanity check.
    np.place(y_train, y_train <= 30, 0) # 0 for young
    np.place(y_train, y_train >= 60, 1) # 1 for old

    np.place(y_test, y_test <= 30, 0)  # 0 for young
    np.place(y_test, y_test >= 60, 1)  # 1 for old

    return x_train, x_test, y_train, y_test

def print_metadata():
    x_train, x_test, y_train, y_test = read(remove_nans=True)
    total = len(y_train)+len(y_test)
    total_train = len(y_train)
    total_test = len(y_test)
    print("Total objects: {n}".format(n=total))
    print("Train objects: {n}".format(n=total_train))
    print("Test objects: {n}".format(n=total_test))
    print("Percentage of objects of age 'old' in train: {n}%".format(n=round(100*(len(np.where(y_train == 1)[0])/ total_train), 3)))
    print("Percentage of objects of age 'old' in test: {n}%".format(n=round(100*(len(np.where(y_test == 1)[0]) / total_test), 3)))
    print("Percentage of objects of age 'young' in train: {n}%".format(n=round(100*(len(np.where(y_train == 0)[0])/ total_train), 3)))
    print("Percentage of objects of age 'young' in test: {n}%".format(n=round(100*(len(np.where(y_test == 0)[0])/ total_test), 3)))
    print("All ages: {n}".format(n=sorted(set(np.unique(np.concatenate((y_train, y_test)))))))

def draw_some_examples(type='both'): # 'both', 'old', 'young'
    x_train, x_test, y_train, y_test = read(remove_nans=True)

    if type == 'old':
        for i, _ in enumerate(x_train):
            if int(y_train[i]) == 1: # 1 is old
                points = x_train[i]
                template_ = o3d.geometry.PointCloud()
                template_.points = o3d.utility.Vector3dVector(points)
                o3d.visualization.draw_geometries([template_])
    elif type == 'young':
        for i, _ in enumerate(x_train):
            if int(y_train[i]) == 0: # 0 is young
                points = x_train[i]
                template_ = o3d.geometry.PointCloud()
                template_.points = o3d.utility.Vector3dVector(points)
                o3d.visualization.draw_geometries([template_])
    elif type == 'both':
        for i, _ in enumerate(x_train):
            label = int(y_train[i])
            points = x_train[i]
            template_ = o3d.geometry.PointCloud()
            template_.points = o3d.utility.Vector3dVector(points)
            o3d.visualization.draw_geometries([template_])
    else:
        print("Invalid type. Valid types are: 'old', 'young', 'both'.")



if __name__ == "__main__":
    print_metadata()
    #draw_some_examples(type='old')