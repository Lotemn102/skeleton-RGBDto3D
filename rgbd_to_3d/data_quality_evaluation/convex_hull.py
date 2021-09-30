"""
Find the convex hull of the object in rgb images, and the in 3D vicon points.
"""
import math

import cv2
import numpy as np
import random as rng
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from data_cleaning.vicon_data_reader import VICONReader


def find_convex_hull_single_rgb():
    """
    Find the convex hull of object in rgb image.
    Source: https://learnopencv.com/convex-hull-using-opencv-in-python-and-c/

    :return: None.
    """
    IMAGE_PATH = '/media/lotemn/Other/project-data/trimmed/Sub013/Squat/Front/rgb_frames/'
    DEPTH_PATH = '/media/lotemn/Other/project-data/trimmed/Sub013/Squat/Front/depth_frames/'

    DEPTH_UNITS = 0.001  # Not sure what it represents.. That's what it says in the RealSense viewer.
    MAX_DIST = 1.8  # In meters.
    CLIPPING_DIST = MAX_DIST / DEPTH_UNITS

    images_indices = [6391, 6516, 7700, 7745]

    for index in images_indices:
        path = IMAGE_PATH + str(index) + ".png"

        # Load the rgb image
        rgb_image = cv2.imread(path)

        # Use the correlated depth frame as mask.
        depth_map = np.fromfile(DEPTH_PATH + str(index) + ".raw", dtype='int16', sep="")
        depth_map = depth_map.reshape([640, 480])

        # Threshold all pixels that are far from clipping distance.
        depth_mask = np.where((depth_map > CLIPPING_DIST) | (depth_map <= 0), 0,
                              255)  # Those dark blue halos around the
        depth_mask = np.stack((depth_mask,) * 3, axis=-1)
        depth_mask = depth_mask.astype('uint8')
        kernel = np.ones((10, 10), 'uint8')
        dilated_mask = cv2.dilate(depth_mask, kernel, iterations=2)

        # Combine the image and the mask
        masked = cv2.bitwise_and(rgb_image, dilated_mask)
        masked[np.all(masked == (0, 0, 0), axis=-1)] = (255, 255, 255)

        # Convert image to greyscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (3, 3))

        # Threshold the image
        #ret, thresh = cv2.threshold(gray, 127, 255, 0)

        # Detect edges using Canny
        threshold = 100
        canny_output = cv2.Canny(gray, threshold, threshold * 2)

        cv2.imshow('Contours', gray)
        cv2.waitKey(0)


        # Find contours
        contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the convex hull object for each contour
        hull_list = []
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)

        # Draw contours + hull results
        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            cv2.drawContours(rgb_image, contours, i, color)
            cv2.drawContours(rgb_image, hull_list, i, color)
        # Show in a window
        cv2.imshow('Contours', rgb_image)
        cv2.waitKey(0)


        '''
        # Find the contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # For each contour, find the convex hull and draw it
        # on the original image.
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            cv2.drawContours(rgb_image, [hull], -1, (255, 0, 0), 2)
        # Display the final convex hull image
        cv2.imshow('ConvexHull', rgb_image)
        cv2.waitKey(0)
        '''

def find_convex_hull_single_rgb_2():
    IMAGE_PATH = '/media/lotemn/Other/project-data/trimmed/Sub013/Squat/Front/rgb_frames/'
    DEPTH_PATH = '/media/lotemn/Other/project-data/trimmed/Sub013/Squat/Front/depth_frames/'

    DEPTH_UNITS = 0.001  # Not sure what it represents.. That's what it says in the RealSense viewer.
    MAX_DIST = 2  # In meters.
    CLIPPING_DIST = MAX_DIST / DEPTH_UNITS

    images_indices = [6391, 6516, 7700, 7745]

    for index in images_indices:
        path = IMAGE_PATH + str(index) + ".png"

        # Load the rgb image
        rgb_image = cv2.imread(path)

        # Use the correlated depth frame as mask.
        depth_map = np.fromfile(DEPTH_PATH + str(index) + ".raw", dtype='int16', sep="")
        depth_map = depth_map.reshape([640, 480])

        # Threshold all pixels that are far from clipping distance.
        depth_mask = np.where((depth_map > CLIPPING_DIST) | (depth_map <= 0), 0,
                              255)  # Those dark blue halos around the
        depth_mask = np.stack((depth_mask,) * 3, axis=-1)
        depth_mask = depth_mask.astype('uint8')

        # Blur out the background
        blurred_object = cv2.GaussianBlur(rgb_image, (51, 51), 0)
        gray_pixel = 128

        # Combine the original with the blurred frame based on mask
        frame = np.where(depth_mask == (255, 255, 255), blurred_object, gray_pixel)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the convex hull object for each contour
        hull_list = []
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)
        # Draw contours + hull results
        drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            cv2.drawContours(drawing, contours, i, color)
            #cv2.drawContours(drawing, hull_list, i, color)
        # Show in a window
        cv2.imshow('Contours', thresh)
        cv2.waitKey(0)

def find_convex_hull_vicon_single_sessions():
    CSV_PATH = '/media/lotemn/Other/project-data/trimmed/Sub013/Stand/Front/Sub013_Stand_Front.csv'

    reader = VICONReader(CSV_PATH)
    points_map = reader.get_points()
    all_points = []

    for index in points_map.keys():
        list_points = []
        points = points_map[index]

        # Convert from struct Point to list[3].
        for p in points:
            temp = [p.x, p.z]

            if not math.isnan(temp[0]):
                list_points.append(temp)

        list_points = np.array(list_points)
        all_points.extend(list_points)

    all_points = np.array(all_points)
    # Calculate the convex hull.
    hull = ConvexHull(all_points)

    # Draw the result.
    hull_indices = hull.vertices

    # These are the actual points.
    hull_pts = all_points[hull_indices, :]

    '''
    # PLOT IN 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot defining corner points
    ax.plot(hull_pts.T[0], hull_pts.T[1], hull_pts.T[2], "ko")

    # 12 = 2 * 6 faces are the simplices (2 simplices per square face)
    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(list_points[s, 0], list_points[s, 1], list_points[s, 2], "r-")

    # Make axis label
    for i in ["x", "y", "z"]:
        eval("ax.set_{:s}label('{:s}')".format(i, i))

    ax.view_init(0, 90)
    plt.show()
    '''

    plt.plot(all_points[:, 0], all_points[:, 1], 'ko', markersize=10)
    plt.fill(hull_pts[:, 0], hull_pts[:, 1], fill=False, edgecolor='b')
    plt.savefig("stand_convex_hull.png")

def find_convex_hull_vicon_multiple_sessions():
    paths = []

    for i in range(4, 15):
        subject_name = 'Sub00' + str(i) if i < 10 else 'Sub0' + str(i)
        CSV_PATH = '/media/lotemn/Other/project-data/trimmed/' + subject_name + '/Stand/Front/' + subject_name + '_Stand_Front.csv'
        paths.append(CSV_PATH)

    all_points = []

    for path in paths:
        reader = VICONReader(path)
        points_map = reader.get_points()
        first_frame = list(points_map.keys())[0]
        points = points_map[first_frame]

        # Convert from struct Point to list[3].
        for p in points:
            temp = [p.x, p.z]

            if not math.isnan(temp[0]):
                all_points.append(temp)

    all_points = np.array(all_points)
    # Calculate the convex hull.
    hull = ConvexHull(all_points)

    # Draw the result.
    hull_indices = hull.vertices

    # These are the actual points.
    hull_pts = all_points[hull_indices, :]

    '''
    # PLOT IN 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot defining corner points
    ax.plot(hull_pts.T[0], hull_pts.T[1], hull_pts.T[2], "ko")

    # 12 = 2 * 6 faces are the simplices (2 simplices per square face)
    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(list_points[s, 0], list_points[s, 1], list_points[s, 2], "r-")

    # Make axis label
    for i in ["x", "y", "z"]:
        eval("ax.set_{:s}label('{:s}')".format(i, i))

    ax.view_init(0, 90)
    plt.show()
    '''

    plt.plot(all_points[:, 0], all_points[:, 1], 'ko', markersize=10)
    plt.fill(hull_pts[:, 0], hull_pts[:, 1], fill=False, edgecolor='b')
    plt.savefig("stand_convex_hull_all.png")

if __name__ == "__main__":
    find_convex_hull_vicon_multiple_sessions()





