"""
Find the convex hull of the object in rgb images, and the in 3D vicon points.
"""

import cv2
import numpy as np
import random as rng


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
        blurred_object = cv2.GaussianBlur(rgb_image, (71, 71), 0)
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
            #cv2.drawContours(drawing, contours, i, color)
            cv2.drawContours(drawing, hull_list, i, color)
        # Show in a window
        cv2.imshow('Contours', drawing)
        cv2.waitKey(0)





if __name__ == "__main__":
    find_convex_hull_single_rgb_2()





