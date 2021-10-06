"""
Extract a single depth image, and another one after clipping the visible distance from the camera.
"""
import copy

import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_hist(depth_map, max_depth):  # to make a histogram (count distribution frequency)
    values = [0]*((np.max(max_depth))+1)
    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            t = depth_map[i, j]
            values[t] += 1
    return values


def get_cdf(hist):
    hist = iter(hist)
    cdf = [next(hist)]
    for i in hist:
        cdf.append(cdf[-1] + i)
    return np.array(cdf)


def hist_equalizer_16_bit(depth_map, max_depth):
    flat = depth_map.flatten()
    hist = get_hist(depth_map, max_depth)
    cdf = get_cdf(hist)
    nj = (cdf - cdf.min()) * (max_depth-1)
    N = cdf.max() - cdf.min()

    # re-normalize the cdf
    cs = nj / N
    cs = cs.astype('uint16')
    img_new = cs[flat]
    depth_map = np.reshape(img_new, depth_map.shape)
    return depth_map


def generate_frames_without_processing():
    """
    Generate some depth frames as png, without any processing on them.

    :return: None.
    """
    IMAGE_PATH = '/media/lotemn/Other/project-data/trimmed/Sub013/Squat/Front/depth_frames/'

    images_indices = [6391, 6516, 7700, 7745]
    images_without_clipping = []

    # Generate some images without distance clipping
    for index in images_indices:
        depth_image = np.fromfile(IMAGE_PATH + str(index) + ".raw", dtype='int16', sep="")
        depth_image = depth_image.reshape([640, 480])
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.04),
                                           cv2.COLORMAP_JET)
        images_without_clipping.append(depth_colormap)

    # Stack images.
    resized_1 = cv2.resize(images_without_clipping[0], (0, 0), None, .5, .5)
    resized_2 = cv2.resize(images_without_clipping[1], (0, 0), None, .5, .5)
    resized_3 = cv2.resize(images_without_clipping[2], (0, 0), None, .5, .5)
    resized_4 = cv2.resize(images_without_clipping[3], (0, 0), None, .5, .5)

    vertical1 = np.vstack((resized_1, resized_2))
    vertical2 = np.vstack((resized_3, resized_4))
    horizontal = np.hstack((vertical1, vertical2))
    cv2.imwrite("without_processing.png", horizontal)


def generate_frames_with_depth_mask():
    """
    Generate depth frames using threshold to remove the background + removing bad sampled pixels (mostly around the object).

    :return: None
    """
    IMAGE_PATH = 'F:/Kimmel-Agmon/data/Sub013/Sub013/Squat/Front/depth_frames/'

    images_indices = [6391, 6516, 7700, 7745]
    images = []

    DEPTH_UNITS = 0.001  # Not sure what it represents.. That's what it says in the RealSense viewer.
    MAX_DIST = 3  # In meters.
    CLIPPING_DIST = MAX_DIST / DEPTH_UNITS

    # Generate image with distance clipping
    for index in images_indices:
        depth_map = np.fromfile(IMAGE_PATH + str(index) + ".raw", dtype='int16', sep="")
        depth_map = depth_map.reshape([640, 480])
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.04),
                                           cv2.COLORMAP_JET)

        # Threshold all pixels that are far from clipping distance.
        depth_mask = np.where((depth_map > CLIPPING_DIST) | (depth_map <= 0), 0,
                              255)  # Those dark blue halos around the
        # object are pixels with 0 values, i'm not sure why realsense generated them.
        depth_mask = np.stack((depth_mask,) * 3, axis=-1)
        depth_mask = depth_mask.astype('uint8')
        result = cv2.bitwise_and(depth_colormap, depth_mask)
        result[np.all(result == (128, 0, 0), axis=-1)] = (0, 0, 0)
        images.append(result)

    # Stack images.
    resized_1 = cv2.resize(images[0], (0, 0), None, .5, .5)
    resized_2 = cv2.resize(images[1], (0, 0), None, .5, .5)
    resized_3 = cv2.resize(images[2], (0, 0), None, .5, .5)
    resized_4 = cv2.resize(images[3], (0, 0), None, .5, .5)

    vertical1 = np.vstack((resized_1, resized_2))
    vertical2 = np.vstack((resized_3, resized_4))
    horizontal = np.hstack((vertical1, vertical2))
    cv2.imwrite("with_depth_mask.png", horizontal)


def generate_frames_with_depth_mask_and_histogram_equalization():
    """
    Generate depth frames using threshold to remove the background + histogram equalization.

    :return:
    """
    IMAGE_PATH = '/media/lotemn/Other/project-data/trimmed/Sub013/Squat/Front/depth_frames/'

    images_indices = [6391, 6516, 7700, 7745]
    images = []

    DEPTH_UNITS = 0.001  # Not sure what it represents.. That's what it says in the RealSense viewer.
    MAX_DIST = 3  # In meters.
    CLIPPING_DIST = MAX_DIST / DEPTH_UNITS

    # Generate image with distance clipping and histogram equalizer.
    for index in images_indices:
        depth_map = np.fromfile(IMAGE_PATH + str(index) + ".raw", dtype='int16', sep="")
        original_depth_map = depth_map.reshape([640, 480])

        # Threshold the pixels that are beyond the clipping distance.
        depth_map = np.where((original_depth_map > CLIPPING_DIST) | (original_depth_map <= 0), 0, original_depth_map)

        # Equalize hist
        t1 = int(np.power(2, 16))
        t2 = int(CLIPPING_DIST)
        depth_map = hist_equalizer_16_bit(depth_map, int(t2))

        # Colormap
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.04),
                                           cv2.COLORMAP_JET)

        # Threshold all pixels that are far from clipping distance.
        depth_mask = np.where((original_depth_map > CLIPPING_DIST) | (original_depth_map <= 0), 0,
                              255)  # Those dark blue halos around the

        # object are pixels with 0 values, i'm not sure why realsense generated them... Numeric errors? measurement errors?
        depth_mask = np.stack((depth_mask,) * 3, axis=-1)
        depth_mask = depth_mask.astype('uint8')
        result = cv2.bitwise_and(depth_colormap, depth_mask)
        result[np.all(result == (128, 0, 0), axis=-1)] = (0, 0, 0)
        images.append(result)

    # Stack images.
    resized_1 = cv2.resize(images[0], (0, 0), None, .5, .5)
    resized_2 = cv2.resize(images[1], (0, 0), None, .5, .5)
    resized_3 = cv2.resize(images[2], (0, 0), None, .5, .5)
    resized_4 = cv2.resize(images[3], (0, 0), None, .5, .5)

    vertical1 = np.vstack((resized_1, resized_2))
    vertical2 = np.vstack((resized_3, resized_4))
    horizontal = np.hstack((vertical1, vertical2))
    cv2.imwrite("with_depth_mask_and_hist_equalization.png", horizontal)


if __name__ == "__main__":
    #generate_frames_without_processing()
    generate_frames_with_depth_mask()
    #generate_frames_with_depth_mask_and_histogram_equalization()





