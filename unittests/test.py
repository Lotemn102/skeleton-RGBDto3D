import numpy as np
import cv2
import os


path = '/media/lotemn/Other/project-data/frames/Sub004/RealSenseDepth/Left/Front/'
all_frames_files_realsense_depth = os.listdir(path)
all_frames_files_realsense_depth.remove('log.json')
all_frames_files_realsense_depth = sorted(all_frames_files_realsense_depth, key=lambda x: int(x[:-4]))

for frame in all_frames_files_realsense_depth:
    depth_image = np.fromfile(path + frame, dtype='int16', sep="")
    depth_image = depth_image.reshape([640, 480])
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),
                                       cv2.COLORMAP_JET)
    cv2.imshow("dfdf", depth_colormap)
    cv2.waitKey(30)