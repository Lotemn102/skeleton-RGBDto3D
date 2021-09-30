import numpy as np
import cv2
import os
import json


path = '/media/lotemn/Other/project-data/trimmed/'
total = 0

for i in range(3, 16):
    subject_name = 'Sub00' + str(i) if i < 10 else 'Sub0' + str(i)
    subject_num = i

    for position in ['Stand', 'Squat', 'Tight', 'Left', 'Right']:
        if subject_num == 2 and position not in ['Squat', 'Stand']:
            continue

        if subject_num == 3 and position != 'Squat':
            continue

        if subject_num == 4 and position == 'Squat':
            continue

        for angle in ['Front', 'Back', 'Side']:
            temp_path = path + subject_name + '/' + position + '/' + angle + '/log.json'
            f = open(temp_path)
            data = json.load(f)
            num_frames = int(data['Number of frames Realsense RGB'])
            print(num_frames)
            total = total + num_frames
            f.close()


print(total)


