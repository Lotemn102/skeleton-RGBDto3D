import os

folder_path_rgb = '../preprocessing/frames/Sub004/RealSense/Left/Front'
folder_path_depth = '../preprocessing/frames/Sub004/RealSenseDepth/Left/Front'

all_frames_files_rgb = os.listdir(folder_path_rgb)
all_frames_files_rgb.remove('log.json')
all_frames_files_rgb = sorted(all_frames_files_rgb, key=lambda x: int(x[:-4]))

all_frames_files_depth = os.listdir(folder_path_depth)
all_frames_files_depth.remove('log.json')
all_frames_files_depth = sorted(all_frames_files_depth, key=lambda x: int(x[:-4]))

print(len(all_frames_files_rgb))
print(len(all_frames_files_depth))

for i in range(len(all_frames_files_depth)):
    if all_frames_files_rgb[i] != all_frames_files_depth[i]:
        print(all_frames_files_rgb[i], all_frames_files_depth[i])