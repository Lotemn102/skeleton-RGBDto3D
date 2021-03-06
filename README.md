# skeleton-RGBDto3D
This project consists of 2 sub-projects:
1. Train a network for extracting 3d skeleton points from RGBD images.
2. Train a regressor for finding a person's age from its 3d skeleton.

The `Issues` section in the repository holds all my progress summaries and meeting summaries.

## Repository tree structure
This repository contains folders for both projects.

```
|--- angles_to_age
|---  rgbd_to_3d
|--- report
|--- vicon_to_age
```

- `rgbd_to_3d` contains all code for the first sub-project.
- `vicon_to_age` contains all code for second sub-project, using PointNet.
- `angles_to_age` contains all code for second sub-project, using classical classifiers.
- `report` contains my report and presentation if you wish to get more details.

## rgbd_to_3d

### Main steps were done

- **Synchronizing the first 15 recording sessions** - In each session there are recordings of 3 realsense cameras (videos) 
from the same angles (`Front`, `Back`, `Side`), and recording from the vicon sensors (csv file with 39 3d coordinates for each frame). 
  - These recording are not 
  synchronized by default, i.e in the first frame in each recording the object is in a slightly different position. These 
  is because when recording the object, there are less than 4 people in the recording lab, so a single person needs to
  start the recording in camera X, and then start the recording in camera Y and so on.
  - The FPS of the realsense by default is 30, and the FPS of the vicon in 120. In some places, if the realsense cameras
  were connected to USB2 instead of USB3, the realsense FPS is 15. So even if the recordings were synchronized to the first
  frame, they would be un-synchronized several frames after.
  - Since 2 of the realsense cameras in each session are connected to the same laptop, there is a frame drop in the output
  of these cameras. i.e instead of the output will be in for example 30 FPS, the output has frame-drop in random places in the
  recording.
  - There are no timestamps in the vicon csv data.

  In order to solve these problems, the following steps were done:
  - At the beginning of each recording, the subject is asked to perform T-pose, i.e raise its hands to create the shape of T.
  This shape was used to decide which frame is the first frame in each recording. I first tried to use `OpenPose` to 
  extract the frame with the T-pose. The accuracy of the `Front` and `Back` positions was OK, but on the `Side` position 
  `OpenPose` was not able to detect the object's skeletion at all. I've decided to manually detect the frames 
  with the T-pose "by eye".
  - To solve the different FPS problem, we have decided to take every 4th frame from the vicon data. I've also tried to average 
  every 4 frames (or 8 if the video's fps is 15). The measurement to see which method is better was to calculate an angle in the neck. There was no big 
  difference between the two methods, I've decided to use the first one.
  - To solve the realsense frame-drop problem, I have extracted for each realsense recording the frames numbers from the bag files.
  The frame numbers were the frames numbers realsense was able to save to the bag file. For example, frames `1, 2, 3, 4, 6, 8, 10`.
  In this example, realsense was not able to save frames `5, 7, 9`. For each recording, after extracting the frames numbers, 
  i've used the differences in the frames in order to "keep" only the corresponding frames from the vicon data. For example, if 
  the realsense frame numbers series is `1, 2, 4`, the corresponding frames to keep are `1, 5, 13`:

  <p align="center">
  <img width="700" src="https://user-images.githubusercontent.com/35609587/136187306-6200be3e-d1d8-4558-9641-ca6418102a7a.png">
  </p>

  Validation of the synchronizing was done manually as well - After trimming each realsense video with the corresponding vicon 
  frames, i have visualized the video and made sure the object postion is "the same" for each frame. Sometimes, due to the
  framedrop, the T-pose was not detected correctly, and i had to pick another frame as first frame,
  where the object is doing some other movement easy to detect in the video and the vicon data.
- **Projecting the vicon points into the realsense pixels** - `OpenPose` gets as input RGB image, and outputs N pixels, 
in which the keypoints were detected. We have decided to use `OpenPose` as a basic network. Therefore, we had to project 
the vicon points into the realsense frames. The projection is not trivial since these are 2 different coordinates systems.
I have used `Kabsch` algorithm in order to find the transformation between the 2 coordinate systems. After applying this transformation
on the vicon points, they were projected into the rgb frames using the realsense intrinsic parameters (written in the bag file).
There was error of ~60mm in the projection. More work is needed on this step, right now the work
focuses on finding frames were the object is standing still, to improve the calibration. In future recordings there will
be placed a static object with some points on it, so the calibration process should be more accurate.
  
  

### Folder tree structure
```
|--- rgbd_to_3d
  |---  assets
  |--- papers
  |--- data_cleaning
  |--- models
  |--- noy_mark_test
  |--- unittests
```

- `assets` folder contains template generator for the `frames_sync.json`, a template for the vicon csv file, a mapping of 
the missing vicon points per recording, and the first frames of each recording that were detected manually (in `frames_sync.json`).
- `data_cleaning` folder contain scripts for creating the dataset and projecting the vicon points:
  - `check_angles_accuracy_on_different_fps.py` was used to check the angles differences when using every 4th frame from the
  vicon or averaging every 4 frames.
  - `cvat_data_reader.py` is used to read the `CVAT` json files. `CVAT` is a free online annotation tool. I used it to 
  annotate the skeleton points on the rgb frames, so we will be able to construct 3d realsense points for each marker.
  These 3d realsense points were later used in `Kabsch` algorithm in order to find the transformation between them and the
  vicon points.
  - `generate_frame.py` - was used to extract frames from the bag files, and draw the vicon points to rgb image. These frames
  were later used to find the T-pose in each recording.
  - `kabsch.py` - contains implementation for `Kabsch` algorithm.
  - `missing_data.py` - used to get information about the amount of missing realsense frames (due to frame drop) and missing
  vicon points.
  - `projection.py` - script for extracting 3d realsense points from the annotated pixels, calculating `Kabsch` transformation
  and projecting the points into the rgb frames.
  - `realsense_data_reader` - is used to read bag files.
  - `remove_depth_noise.py` - contains script for removing the depth background in a single image. It was not used yet, 
  i was thinking to use it after converting the `OpenPose` input layer to get RGBD input (instead of RGB).
  - `structs` - contains some structs for the data readers.
  - `trim_data.py` - trim the realsense video and vicon data, based on the synchronized frames manually found beforehand.
  - `validate_trimming.py` - visualize the realsense video and vicon points after trimming.
  - `vicon_data_reader.py` - used to read the csv vicon files and parse them into points.
  - `visualize_functions` - used to visualized the realsense and vicon data.
- `models` folder contains summary i've written of the `OpenPose` architecture.
- `noy_mark_test` folder contains scripts for running Noy & Mark's tool. This tool was supposed to find the T-pose in the frames
but it didn't work for me. 
- `unittests` folder contains unittests for the different scripts.


### How to re-produce my steps
- **How to generate frames from a bag file** - in the file `data_cleaning/generate_frames.py` use the function `aux_generate_realsense_frames` to
create realsense frames, and the function `aux_generate_vicon_frames` to draw the vicon points on images. 
- **Where should I write the T-pose frames after manually detecting them** - in the file `data_cleaning/assets/frames_sync.json`.
- **How to trim the data** - After updating  `data_cleaning/assets/frames_sync.json` with the correct frames, call `trim_all`
function in `trim_data.py`.
- **How to validate trimming** - Run `data_cleaning/validate_trimming.py`.
- **How to calculate Kabsch transformation** -  Call the function `kabsch` from `kabsch.py`.

## vicon_to_age & angles_to_age

### Main steps were done
- **Generated dataset** - Original dataset contains all frames from all recordings. Total of ~250,000 frames. each sample has 39 points, and age as a label. Dataset consists of 23 different people, 7 'old' and 16 'young'. Many of the samples in the dataset are actually the same frame+-, due to the Vicon high fps. In order to increase the dataset variance, i have sampled only frames that there average euclidean distance is 80mm. After this process, only ~3000 samples were left. 

  <p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/35609587/138662416-087a7a9e-a0a0-4a04-a0c2-1198ff7dfd8d.png">
  </p>
  
- `PointNet` - I have first trained `PointNet` as is on this data, for binary classification of `young` vs `old`. There was high overfitting on the train set. 
- Classifying by angles - I have calculated for each frame 4 angles, as Omer and Maayan asked. I have trained several classical clasifiers (KNN, SVM, random forest) on that data. The results were poor. I have applied dimentionality reduction algorithms in order to visualize the data. It seems that there is not enough data for the classification.


### How to re-produce my steps
- In order to regenerate the dataset, run `create_splitted_dataset()` in `create_dataset.py`.
- In order to re-train `PointNet` on the dataset, run `main()` in `vicon_to_age/train.py`.
- In order to apply dimentions reduction and visualize the data, run `plot_all()` in `angles_to_age/dimensionality_reduction.py`.
