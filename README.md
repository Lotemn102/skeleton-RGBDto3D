# skeleton-RGBDto3D
This project consists of 2 sub-projects:
1. Train a network for extracting 3d skeleton points from RGBD images.
2. Train a regressor for finding a person's age from its 3d skeleton.

The `Issues` section in the repository holds all my progress summaries and meeting summaries.

## Repository tree structure
This repository contains folders for both projects.

```
|---  rgbd_to_3d
|--- vicon_to_age
```

- `rgbd_to_3d` contains all code for the first project.
- `vicon_to_age` contains all code for second project.

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

  In order to solve this problems, the following steps were done:
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
|---  assets
    |--- emails
|--- data_cleaning
|--- data_quality_evaluation
|--- models
|--- noy-mark-test
|--- unittests
```



### How to re-produce my steps
- **How to generate frames from a bag file**
- **Where should I write the T-pose frames after manually detecting them**
- **How to trim the data**
- **How to validate trimming**


