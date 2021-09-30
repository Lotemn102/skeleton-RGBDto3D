"""
Rename vicon frames in the csv such that there would be 1 to 1 mapping with the realsense frame numbers.
This is because i have originally saved the trimmed vicon frames with their original vicon frame number.
"""

import os
import pandas as pd


def rename_vicon_frames(images_folder_path: str, csv_path: str):
    # Read all frame numbers in the image folder path.
    frame_numbers = []
    for root, dirs, files in os.walk(images_folder_path):
        for file in files:
            if file.endswith(".png"):
                frame_numbers.append(int(file[:-4]))

    # Sort frames.
    sorted_frame_numbers = sorted(frame_numbers)

    # Read csv file.
    file = pd.read_csv(csv_path)

    # Remove old frames column.
    file = file.drop(file.columns[0], axis=1)

    # Add a new column of the images frames names.
    # First add some padding in the beginning of the column, to fit the csv structure.
    col = ["120", "", "Frame", ""]
    col.extend(sorted_frame_numbers)
    col = pd.Series(col)

    # Add the column in the first index.
    file = pd.concat((col, file), axis=1)

    # Rename old csv
    # TODO: Maybe remove the old versions later?
    splitted = csv_path.split("/")
    file_name = splitted[-1]
    new_file_name = file_name[:-4] + "_old.csv"
    splitted = splitted[:-1]
    splitted.append(new_file_name)
    new_path = "/".join(splitted)
    os.rename(csv_path, new_path)

    # Save csv.
    new_csv_path = csv_path[:-4]
    new_csv_path = new_csv_path + ".csv"
    header = ['Trajectories']
    header.extend([' ' for _ in range(118)])
    file.to_csv(new_csv_path, index=False, header=header)

def aux():
    """
    Rename frames for all files.

    :return: None
    """
    angles = ['Front', 'Side', 'Back']
    positions = ['Stand', 'Squat', 'Tight', 'Left', 'Right']

    for i in range(2, 16):
        sub_name = 'Sub00' + str(i) if i < 10 else 'Sub0' + str(i)

        for position in positions:
            for angle in angles:
                try:
                    rename_vicon_frames(images_folder_path="../../data/{sub_name}/{sub_name}/{position}/{angle}"
                                                           "/rgb_frames".format(sub_name=sub_name, position=position,
                                                                                angle=angle),
                                        csv_path="../../data/{sub_name}/{sub_name}/{position}/{angle}/{sub_name}_"
                                                 "{position}_{angle}.csv".format(sub_name=sub_name, position=position,
                                                                                angle=angle))
                except FileExistsError:
                    print("Already exist {name}, {pos}, {angle}. Skipping.".format(name=sub_name, pos=position, angle=angle))
                except FileNotFoundError:
                    print("Not found {name}, {pos}, {angle}. Skipping.".format(name=sub_name, pos=position, angle=angle))



if __name__ == "__main__":
    pass