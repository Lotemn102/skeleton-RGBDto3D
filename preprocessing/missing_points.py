from vicon_data_reader import VICONReader
import math
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


def check_missing_points(csv_path: str, filename: str):
    vicon_reader = VICONReader(vicon_file_path=csv_path)
    vicon_points = vicon_reader.get_points()  # Dictionary of <frame_id, List<Point>>

    max_missing_frames_per_point = {} # Dictionary of <point_index, max_number_of_repeated_frames_the_point_is_missing>
    missing_counter = {}

    for i in range(0, 39):
        missing_counter[i] = 0
        max_missing_frames_per_point[i] = 0

    for frame_idx, frame_points in enumerate(vicon_points.values()):
        for point_idx, point in enumerate(frame_points):
            if math.isnan(point.x) or math.isnan(point.y) or math.isnan(point.z):
                missing_counter[point_idx] += 1

                if missing_counter[point_idx] > max_missing_frames_per_point[point_idx]:
                    max_missing_frames_per_point[point_idx] = missing_counter[point_idx]
            else:
                missing_counter[point_idx] = 0

    x = list(max_missing_frames_per_point.keys())
    y = list(max_missing_frames_per_point.values())
    sns.set_style("darkgrid")
    plt.rcParams['xtick.labelsize'] = 7
    plt.scatter(x, y)
    plt.xticks(x)
    plt.xlabel("Point index")
    plt.ylabel("Max number of following frames the point is missing")
    plt.title("Max number of following frames points are missing in " + filename)
    plt.savefig('plots/' + filename + '.png')
    plt.close()



if __name__ == "__main__":
    for root, dirs, files in os.walk("D:\\Movement Sense Research\\Vicon Validation Study"):
        for file in files:
            if file.endswith(".csv"):
                if "Sub001" in file or "Sub002" in file:
                    continue
                else:
                    check_missing_points(root + "\\" + file, file)