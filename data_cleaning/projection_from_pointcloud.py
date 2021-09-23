import pyrealsense2 as rs
import numpy as np
import open3d as o3d

from data_cleaning.realsense_data_reader import RealSenseReader


def read_points_from_pc(bag_file_path):
    # Open bag file
    reader = RealSenseReader(bag_file_path=bag_file_path, type='DEPTH', frame_rate=30)
    pipe, _ = reader.setup_pipeline()

    # Get point cloud of first frame
    colorizer = rs.colorizer()

    # Wait for the next set of frames from the camera
    frames = pipe.wait_for_frames()

    while frames.frame_number < 5842:
        frames = pipe.wait_for_frames()

    print("Found first frame. Starting reading point cloud...")

    colorized = colorizer.process(frames)

    # Create save_to_ply object
    ply = rs.save_to_ply("pc.ply")

    # Set options to the desired values
    # In this example we'll generate a textual PLY with normals (mesh is already created by default)
    ply.set_option(rs.save_to_ply.option_ply_binary, False)
    ply.set_option(rs.save_to_ply.option_ply_normals, True)

    print("Saving to ply file...")
    # Apply the processing block to the frameset which contains the depth frame and the texture
    ply.process(colorized)
    print("Done")

    # Read points
    pcd = o3d.io.read_point_cloud("pc.ply") # TODO: Is the points are in meters?
    points = np.asarray(pcd.points)

    # Visualize pointcloud
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

    return points

def read_points_from_deprojection(bag_file_path):
    # Open bag file
    reader = RealSenseReader(bag_file_path=bag_file_path, type='BOTH', frame_rate=30)
    pipe, config = reader.setup_pipeline()
    profile = config.resolve(rs.pipeline_wrapper(pipe))

    frames = pipe.wait_for_frames()

    while frames.frame_number < 5842:
        frames = pipe.wait_for_frames()

    align_to = rs.stream.color
    align = rs.align(align_to)
    aligned_frames = align.process(frames)

    # Get depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data(), dtype=np.uint16)

    points_3d = []

    # Find 3d coordinate
    for i in range(640):
        for j in range(480):
            depth_pixel = [i, j]
            depth_value = aligned_depth_image[depth_pixel[1]][depth_pixel[0]]
            depth_value_in_meters = depth_value * depth_scale # source: https://dev.intelrealsense.com/docs/rs-align-advanced
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_value_in_meters)
            points_3d.append(depth_point) # Points are in meters!

    # Show points
    pcd = o3d.geometry.PointCloud()
    points = np.array([(point[0], point[1], point[2]) for point in points_3d])
    pcd.points = o3d.utility.Vector3dVector(points)
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(pcd)
    visualizer.run()
    visualizer.close()

    return points


if __name__ == "__main__":
    path = '../../data/Sub007_Left_Front.bag'
    points_deprojected = read_points_from_deprojection(bag_file_path=path)
    points_pc = read_points_from_pc(bag_file_path=path)


