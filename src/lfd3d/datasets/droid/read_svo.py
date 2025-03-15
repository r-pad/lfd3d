import json
import os
from glob import glob

import pyzed.sl as sl
from tqdm import tqdm


def glob_with_progress(pattern):
    # Find all paths that match up to the wildcard
    base_parts = pattern.split("*")[0]
    base_dir = os.path.dirname(base_parts.rstrip("/"))

    # Generate the complete pattern including wildcards
    search_pattern = pattern

    # Use glob to get all matching directories first
    matching_dirs = glob(os.path.dirname(search_pattern))

    all_files = []
    # Show progress through directories
    for dir_path in tqdm(matching_dirs, desc="Searching directories"):
        if os.path.isdir(dir_path):
            # Get files in this directory matching the basename pattern
            file_pattern = os.path.join(dir_path, os.path.basename(pattern))
            files = glob(file_pattern)
            all_files.extend(files)

    return all_files


def get_camera_parameters(svo_path):
    # Create a Camera object
    zed = sl.Camera()

    # Create initial parameters
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening SVO file: {status}")
        return

    # Get camera information
    camera_info = zed.get_camera_information()

    # Get calibration parameters
    calibration_params = camera_info.camera_configuration.calibration_parameters

    # Get intrinsics for left and right cameras
    left_intrinsics = calibration_params.left_cam
    right_intrinsics = calibration_params.right_cam

    # Extract focal length, principal point, and distortion
    left_fx, left_fy = left_intrinsics.fx, left_intrinsics.fy
    left_cx, left_cy = left_intrinsics.cx, left_intrinsics.cy
    left_distortion = left_intrinsics.disto

    right_fx, right_fy = right_intrinsics.fx, right_intrinsics.fy
    right_cx, right_cy = right_intrinsics.cx, right_intrinsics.cy
    right_distortion = right_intrinsics.disto

    # Get extrinsics (stereo transform - rotation and translation)
    stereo_transform = calibration_params.stereo_transform
    rotation = stereo_transform.get_rotation_matrix()
    translation = stereo_transform.get_translation()

    # Access components using x, y, z properties
    tx = translation.get()[0]
    ty = translation.get()[1]
    tz = translation.get()[2]

    # Baseline is the absolute value of tx (in mm)
    baseline = abs(tx)
    # Get resolution
    resolution = camera_info.camera_configuration.resolution

    # Close the camera
    zed.close()

    return {
        "fx": left_fx,
        "fy": left_fy,
        "cx": left_cx,
        "cy": left_cy,
        "baseline": baseline,
    }


def process_svo_file(svo_path):
    parameters = get_camera_parameters(svo_path)
    # You can add additional processing or saving of parameters here
    return parameters


def main():
    base_dir = "/home/sriram/Desktop/autobot_mount/1.0.1/"

    svo_files = glob_with_progress(f"{base_dir}/*/*/*/*/recordings/SVO/*")

    intrinsics = {}
    for file in tqdm(svo_files, desc="Processing SVO files"):
        svo_id = int(os.path.basename(file).split(".")[0])
        if svo_id in intrinsics:
            continue

        params = process_svo_file(file)
        intrinsics[svo_id] = params

    with open("zed_intrinsics.json", "w") as f:
        json.dump(intrinsics, f, indent=4)


if __name__ == "__main__":
    main()
