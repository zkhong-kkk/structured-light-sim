import blenderproc as bproc

import argparse
from pathlib import Path
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "--camera-id",
    choices=[
        "upper_left",
        "upper_right",
        "lower_left",
        "lower_right",
    ],
    required=True,
)
parser.add_argument(
    "--output",
    required=True,
)
args = parser.parse_args()

output_path = Path(args.output).resolve()
output_path.parent.mkdir(parents=True, exist_ok=True)

bproc.init()
bproc.camera.set_resolution(640, 480)

target = np.array([0.0, 0.0, 0.05])

rigs = {
    "upper_left": {
        "camera": [-0.07, -0.07, 0.50],
        "projector": [0.0, 0.08, 0.52],
    },
    "upper_right": {
        "camera": [0.07, -0.07, 0.50],
        "projector": [0.0, 0.08, 0.52],
    },
    "lower_left": {
        "camera": [-0.28, -0.42, 0.025],
        "projector": [0.0, -0.48, 0.015],
    },
    "lower_right": {
        "camera": [0.28, -0.42, 0.025],
        "projector": [0.0, -0.48, 0.015],
    },
}

rig = rigs[args.camera_id]

camera_location = np.array(
    rig["camera"],
    dtype=np.float64,
)
camera_rotation = bproc.camera.rotation_from_forward_vec(
    target - camera_location
)
camera_to_world = bproc.math.build_transformation_mat(
    camera_location,
    camera_rotation,
)
bproc.camera.add_camera_pose(camera_to_world)

projector_location = np.array(
    rig["projector"],
    dtype=np.float64,
)
projector_rotation = bproc.camera.rotation_from_forward_vec(
    target - projector_location
)
projector_to_world = bproc.math.build_transformation_mat(
    projector_location,
    projector_rotation,
)

camera_K = bproc.camera.get_intrinsics_as_K_matrix()

# 当前仿真中投影仪使用与相机相同的视场角和分辨率
projector_K = camera_K.copy()

np.savez_compressed(
    output_path,
    camera_K=camera_K,
    projector_K=projector_K,
    camera_to_world=camera_to_world,
    projector_to_world=projector_to_world,
    image_width=np.int32(640),
    image_height=np.int32(480),
)

print("camera_id:", args.camera_id)
print("camera K:")
print(camera_K)
print("camera to world:")
print(camera_to_world)
print("projector to world:")
print(projector_to_world)
print("saved to:", output_path)
