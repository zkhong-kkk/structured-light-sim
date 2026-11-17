import argparse
from pathlib import Path

import h5py
import numpy as np
import open3d as o3d
import cv2
from scipy.ndimage import distance_transform_edt


parser = argparse.ArgumentParser()
parser.add_argument(
    "--projector-u-calibration",
    default=None,
)
parser.add_argument(
    "--local-range-threshold",
    type=float,
    default=0.01,
)
parser.add_argument(
    "--roi-radius",
    type=float,
    default=10.0,
)

parser.add_argument(
    "--roi-min-z",
    type=float,
    default=-10.0,
)

parser.add_argument(
    "--roi-max-z",
    type=float,
    default=10.0,
)
parser.add_argument(
    "--local-kernel",
    type=int,
    default=5,
)
parser.add_argument(
    "--depth-cutoff",
    type=float,
    default=2.0,
)
parser.add_argument(
    "--flip-projector-u",
    action="store_true",
)
parser.add_argument(
    "--phase",
    default="outputs/phase_unwrapped/unwrapped_phase_data.npz",
)
parser.add_argument(
    "--calibration",
    default="configs/sim_calibration.npz",
)
parser.add_argument(
    "--capture",
    default="outputs/multifreq/freq_12/phase_000/0.hdf5",
)
parser.add_argument(
    "--output",
    default="outputs/reconstruction",
)
parser.add_argument(
    "--max-distance",
    type=float,
    default=2.0,
)
args = parser.parse_args()

output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

phase_data = np.load(args.phase)
calibration = np.load(args.calibration)

projector_u = phase_data["projector_u"]
valid_mask = phase_data["valid_mask"].astype(bool)

camera_K = calibration["camera_K"]
projector_K = calibration["projector_K"]
camera_to_world = calibration["camera_to_world"]
projector_to_world = calibration["projector_to_world"]

width = int(calibration["image_width"])
height = int(calibration["image_height"])

with h5py.File(args.capture, "r") as f:
    raw_rgb = f["colors"][:]

    if np.issubdtype(raw_rgb.dtype, np.integer):
        color_scale = np.iinfo(raw_rgb.dtype).max
    else:
        color_scale = max(float(raw_rgb.max()), 1.0)

    rgb = np.clip(
        raw_rgb.astype(np.float32) / color_scale,
        0.0,
        1.0,
    )
    depth = f["depth"][:].astype(np.float32)


def world_rays(u, v, K, pose):
    """把图像像素转换成世界坐标系单位射线。"""

    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    # BlenderProc图像坐标到相机坐标
    pixels = np.stack(
        [
            u,
            (height - 1) - v,
            np.ones_like(u),
        ],
        axis=1,
    )

    directions = (
        np.linalg.inv(K) @ pixels.T
    ).T

    # Blender/OpenGL相机朝局部-Z方向观察
    directions[:, 2] *= -1.0

    rotation = pose[:3, :3]
    directions_world = directions @ rotation.T

    directions_world /= np.linalg.norm(
        directions_world,
        axis=1,
        keepdims=True,
    )

    return directions_world


rows, cols = np.nonzero(valid_mask)

camera_u = cols.astype(np.float64)
camera_v = rows.astype(np.float64)
projector_columns = projector_u[rows, cols].astype(np.float64)
if args.flip_projector_u:
    projector_columns = (width - 1) - projector_columns
if args.projector_u_calibration is not None:
    projector_u_calibration = np.load(
        args.projector_u_calibration
    )

    projector_columns = (
        float(projector_u_calibration["scale"])
        * projector_columns
        + float(projector_u_calibration["offset"])
    )
camera_rays = world_rays(
    camera_u,
    camera_v,
    camera_K,
    camera_to_world,
)

# 一个投影仪列对应一个穿过投影仪光心的平面。
projector_top_rays = world_rays(
    projector_columns,
    np.zeros_like(projector_columns),
    projector_K,
    projector_to_world,
)

projector_bottom_rays = world_rays(
    projector_columns,
    np.full_like(projector_columns, height - 1),
    projector_K,
    projector_to_world,
)

plane_normals = np.cross(
    projector_top_rays,
    projector_bottom_rays,
)
plane_normals /= np.linalg.norm(
    plane_normals,
    axis=1,
    keepdims=True,
)

camera_center = camera_to_world[:3, 3]
projector_center = projector_to_world[:3, 3]

numerator = np.einsum(
    "ij,j->i",
    plane_normals,
    projector_center - camera_center,
)

denominator = np.einsum(
    "ij,ij->i",
    plane_normals,
    camera_rays,
)

distance = numerator / denominator

preliminary_mask = (
    np.isfinite(distance)
    & (np.abs(denominator) > 1e-8)
    & (distance > 0.0)
    & (distance < args.max_distance)
    & np.isfinite(depth[rows, cols])
    & (depth[rows, cols] < args.depth_cutoff)
)

# 将三角测量距离恢复到图像平面
range_image = np.zeros(
    (height, width),
    dtype=np.float32,
)

range_image[
    rows[preliminary_mask],
    cols[preliminary_mask],
] = distance[preliminary_mask].astype(np.float32)

range_mask = range_image > 0.0

_, nearest = distance_transform_edt(
    ~range_mask,
    return_indices=True,
)

filled_range = range_image.copy()
filled_range[~range_mask] = range_image[
    nearest[0][~range_mask],
    nearest[1][~range_mask],
]

kernel_size = args.local_kernel
if kernel_size % 2 == 0:
    kernel_size += 1

median_range = cv2.medianBlur(
    filled_range,
    kernel_size,
)

local_range_error = np.abs(
    distance
    - median_range[rows, cols]
)

keep = (
    preliminary_mask
    & (local_range_error <= args.local_range_threshold)
)

candidate_points = (
    camera_center[np.newaxis, :]
    + distance[:, np.newaxis] * camera_rays
)

radial_distance = np.linalg.norm(
    candidate_points[:, :2],
    axis=1,
)

roi_mask = (
    np.isfinite(candidate_points).all(axis=1)
    & (radial_distance <= args.roi_radius)
    & (candidate_points[:, 2] >= args.roi_min_z)
    & (candidate_points[:, 2] <= args.roi_max_z)
)

keep &= roi_mask

points = candidate_points[keep]

colors = rgb[rows, cols][keep]
kept_rows = rows[keep]
kept_cols = cols[keep]

# 使用仿真深度生成真值点，用来计算重建误差。
pixels = np.stack(
    [
        kept_cols,
        (height - 1) - kept_rows,
        np.ones_like(kept_cols),
    ],
    axis=1,
).astype(np.float64)

points_camera = (
    np.linalg.inv(camera_K) @ pixels.T
).T

points_camera *= depth[
    kept_rows,
    kept_cols,
    np.newaxis,
]

points_camera[:, 2] *= -1.0

ground_truth = (
    points_camera @ camera_to_world[:3, :3].T
    + camera_center
)

errors = np.linalg.norm(
    points - ground_truth,
    axis=1,
)

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)

ply_path = output_dir / "structured_light_point_cloud.ply"
success = o3d.io.write_point_cloud(
    str(ply_path),
    point_cloud,
    write_ascii=False,
)

np.savez_compressed(
    output_dir / "reconstruction_data.npz",
    points=points.astype(np.float32),
    colors=colors.astype(np.float32),
    pixel_rows=kept_rows,
    pixel_cols=kept_cols,
    errors=errors.astype(np.float32),
)

metrics = (
    f"point_count: {len(points)}\n"
    f"rmse_m: {np.sqrt(np.mean(errors**2)):.8f}\n"
    f"median_error_m: {np.median(errors):.8f}\n"
    f"p95_error_m: {np.percentile(errors, 95):.8f}\n"
    f"max_error_m: {np.max(errors):.8f}\n"
)

(output_dir / "metrics.txt").write_text(metrics)

print(metrics)
print("PLY written:", success)
print("saved to:", output_dir.resolve())
