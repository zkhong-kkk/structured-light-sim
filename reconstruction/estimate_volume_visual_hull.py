import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


parser = argparse.ArgumentParser()
parser.add_argument(
    "--views",
    type=int,
    nargs="+",
    default=[0, 60, 120, 180, 240, 300],
)
parser.add_argument(
    "--input",
    default="outputs/multiview_reconstruction_exr_lshape_calibrated",
)
parser.add_argument(
    "--calibration",
    default="configs/sim_calibration.npz",
)
parser.add_argument(
    "--output",
    default="outputs/volume_lshape_visual_hull",
)
parser.add_argument("--voxel-size", type=float, default=0.002)
parser.add_argument("--min-views", type=int, default=5)
parser.add_argument("--x-min", type=float, default=-0.09)
parser.add_argument("--x-max", type=float, default=0.09)
parser.add_argument("--y-min", type=float, default=-0.09)
parser.add_argument("--y-max", type=float, default=0.09)
parser.add_argument("--z-min", type=float, default=0.0)
parser.add_argument("--z-max", type=float, default=0.17)
parser.add_argument(
    "--ground-truth-volume",
    type=float,
    default=0.003072,
)
args = parser.parse_args()

input_root = Path(args.input)
output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

calibration = np.load(args.calibration)

camera_K = calibration["camera_K"].astype(np.float64)
camera_to_world = calibration["camera_to_world"].astype(np.float64)

width = int(calibration["image_width"])
height = int(calibration["image_height"])

cv_to_blender = np.diag(
    [1.0, -1.0, -1.0, 1.0]
)

camera_cv_to_world = (
    camera_to_world @ cv_to_blender
)

world_to_camera_cv = np.linalg.inv(
    camera_cv_to_world
)

silhouettes = {}

kernel = np.ones(
    (7, 7),
    dtype=np.uint8,
)

for view in args.views:
    path = (
        input_root
        / f"view_{view:03d}"
        / "reconstruction"
        / "reconstruction_data.npz"
    )

    data = np.load(path)

    rows = data["pixel_rows"].astype(np.int32)
    cols = data["pixel_cols"].astype(np.int32)

    valid_pixels = (
        (rows >= 0)
        & (rows < height)
        & (cols >= 0)
        & (cols < width)
    )

    rows = rows[valid_pixels]
    cols = cols[valid_pixels]

    sparse_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    sparse_mask[rows, cols] = 255

    closed_mask = cv2.morphologyEx(
        sparse_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        closed_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    silhouette = np.zeros_like(
        sparse_mask
    )

    for contour in contours:
        if cv2.contourArea(contour) >= 100:
            cv2.drawContours(
                silhouette,
                [contour],
                contourIdx=-1,
                color=255,
                thickness=cv2.FILLED,
            )

    silhouettes[view] = silhouette

    cv2.imwrite(
        str(
            output_dir
            / f"silhouette_{view:03d}.png"
        ),
        silhouette,
    )

    print(
        f"view {view:03d}: silhouette pixels="
        f"{np.count_nonzero(silhouette)}"
    )

pitch = args.voxel_size

x_values = np.arange(
    args.x_min + pitch / 2.0,
    args.x_max,
    pitch,
)

y_values = np.arange(
    args.y_min + pitch / 2.0,
    args.y_max,
    pitch,
)

z_values = np.arange(
    args.z_min + pitch / 2.0,
    args.z_max,
    pitch,
)

xx, yy, zz = np.meshgrid(
    x_values,
    y_values,
    z_values,
    indexing="ij",
)

voxel_points = np.column_stack(
    [
        xx.ravel(),
        yy.ravel(),
        zz.ravel(),
    ]
)

support_count = np.zeros(
    len(voxel_points),
    dtype=np.uint8,
)

for view in args.views:
    angle = np.deg2rad(view)

    cosine = np.cos(angle)
    sine = np.sin(angle)

    canonical_to_world = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    points_world = (
        voxel_points
        @ canonical_to_world.T
    )

    world_homogeneous = np.column_stack(
        [
            points_world,
            np.ones(len(points_world)),
        ]
    )

    points_camera = (
        world_to_camera_cv
        @ world_homogeneous.T
    ).T[:, :3]

    positive_depth = points_camera[:, 2] > 0.0

    projected = (
        camera_K
        @ points_camera.T
    ).T

    u = np.zeros(
        len(voxel_points),
        dtype=np.int32,
    )

    v = np.zeros(
        len(voxel_points),
        dtype=np.int32,
    )

    u[positive_depth] = np.rint(
        projected[positive_depth, 0]
        / projected[positive_depth, 2]
    ).astype(np.int32)

    v[positive_depth] = np.rint(
        projected[positive_depth, 1]
        / projected[positive_depth, 2]
    ).astype(np.int32)

    inside_image = (
        positive_depth
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )

    supported = np.zeros(
        len(voxel_points),
        dtype=bool,
    )

    valid_indices = np.nonzero(
        inside_image
    )[0]

    supported[valid_indices] = (
        silhouettes[view][
            v[valid_indices],
            u[valid_indices],
        ]
        > 0
    )

    support_count += supported.astype(
        np.uint8
    )

    print(
        f"view {view:03d}: supported voxels="
        f"{supported.sum()}"
    )

occupied = (
    support_count >= args.min_views
)

occupied_points = voxel_points[
    occupied
]

voxel_volume = pitch ** 3
volume = occupied.sum() * voxel_volume

ground_truth = args.ground_truth_volume

error_percent = (
    abs(volume - ground_truth)
    / ground_truth
    * 100.0
)

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(
    occupied_points
)

o3d.io.write_point_cloud(
    str(output_dir / "visual_hull_voxels.ply"),
    cloud,
    write_ascii=False,
)

np.savez_compressed(
    output_dir / "visual_hull_data.npz",
    occupied=occupied.reshape(xx.shape),
    x=x_values,
    y=y_values,
    z=z_values,
    support_count=support_count.reshape(xx.shape),
)

if len(occupied_points) > 0:
    extent = (
        occupied_points.max(axis=0)
        - occupied_points.min(axis=0)
        + pitch
    )
else:
    extent = np.zeros(3)

metrics = (
    f"grid_shape: {xx.shape}\n"
    f"total_voxels: {len(voxel_points)}\n"
    f"occupied_voxels: {occupied.sum()}\n"
    f"min_views: {args.min_views}\n"
    f"voxel_size_m: {pitch}\n"
    f"extent_m: {extent.tolist()}\n"
    f"volume_m3: {volume:.9f}\n"
    f"volume_cm3: {volume * 1e6:.3f}\n"
    f"ground_truth_cm3: {ground_truth * 1e6:.3f}\n"
    f"error_percent: {error_percent:.3f}\n"
)

(output_dir / "volume_metrics.txt").write_text(
    metrics
)

print(metrics)
print("saved to:", output_dir.resolve())
