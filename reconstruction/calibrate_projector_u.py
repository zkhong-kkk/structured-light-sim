import argparse
from pathlib import Path

import h5py
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--phase", required=True)
parser.add_argument("--capture", required=True)
parser.add_argument(
    "--calibration",
    default="configs/sim_calibration.npz",
)
parser.add_argument(
    "--output",
    default="configs/projector_u_calibration.npz",
)
parser.add_argument(
    "--depth-cutoff",
    type=float,
    default=2.0,
)
args = parser.parse_args()

phase_data = np.load(args.phase)
calibration = np.load(args.calibration)

projector_u = phase_data["projector_u"]
valid = phase_data["valid_mask"].astype(bool)

camera_K = calibration["camera_K"]
projector_K = calibration["projector_K"]
camera_to_world = calibration["camera_to_world"]
projector_to_world = calibration["projector_to_world"]

width = int(calibration["image_width"])
height = int(calibration["image_height"])

with h5py.File(args.capture, "r") as f:
    depth = f["depth"][:].astype(np.float64)

rows, cols = np.nonzero(
    valid
    & np.isfinite(depth)
    & (depth > 0.0)
    & (depth < args.depth_cutoff)
)

depth_values = depth[rows, cols]

camera_pixels = np.stack(
    [
        cols,
        (height - 1) - rows,
        np.ones_like(cols),
    ],
    axis=1,
).astype(np.float64)

points_camera = (
    np.linalg.inv(camera_K)
    @ camera_pixels.T
).T

points_camera *= depth_values[:, np.newaxis]
points_camera[:, 2] *= -1.0

points_world = (
    points_camera @ camera_to_world[:3, :3].T
    + camera_to_world[:3, 3]
)

world_homogeneous = np.column_stack(
    [
        points_world,
        np.ones(len(points_world)),
    ]
)

points_projector = (
    np.linalg.inv(projector_to_world)
    @ world_homogeneous.T
).T

points_projector[:, 2] *= -1.0

projected = (
    projector_K
    @ points_projector[:, :3].T
).T

expected_u = projected[:, 0] / projected[:, 2]

# 当前解码坐标需要先做水平方向反转
measured_u = (
    (width - 1)
    - projector_u[rows, cols]
).astype(np.float64)

fit_valid = (
    np.isfinite(measured_u)
    & np.isfinite(expected_u)
    & (points_projector[:, 2] > 0.0)
    & (expected_u >= 0.0)
    & (expected_u < width)
)

x = measured_u[fit_valid]
y = expected_u[fit_valid]

keep = np.ones(len(x), dtype=bool)

for _ in range(10):
    design = np.column_stack(
        [
            x[keep],
            np.ones(keep.sum()),
        ]
    )

    scale, offset = np.linalg.lstsq(
        design,
        y[keep],
        rcond=None,
    )[0]

    residual = y - (scale * x + offset)

    median = np.median(residual[keep])
    mad = np.median(
        np.abs(residual[keep] - median)
    )

    sigma = max(
        1.4826 * mad,
        0.1,
    )

    new_keep = (
        np.abs(residual - median)
        <= 3.0 * sigma
    )

    if np.array_equal(new_keep, keep):
        break

    keep = new_keep

before_error = y[keep] - x[keep]
after_error = (
    y[keep]
    - (scale * x[keep] + offset)
)

output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    output_path,
    scale=np.float64(scale),
    offset=np.float64(offset),
    inlier_count=np.int64(keep.sum()),
)

print("sample count:", len(x))
print("inlier count:", keep.sum())
print("scale:", scale)
print("offset:", offset)
print(
    "RMSE before:",
    np.sqrt(np.mean(before_error**2)),
)
print(
    "RMSE after:",
    np.sqrt(np.mean(after_error**2)),
)
print("saved to:", output_path.resolve())
