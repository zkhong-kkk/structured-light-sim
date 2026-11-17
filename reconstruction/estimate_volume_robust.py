import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    default="outputs/fusion_exr_consistent/fused_object.npz",
)
parser.add_argument(
    "--output",
    default="outputs/volume_exr_robust",
)
parser.add_argument(
    "--trim-percent",
    type=float,
    default=1.0,
)
parser.add_argument(
    "--support-z",
    type=float,
    default=0.0,
)
parser.add_argument(
    "--base-band",
    type=float,
    default=0.015,
)
parser.add_argument(
    "--ground-truth-volume",
    type=float,
    default=0.004096,
)
args = parser.parse_args()

output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

data = np.load(args.input)
points = data["points"].astype(np.float64)

finite = np.isfinite(points).all(axis=1)
points = points[finite]

q = args.trim_percent

x_low, x_high = np.percentile(
    points[:, 0],
    [q, 100.0 - q],
)
y_low, y_high = np.percentile(
    points[:, 1],
    [q, 100.0 - q],
)

# 顶面使用99.9分位，底面由转台平面确定
z_high = np.percentile(
    points[:, 2],
    99.9,
)

trim_mask = (
    (points[:, 0] >= x_low)
    & (points[:, 0] <= x_high)
    & (points[:, 1] >= y_low)
    & (points[:, 1] <= y_high)
    & (points[:, 2] >= args.support_z)
    & (points[:, 2] <= z_high)
)

trimmed_points = points[trim_mask]

base_limit = (
    trimmed_points[:, 2].min()
    + args.base_band
)

base_points = trimmed_points[
    trimmed_points[:, 2] <= base_limit
]

if len(base_points) < 3:
    raise RuntimeError(
        "Not enough points near support plane"
    )

footprint_hull = ConvexHull(
    base_points[:, :2]
)

footprint_xy = base_points[
    footprint_hull.vertices,
    :2,
]

support_points = np.column_stack(
    [
        footprint_xy,
        np.full(
            len(footprint_xy),
            args.support_z,
        ),
    ]
)

augmented_points = np.vstack(
    [
        trimmed_points,
        support_points,
    ]
)

convex_mesh = trimesh.convex.convex_hull(
    augmented_points
)

volume = abs(convex_mesh.volume)
ground_truth = args.ground_truth_volume

error_percent = (
    abs(volume - ground_truth)
    / ground_truth
    * 100.0
)

convex_mesh.export(
    output_dir / "robust_convex_hull.ply"
)
convex_mesh.export(
    output_dir / "robust_convex_hull.stl"
)

extent = np.array(
    [
        x_high - x_low,
        y_high - y_low,
        z_high - args.support_z,
    ]
)

metrics = (
    f"input_points: {len(points)}\n"
    f"trimmed_points: {len(trimmed_points)}\n"
    f"watertight: {convex_mesh.is_watertight}\n"
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
