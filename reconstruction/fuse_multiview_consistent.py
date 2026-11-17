import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


parser = argparse.ArgumentParser()
parser.add_argument(
    "--views",
    type=int,
    nargs="+",
    default=[0, 60, 120, 180, 240, 300],
)
parser.add_argument(
    "--input",
    default="outputs/multiview_reconstruction_exr_calibrated",
)
parser.add_argument(
    "--output",
    default="outputs/fusion_exr_consistent",
)
parser.add_argument("--radius", type=float, default=0.14)
parser.add_argument("--min-z", type=float, default=0.003)
parser.add_argument("--max-z", type=float, default=0.25)
parser.add_argument("--support-radius", type=float, default=0.003)
parser.add_argument("--min-views", type=int, default=2)
parser.add_argument("--voxel-size", type=float, default=0.0015)
args = parser.parse_args()

input_root = Path(args.input)
output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

view_points = []
view_colors = []

for view in args.views:
    path = (
        input_root
        / f"view_{view:03d}"
        / "reconstruction"
        / "reconstruction_data.npz"
    )

    data = np.load(path)

    points = data["points"].astype(np.float64)
    colors = data["colors"].astype(np.float64)

    # 撤销转台旋转
    angle = np.deg2rad(-view)
    cosine = np.cos(angle)
    sine = np.sin(angle)

    rotation = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    points = points @ rotation.T

    radial = np.linalg.norm(
        points[:, :2],
        axis=1,
    )

    mask = (
        np.isfinite(points).all(axis=1)
        & (radial <= args.radius)
        & (points[:, 2] >= args.min_z)
        & (points[:, 2] <= args.max_z)
    )

    points = points[mask]
    colors = colors[mask]

    print(
        f"view {view:03d}: input points={len(points)}"
    )

    view_points.append(points)
    view_colors.append(colors)

trees = [
    cKDTree(points)
    for points in view_points
]

supported_points = []
supported_colors = []

for index, view in enumerate(args.views):
    points = view_points[index]
    colors = view_colors[index]

    # 当前视角自身算一次观测
    support_count = np.ones(
        len(points),
        dtype=np.int32,
    )

    for other_index, other_tree in enumerate(trees):
        if other_index == index:
            continue

        distances, _ = other_tree.query(
            points,
            k=1,
            distance_upper_bound=args.support_radius,
            workers=-1,
        )

        support_count += np.isfinite(
            distances
        ).astype(np.int32)

    keep = support_count >= args.min_views

    print(
        f"view {view:03d}: supported "
        f"{keep.sum()}/{len(points)}"
    )

    supported_points.append(points[keep])
    supported_colors.append(colors[keep])

points = np.concatenate(
    supported_points,
    axis=0,
)
colors = np.concatenate(
    supported_colors,
    axis=0,
)

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(points)
cloud.colors = o3d.utility.Vector3dVector(colors)

cloud = cloud.voxel_down_sample(
    args.voxel_size
)

cloud, _ = cloud.remove_statistical_outlier(
    nb_neighbors=30,
    std_ratio=2.0,
)

cloud.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.01,
        max_nn=50,
    )
)

cloud.orient_normals_consistent_tangent_plane(30)

final_points = np.asarray(cloud.points).copy()
final_colors = np.asarray(cloud.colors).copy()

o3d.io.write_point_cloud(
    str(output_dir / "fused_object.ply"),
    cloud,
    write_ascii=False,
)

np.savez_compressed(
    output_dir / "fused_object.npz",
    points=final_points.astype(np.float32),
    colors=final_colors.astype(np.float32),
)

raw_extent = np.ptp(
    final_points,
    axis=0,
)

robust_extent = (
    np.percentile(final_points, 99, axis=0)
    - np.percentile(final_points, 1, axis=0)
)

print("final point count:", len(final_points))
print("raw extent:", raw_extent)
print("1%-99% extent:", robust_extent)
print(
    "saved to:",
    (output_dir / "fused_object.ply").resolve(),
)
