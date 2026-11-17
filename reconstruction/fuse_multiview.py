import argparse
from pathlib import Path

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
    default="outputs/multiview_reconstruction",
)
parser.add_argument(
    "--output",
    default="outputs/fusion",
)
parser.add_argument(
    "--radius",
    type=float,
    default=0.14,
)
parser.add_argument(
    "--min-z",
    type=float,
    default=0.003,
)
parser.add_argument(
    "--max-z",
    type=float,
    default=0.25,
)
parser.add_argument(
    "--voxel-size",
    type=float,
    default=0.0015,
)
args = parser.parse_args()

input_root = Path(args.input)
output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

all_points = []
all_colors = []

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

    # 撤销转台旋转，恢复到0°物体坐标系
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

    radial_distance = np.sqrt(
        points[:, 0] ** 2 + points[:, 1] ** 2
    )

    object_mask = (
        np.isfinite(points).all(axis=1)
        & (radial_distance <= args.radius)
        & (points[:, 2] >= args.min_z)
        & (points[:, 2] <= args.max_z)
    )

    points = points[object_mask]
    colors = colors[object_mask]

    print(
        f"view {view:03d}: retained {len(points)} points"
    )

    all_points.append(points)
    all_colors.append(colors)

points = np.concatenate(all_points, axis=0)
colors = np.concatenate(all_colors, axis=0)

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(points)
cloud.colors = o3d.utility.Vector3dVector(colors)

cloud = cloud.voxel_down_sample(args.voxel_size)

cloud, indices = cloud.remove_statistical_outlier(
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

ply_path = output_dir / "fused_object.ply"
o3d.io.write_point_cloud(
    str(ply_path),
    cloud,
    write_ascii=False,
)

fused_points = np.asarray(cloud.points)
fused_colors = np.asarray(cloud.colors)

np.savez_compressed(
    output_dir / "fused_object.npz",
    points=fused_points.astype(np.float32),
    colors=fused_colors.astype(np.float32),
)

minimum = fused_points.min(axis=0)
maximum = fused_points.max(axis=0)
extent = maximum - minimum

print("fused point count:", len(fused_points))
print("bounding-box minimum:", minimum)
print("bounding-box maximum:", maximum)
print("bounding-box extent:", extent)
print("saved to:", ply_path.resolve())
