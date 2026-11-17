import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import ConvexHull, Delaunay

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    default="outputs/fusion/fused_object.ply",
)
parser.add_argument(
    "--output",
    default="outputs/volume",
)
parser.add_argument(
    "--poisson-depth",
    type=int,
    default=9,
)
parser.add_argument(
    "--ground-truth-volume",
    type=float,
    default=0.004096,
    help="Ground-truth volume in cubic metres",
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
    "--cap-spacing",
    type=float,
    default=0.002,
)
args = parser.parse_args()

output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

cloud = o3d.io.read_point_cloud(args.input)

if len(cloud.points) == 0:
    raise RuntimeError("Input point cloud is empty")

if not cloud.has_normals():
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.01,
            max_nn=50,
        )
    )

cloud.orient_normals_consistent_tangent_plane(30)

# 统一法线整体朝外
points = np.asarray(cloud.points).copy()
normals = np.asarray(cloud.normals).copy()
centroid = points.mean(axis=0)

orientation_score = np.median(
    np.sum(
        normals * (points - centroid),
        axis=1,
    )
)

if orientation_score < 0:
    normals *= -1.0

cloud.normals = o3d.utility.Vector3dVector(normals)
# 从靠近转台的点估计物体底部轮廓
base_limit = points[:, 2].min() + args.base_band
base_points = points[points[:, 2] <= base_limit]

if len(base_points) < 10:
    raise RuntimeError("Not enough points near the support plane")

base_xy = base_points[:, :2]

hull_2d = ConvexHull(base_xy)
footprint = base_xy[hull_2d.vertices]

minimum_xy = footprint.min(axis=0)
maximum_xy = footprint.max(axis=0)

grid_x = np.arange(
    minimum_xy[0],
    maximum_xy[0] + args.cap_spacing,
    args.cap_spacing,
)

grid_y = np.arange(
    minimum_xy[1],
    maximum_xy[1] + args.cap_spacing,
    args.cap_spacing,
)

xx, yy = np.meshgrid(grid_x, grid_y)

grid_xy = np.stack(
    [xx.ravel(), yy.ravel()],
    axis=1,
)

footprint_test = Delaunay(footprint)
inside = footprint_test.find_simplex(grid_xy) >= 0
cap_xy = grid_xy[inside]

cap_points = np.column_stack(
    [
        cap_xy,
        np.full(
            len(cap_xy),
            args.support_z,
            dtype=np.float64,
        ),
    ]
)

# 底面法线朝下
cap_normals = np.zeros_like(cap_points)
cap_normals[:, 2] = -1.0

augmented_points = np.vstack(
    [points, cap_points]
)
augmented_normals = np.vstack(
    [normals, cap_normals]
)

cloud.points = o3d.utility.Vector3dVector(
    augmented_points
)
cloud.normals = o3d.utility.Vector3dVector(
    augmented_normals
)

print("original points:", len(points))
print("support-cap points:", len(cap_points))
print("augmented points:", len(augmented_points))
print("Running Poisson reconstruction...")

poisson_mesh, densities = (
    o3d.geometry.TriangleMesh
    .create_from_point_cloud_poisson(
        cloud,
        depth=args.poisson_depth,
        scale=1.05,
        linear_fit=True,
    )
)

# 只保留最大的连通网格
triangle_clusters, triangle_counts, _ = (
    poisson_mesh.cluster_connected_triangles()
)

triangle_clusters = np.asarray(triangle_clusters)
triangle_counts = np.asarray(triangle_counts)

largest_cluster = np.argmax(triangle_counts)

poisson_mesh.remove_triangles_by_mask(
    triangle_clusters != largest_cluster
)
poisson_mesh.remove_unreferenced_vertices()
poisson_mesh.compute_vertex_normals()

vertices = np.asarray(poisson_mesh.vertices)
faces = np.asarray(poisson_mesh.triangles)

poisson_trimesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces,
    process=True,
)

trimesh.repair.fix_normals(poisson_trimesh)

if not poisson_trimesh.is_watertight:
    trimesh.repair.fill_holes(poisson_trimesh)

poisson_watertight = poisson_trimesh.is_watertight

if poisson_watertight:
    poisson_volume = abs(poisson_trimesh.volume)
else:
    poisson_volume = np.nan

poisson_trimesh.export(
    output_dir / "poisson_mesh.ply"
)
poisson_trimesh.export(
    output_dir / "poisson_mesh.stl"
)

# 凸包作为稳定的基准结果
convex_hull = trimesh.convex.convex_hull(points)
convex_hull_volume = abs(convex_hull.volume)
convex_hull.export(output_dir / "convex_hull.ply")

# 包围盒体积作为另一项诊断
minimum = points.min(axis=0)
maximum = points.max(axis=0)
extent = maximum - minimum
bounding_box_volume = np.prod(extent)

ground_truth = args.ground_truth_volume

if np.isfinite(poisson_volume):
    poisson_error = (
        abs(poisson_volume - ground_truth)
        / ground_truth
        * 100.0
    )
else:
    poisson_error = np.nan

hull_error = (
    abs(convex_hull_volume - ground_truth)
    / ground_truth
    * 100.0
)

bbox_error = (
    abs(bounding_box_volume - ground_truth)
    / ground_truth
    * 100.0
)

metrics = (
    f"input_point_count: {len(points)}\n"
    f"poisson_vertex_count: {len(poisson_trimesh.vertices)}\n"
    f"poisson_face_count: {len(poisson_trimesh.faces)}\n"
    f"poisson_watertight: {poisson_watertight}\n"
    f"poisson_volume_m3: {poisson_volume:.9f}\n"
    f"poisson_volume_cm3: {poisson_volume * 1e6:.3f}\n"
    f"poisson_error_percent: {poisson_error:.3f}\n"
    f"convex_hull_volume_m3: {convex_hull_volume:.9f}\n"
    f"convex_hull_volume_cm3: {convex_hull_volume * 1e6:.3f}\n"
    f"convex_hull_error_percent: {hull_error:.3f}\n"
    f"bounding_box_extent_m: {extent.tolist()}\n"
    f"bounding_box_volume_m3: {bounding_box_volume:.9f}\n"
    f"bounding_box_error_percent: {bbox_error:.3f}\n"
    f"ground_truth_volume_m3: {ground_truth:.9f}\n"
    f"ground_truth_volume_cm3: {ground_truth * 1e6:.3f}\n"
)

(output_dir / "volume_metrics.txt").write_text(metrics)

print(metrics)
print("saved to:", output_dir.resolve())
