import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


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
points = np.asarray(cloud.points)
normals = np.asarray(cloud.normals)
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
poisson_volume = abs(poisson_trimesh.volume)

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

poisson_error = (
    abs(poisson_volume - ground_truth)
    / ground_truth
    * 100.0
)

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
