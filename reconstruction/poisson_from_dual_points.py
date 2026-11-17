import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


parser = argparse.ArgumentParser()

parser.add_argument(
    "--views",
    type=int,
    nargs="+",
    default=[
        0, 30, 60, 90, 120, 150,
        180, 210, 240, 270, 300, 330,
    ],
)

parser.add_argument(
    "--upper-root",
    required=True,
)
parser.add_argument(
    "--upper-calibration",
    required=True,
)
parser.add_argument(
    "--lower-root",
    required=True,
)
parser.add_argument(
    "--lower-calibration",
    required=True,
)
parser.add_argument(
    "--output",
    required=True,
)

parser.add_argument(
    "--object-radius",
    type=float,
    default=0.09,
)
parser.add_argument(
    "--min-z",
    type=float,
    default=0.003,
)
parser.add_argument(
    "--max-z",
    type=float,
    default=0.115,
)
parser.add_argument(
    "--normal-radius",
    type=float,
    default=0.004,
)
parser.add_argument(
    "--voxel-size",
    type=float,
    default=0.001,
)
parser.add_argument(
    "--poisson-depth",
    type=int,
    default=10,
)
parser.add_argument(
    "--poisson-scale",
    type=float,
    default=1.01,
)
args = parser.parse_args()

output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

all_points = []
all_normals = []

sources = [
    (
        "upper",
        Path(args.upper_root),
        args.upper_calibration,
    ),
    (
        "lower",
        Path(args.lower_root),
        args.lower_calibration,
    ),
]


for label, reconstruction_root, calibration_path in sources:
    calibration = np.load(calibration_path)

    camera_location_world = (
        calibration["camera_to_world"][:3, 3]
        .astype(np.float64)
    )

    for view in args.views:
        path = (
            reconstruction_root
            / f"view_{view:03d}"
            / "reconstruction"
            / "reconstruction_data.npz"
        )

        data = np.load(path)
        points_world = data["points"].astype(np.float64)

        angle = np.deg2rad(-view)
        cosine = np.cos(angle)
        sine = np.sin(angle)

        rotation = np.array(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        points_canonical = (
            rotation @ points_world.T
        ).T

        camera_canonical = (
            rotation @ camera_location_world
        )

        radial = np.linalg.norm(
            points_canonical[:, :2],
            axis=1,
        )

        valid = (
            np.isfinite(points_canonical).all(axis=1)
            & (radial <= args.object_radius)
            & (points_canonical[:, 2] >= args.min_z)
            & (points_canonical[:, 2] <= args.max_z)
        )

        points_canonical = points_canonical[valid]

        if len(points_canonical) < 100:
            print(
                f"{label} view {view:03d}: "
                "too few points"
            )
            continue

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(
            points_canonical
        )

        cloud.estimate_normals(
            search_param=(
                o3d.geometry
                .KDTreeSearchParamHybrid(
                    radius=args.normal_radius,
                    max_nn=40,
                )
            )
        )

        cloud.orient_normals_towards_camera_location(
            camera_canonical
        )

        all_points.append(
            np.asarray(cloud.points).copy()
        )
        all_normals.append(
            np.asarray(cloud.normals).copy()
        )

        print(
            f"{label} view {view:03d}: "
            f"{len(points_canonical)} points"
        )


points = np.concatenate(all_points, axis=0)
normals = np.concatenate(all_normals, axis=0)

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(points)
cloud.normals = o3d.utility.Vector3dVector(normals)

print("points before downsampling:", len(cloud.points))

cloud = cloud.voxel_down_sample(
    voxel_size=args.voxel_size
)

cloud.normalize_normals()

print("points after downsampling:", len(cloud.points))

o3d.io.write_point_cloud(
    str(output_dir / "oriented_points.ply"),
    cloud,
    write_ascii=False,
)

mesh, densities = (
    o3d.geometry.TriangleMesh
    .create_from_point_cloud_poisson(
        cloud,
        depth=args.poisson_depth,
        scale=args.poisson_scale,
        linear_fit=True,
        n_threads=-1,
    )
)

mesh.compute_vertex_normals()
mesh.remove_duplicated_vertices()
mesh.remove_duplicated_triangles()
mesh.remove_degenerate_triangles()
mesh.remove_unreferenced_vertices()

o3d.io.write_triangle_mesh(
    str(output_dir / "poisson_mesh_raw.ply"),
    mesh,
    write_ascii=False,
)

np.savez_compressed(
    output_dir / "poisson_density.npz",
    density=np.asarray(densities),
)

vertices = np.asarray(mesh.vertices)
faces = np.asarray(mesh.triangles)

triangle_mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces,
    process=True,
)

metrics = (
    f"input_point_count: {len(points)}\n"
    f"downsampled_point_count: {len(cloud.points)}\n"
    f"vertex_count: {len(vertices)}\n"
    f"face_count: {len(faces)}\n"
    f"watertight: {triangle_mesh.is_watertight}\n"
    f"extent_m: {triangle_mesh.extents.tolist()}\n"
    f"volume_cm3: {abs(triangle_mesh.volume) * 1e6}\n"
)

(output_dir / "poisson_metrics.txt").write_text(
    metrics
)

print(metrics)
