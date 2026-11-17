import argparse
from pathlib import Path

import h5py
import numpy as np
import open3d as o3d
import trimesh
import cv2
from scipy.ndimage import distance_transform_edt

parser = argparse.ArgumentParser()
parser.add_argument("--fill-kernel", type=int, default=11)
parser.add_argument(
    "--min-component-area",
    type=int,
    default=20,
)
parser.add_argument("--fill-iterations", type=int, default=2)
parser.add_argument(
    "--views",
    type=int,
    nargs="+",
    default=[0, 60, 120, 180, 240, 300],
)
parser.add_argument(
    "--reconstruction-root",
    default="outputs/multiview_reconstruction",
)
parser.add_argument(
    "--capture-root",
    default="outputs/multiview",
)
parser.add_argument(
    "--calibration",
    default="configs/sim_calibration.npz",
)
parser.add_argument(
    "--output",
    default="outputs/tsdf",
)
parser.add_argument("--voxel-size", type=float, default=0.0015)
parser.add_argument("--sdf-trunc", type=float, default=0.006)
parser.add_argument("--object-radius", type=float, default=0.14)
parser.add_argument("--min-z", type=float, default=0.003)
parser.add_argument("--max-z", type=float, default=0.25)
args = parser.parse_args()

reconstruction_root = Path(args.reconstruction_root)
capture_root = Path(args.capture_root)
output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

calibration = np.load(args.calibration)

camera_K = calibration["camera_K"].astype(np.float64)
camera_to_world = calibration["camera_to_world"].astype(np.float64)

width = int(calibration["image_width"])
height = int(calibration["image_height"])

intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width,
    height,
    camera_K[0, 0],
    camera_K[1, 1],
    camera_K[0, 2],
    camera_K[1, 2],
)

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=args.voxel_size,
    sdf_trunc=args.sdf_trunc,
    color_type=(
        o3d.pipelines.integration
        .TSDFVolumeColorType.RGB8
    ),
    volume_unit_resolution=16,
    depth_sampling_stride=1,
)

# OpenCV相机坐标转换到Blender相机坐标：
# x保持，y和z反向。
cv_to_blender = np.diag(
    [1.0, -1.0, -1.0, 1.0]
)

camera_cv_to_world = camera_to_world @ cv_to_blender

for view in args.views:
    reconstruction_path = (
        reconstruction_root
        / f"view_{view:03d}"
        / "reconstruction"
        / "reconstruction_data.npz"
    )

    capture_path = (
        capture_root
        / f"view_{view:03d}"
        / "freq_12"
        / "phase_000"
        / "0.hdf5"
    )

    data = np.load(reconstruction_path)

    points_world = data["points"].astype(np.float64)
    pixel_rows = data["pixel_rows"].astype(np.int64)
    pixel_cols = data["pixel_cols"].astype(np.int64)

    # 把当前转台角度下的世界坐标恢复到物体0°坐标系
    angle = np.deg2rad(-view)
    cosine = np.cos(angle)
    sine = np.sin(angle)

    world_to_canonical = np.array(
        [
            [cosine, -sine, 0.0, 0.0],
            [sine, cosine, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    points_homogeneous = np.column_stack(
        [points_world, np.ones(len(points_world))]
    )

    points_canonical = (
        world_to_canonical @ points_homogeneous.T
    ).T[:, :3]

    radial = np.linalg.norm(
        points_canonical[:, :2],
        axis=1,
    )

    object_mask = (
        np.isfinite(points_canonical).all(axis=1)
        & (radial <= args.object_radius)
        & (points_canonical[:, 2] >= args.min_z)
        & (points_canonical[:, 2] <= args.max_z)
    )

    points_canonical = points_canonical[object_mask]
    rows = pixel_rows[object_mask]
    cols = pixel_cols[object_mask]

    camera_cv_to_canonical = (
        world_to_canonical @ camera_cv_to_world
    )
    canonical_to_camera_cv = np.linalg.inv(
        camera_cv_to_canonical
    )

    canonical_homogeneous = np.column_stack(
        [
            points_canonical,
            np.ones(len(points_canonical)),
        ]
    )

    points_camera = (
        canonical_to_camera_cv
        @ canonical_homogeneous.T
    ).T[:, :3]

    depth_values = points_camera[:, 2]

    positive = (
        np.isfinite(depth_values)
        & (depth_values > 0.0)
        & (depth_values < 2.0)
    )

    rows = rows[positive]
    cols = cols[positive]
    depth_values = depth_values[positive]

    depth_image = np.zeros(
        (height, width),
        dtype=np.float32,
    )
    depth_image[rows, cols] = depth_values.astype(np.float32)
    # 对结构光有效区域内的小孔进行补全
    sparse_mask = depth_image > 0.0
    
    kernel_size = args.fill_kernel
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.uint8,
    )
    
    closed_mask = cv2.morphologyEx(
        sparse_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
        iterations=args.fill_iterations,
    )
    
    # 只保留最大的图像连通区域，防止填充离散噪点
    label_count, labels, statistics, _ = (
        cv2.connectedComponentsWithStats(
            closed_mask,
            connectivity=8,
        )
    )
    
    # 保留所有足够大的连通区域。
    # 立方体的正面、顶面和侧面在深度图中可能彼此断开，
    # 不能只保留最大的一个区域。
    fill_mask = np.zeros_like(
        sparse_mask,
        dtype=bool,
    )

    for label_index in range(1, label_count):
        component_area = statistics[
            label_index,
            cv2.CC_STAT_AREA,
        ]

        if component_area >= args.min_component_area:
            fill_mask |= labels == label_index

    # 如果形态学操作没有留下区域，则退回原始有效像素
    if not np.any(fill_mask):
        fill_mask = sparse_mask.copy()   
    distance, nearest_indices = distance_transform_edt(
        ~sparse_mask,
        return_indices=True,
    )
    
    max_fill_distance = (
        kernel_size * args.fill_iterations
    )
    
    fill_mask &= distance <= max_fill_distance
    
    nearest_depth = depth_image[
        nearest_indices[0],
        nearest_indices[1],
    ]
    
    depth_image[fill_mask] = nearest_depth[fill_mask]
    depth_image[~fill_mask] = 0.0
    
    print(
        f"view {view:03d}: depth pixels "
        f"{sparse_mask.sum()} -> {fill_mask.sum()}"
    )
    with h5py.File(capture_path, "r") as f:
        color_image = f["colors"][:].astype(np.uint8)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_image),
        o3d.geometry.Image(depth_image),
        depth_scale=1.0,
        depth_trunc=2.0,
        convert_rgb_to_intensity=False,
    )

    volume.integrate(
        rgbd,
        intrinsic,
        canonical_to_camera_cv,
    )

    print(
        f"view {view:03d}: "
        f"integrated {len(depth_values)} depth pixels, "
        f"range {depth_values.min():.4f}-"
        f"{depth_values.max():.4f} m"
    )

mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()
mesh.remove_duplicated_vertices()
mesh.remove_duplicated_triangles()
mesh.remove_degenerate_triangles()
mesh.remove_unreferenced_vertices()

mesh_path = output_dir / "tsdf_mesh_raw.ply"
o3d.io.write_triangle_mesh(
    str(mesh_path),
    mesh,
    write_ascii=False,
)

cloud = volume.extract_point_cloud()
o3d.io.write_point_cloud(
    str(output_dir / "tsdf_point_cloud.ply"),
    cloud,
    write_ascii=False,
)

vertices = np.asarray(mesh.vertices).copy()
faces = np.asarray(mesh.triangles).copy()

if len(vertices) == 0 or len(faces) == 0:
    raise RuntimeError("TSDF produced an empty mesh")

triangle_mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces,
    process=True,
)

watertight = triangle_mesh.is_watertight
volume_m3 = (
    abs(triangle_mesh.volume)
    if watertight
    else np.nan
)

minimum = vertices.min(axis=0)
maximum = vertices.max(axis=0)
extent = maximum - minimum

metrics = (
    f"vertex_count: {len(vertices)}\n"
    f"face_count: {len(faces)}\n"
    f"body_count: {triangle_mesh.body_count}\n"
    f"watertight: {watertight}\n"
    f"extent_m: {extent.tolist()}\n"
    f"volume_m3: {volume_m3}\n"
    f"volume_cm3: {volume_m3 * 1e6}\n"
)

(output_dir / "tsdf_metrics.txt").write_text(metrics)

print(metrics)
print("saved to:", output_dir.resolve())
