import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    default="outputs/fusion_exr_lshape_consistent/fused_object.npz",
)
parser.add_argument(
    "--output",
    default="outputs/volume_lshape_slices",
)
parser.add_argument(
    "--voxel-size",
    type=float,
    default=0.002,
)
parser.add_argument(
    "--slice-band",
    type=float,
    default=0.003,
)
parser.add_argument(
    "--support-z",
    type=float,
    default=0.0,
)
parser.add_argument(
    "--trim-percent",
    type=float,
    default=1.0,
)
parser.add_argument(
    "--close-iterations",
    type=int,
    default=2,
)
parser.add_argument(
    "--min-contour-area",
    type=float,
    default=5.0,
)
parser.add_argument(
    "--ground-truth-volume",
    type=float,
    default=0.003072,
)
args = parser.parse_args()

output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

data = np.load(args.input)
points = data["points"].astype(np.float64)
points = points[np.isfinite(points).all(axis=1)]

q = args.trim_percent

x_min, x_max = np.percentile(
    points[:, 0],
    [q, 100.0 - q],
)
y_min, y_max = np.percentile(
    points[:, 1],
    [q, 100.0 - q],
)

z_max = np.percentile(
    points[:, 2],
    99.9,
)

xy_mask = (
    (points[:, 0] >= x_min)
    & (points[:, 0] <= x_max)
    & (points[:, 1] >= y_min)
    & (points[:, 1] <= y_max)
    & (points[:, 2] >= args.support_z)
    & (points[:, 2] <= z_max)
)

points = points[xy_mask]

pitch = args.voxel_size
margin = 2.0 * pitch

grid_x_min = x_min - margin
grid_y_min = y_min - margin

grid_width = int(
    np.ceil(
        (x_max - x_min + 2.0 * margin)
        / pitch
    )
) + 1

grid_height = int(
    np.ceil(
        (y_max - y_min + 2.0 * margin)
        / pitch
    )
) + 1

z_centers = np.arange(
    args.support_z + pitch / 2.0,
    z_max,
    pitch,
)

areas = np.full(
    len(z_centers),
    np.nan,
    dtype=np.float64,
)

preview_mask = None
preview_index = len(z_centers) // 2

kernel = np.ones(
    (3, 3),
    dtype=np.uint8,
)

for index, z_center in enumerate(z_centers):
    slice_mask = (
        np.abs(
            points[:, 2] - z_center
        )
        <= args.slice_band
    )

    slice_points = points[slice_mask]

    if len(slice_points) < 10:
        continue

    cols = np.floor(
        (slice_points[:, 0] - grid_x_min)
        / pitch
    ).astype(np.int32)

    rows = np.floor(
        (slice_points[:, 1] - grid_y_min)
        / pitch
    ).astype(np.int32)

    inside = (
        (cols >= 0)
        & (cols < grid_width)
        & (rows >= 0)
        & (rows < grid_height)
    )

    cols = cols[inside]
    rows = rows[inside]

    boundary = np.zeros(
        (grid_height, grid_width),
        dtype=np.uint8,
    )

    boundary[rows, cols] = 255

    boundary = cv2.morphologyEx(
        boundary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=args.close_iterations,
    )

    contours, _ = cv2.findContours(
        boundary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    filled = np.zeros_like(boundary)

    for contour in contours:
        if (
            cv2.contourArea(contour)
            >= args.min_contour_area
        ):
            cv2.drawContours(
                filled,
                [contour],
                contourIdx=-1,
                color=255,
                thickness=cv2.FILLED,
            )

    occupied_pixels = np.count_nonzero(
        filled
    )

    if occupied_pixels > 0:
        areas[index] = (
            occupied_pixels
            * pitch
            * pitch
        )

    if index == preview_index:
        preview_mask = filled.copy()

valid_area = np.isfinite(areas)

if valid_area.sum() < 2:
    raise RuntimeError(
        "Not enough valid horizontal slices"
    )

slice_indices = np.arange(len(areas))

areas = np.interp(
    slice_indices,
    slice_indices[valid_area],
    areas[valid_area],
)

areas = median_filter(
    areas,
    size=3,
    mode="nearest",
)

volume = np.sum(areas) * pitch

ground_truth = args.ground_truth_volume

error_percent = (
    abs(volume - ground_truth)
    / ground_truth
    * 100.0
)

np.savetxt(
    output_dir / "area_profile.csv",
    np.column_stack(
        [z_centers, areas]
    ),
    delimiter=",",
    header="z_m,area_m2",
    comments="",
)

np.savez_compressed(
    output_dir / "slice_volume_data.npz",
    z=z_centers,
    area=areas,
    volume=np.float64(volume),
)

if preview_mask is not None:
    cv2.imwrite(
        str(output_dir / "middle_slice.png"),
        preview_mask,
    )

plt.figure(figsize=(7, 4))
plt.plot(
    z_centers,
    areas * 1e4,
)
plt.xlabel("Height z (m)")
plt.ylabel("Cross-section area (cm²)")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    output_dir / "area_profile.png",
    dpi=160,
)
plt.close()

metrics = (
    f"input_points: {len(points)}\n"
    f"slice_count: {len(z_centers)}\n"
    f"voxel_size_m: {pitch}\n"
    f"mean_area_m2: {areas.mean():.9f}\n"
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
