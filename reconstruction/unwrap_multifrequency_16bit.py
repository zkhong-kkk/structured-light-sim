import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument(
    "--depth-cutoff",
    type=float,
    default=2.0,
)
parser.add_argument("--min-average", type=float, default=1000.0)
parser.add_argument("--min-mod-ratio", type=float, default=0.05)
parser.add_argument("--max-error", type=float, default=0.5)
parser.add_argument("--spatial-threshold", type=float, default=3.0)
parser.add_argument("--saturation-level", type=float, default=65535.0)
parser.add_argument("--projector-width", type=int, default=640)
args = parser.parse_args()

input_dir = Path(args.input)
output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

TWO_PI = 2.0 * np.pi


def load_capture(frequency, phase):
    path = (
        input_dir
        / f"freq_{frequency:02d}"
        / f"phase_{phase:03d}"
        / "0.hdf5"
    )

    with h5py.File(path, "r") as f:
        raw = f["colors"][:]
        depth = f["depth"][:].astype(np.float32)

    rgb = raw.astype(np.float32)

    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )

    peak = rgb.max(axis=2)

    return gray, peak, depth


def decode_frequency(frequency):
    phases = [0, 90, 180, 270]

    captures = [
        load_capture(frequency, phase)
        for phase in phases
    ]

    images = np.stack(
        [capture[0] for capture in captures],
        axis=0,
    )

    peaks = np.stack(
        [capture[1] for capture in captures],
        axis=0,
    )

    depth = captures[0][2]

    i0, i90, i180, i270 = images

    cos_component = i0 - i180
    sin_component = i270 - i90

    wrapped = np.arctan2(
        sin_component,
        cos_component,
    ).astype(np.float32)

    modulation = (
        0.5
        * np.hypot(
            cos_component,
            sin_component,
        )
    ).astype(np.float32)

    average = images.mean(axis=0).astype(np.float32)

    modulation_ratio = (
        modulation / (average + 1e-8)
    ).astype(np.float32)

    saturated = peaks.max(axis=0) >= args.saturation_level

    valid = (
        np.isfinite(depth)
        & (depth < args.depth_cutoff)
        & (average >= args.min_average)
        & (modulation_ratio >= args.min_mod_ratio)
        & (~saturated)
    )

    positive_phase = np.mod(
        wrapped,
        TWO_PI,
    ).astype(np.float32)

    print(
        f"frequency {frequency}: "
        f"valid_ratio={valid.mean():.6f}, "
        f"ratio_median="
        f"{np.median(modulation_ratio[valid]):.6f}"
    )

    return {
        "phase": positive_phase,
        "modulation": modulation,
        "ratio": modulation_ratio,
        "valid": valid,
        "depth": depth,
    }


def wrap_to_pi(value):
    return (
        value + np.pi
    ) % TWO_PI - np.pi


frequency1 = decode_frequency(1)
frequency4 = decode_frequency(4)
frequency12 = decode_frequency(12)

phi1 = frequency1["phase"]
phi4 = frequency4["phase"]
phi12 = frequency12["phase"]

absolute1 = phi1

order4 = np.rint(
    (4.0 * absolute1 - phi4) / TWO_PI
)
absolute4 = phi4 + TWO_PI * order4

order12 = np.rint(
    (3.0 * absolute4 - phi12) / TWO_PI
)
absolute12 = phi12 + TWO_PI * order12

error4 = np.abs(
    wrap_to_pi(
        absolute4 - 4.0 * absolute1
    )
)

error12 = np.abs(
    wrap_to_pi(
        absolute12 - 3.0 * absolute4
    )
)

projector_u_normalized = (
    absolute12 / (TWO_PI * 12.0)
)

projector_u = (
    projector_u_normalized
    * args.projector_width
).astype(np.float32)

base_valid = (
    frequency1["valid"]
    & frequency4["valid"]
    & frequency12["valid"]
    & (error4 <= args.max_error)
    & (error12 <= args.max_error)
    & (projector_u_normalized >= 0.0)
    & (projector_u_normalized < 1.0)
)

# 仅为计算局部中值填充无效位置，不修改最终深度。
_, nearest = distance_transform_edt(
    ~base_valid,
    return_indices=True,
)

u_for_filter = projector_u.copy()
u_for_filter[~base_valid] = projector_u[
    nearest[0][~base_valid],
    nearest[1][~base_valid],
]

median_u = cv2.medianBlur(
    u_for_filter.astype(np.float32),
    3,
)

spatial_error = np.abs(
    projector_u - median_u
)

valid_mask = (
    base_valid
    & (spatial_error <= args.spatial_threshold)
)

projector_u_output = np.full(
    projector_u.shape,
    np.nan,
    dtype=np.float32,
)
projector_u_output[valid_mask] = projector_u[valid_mask]

absolute_phase_output = np.full(
    absolute12.shape,
    np.nan,
    dtype=np.float32,
)
absolute_phase_output[valid_mask] = absolute12[valid_mask]

np.savez_compressed(
    output_dir / "unwrapped_phase_data.npz",
    absolute_phase=absolute_phase_output,
    projector_u=projector_u_output,
    valid_mask=valid_mask,
    depth=frequency1["depth"],
    modulation_ratio_1=frequency1["ratio"],
    modulation_ratio_4=frequency4["ratio"],
    modulation_ratio_12=frequency12["ratio"],
    error_4=error4,
    error_12=error12,
    spatial_error=spatial_error,
)

visual = np.clip(
    projector_u_normalized * 255.0,
    0,
    255,
).astype(np.uint8)

visual[~valid_mask] = 0

color_visual = cv2.applyColorMap(
    visual,
    cv2.COLORMAP_TURBO,
)
color_visual[~valid_mask] = 0

cv2.imwrite(
    str(output_dir / "projector_u_color.png"),
    color_visual,
)
cv2.imwrite(
    str(output_dir / "valid_mask.png"),
    valid_mask.astype(np.uint8) * 255,
)

print("base valid ratio:", base_valid.mean())
print("final valid ratio:", valid_mask.mean())

if np.any(valid_mask):
    print(
        "projector u range:",
        projector_u[valid_mask].min(),
        projector_u[valid_mask].max(),
    )

print("saved to:", output_dir.resolve())
