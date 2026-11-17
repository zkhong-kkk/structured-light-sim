import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "--spatial-threshold",
    type=float,
    default=2.0,
)
parser.add_argument("--input", default="outputs/multifreq")
parser.add_argument("--output", default="outputs/phase_unwrapped")
parser.add_argument("--threshold", type=float, default=5.0)
parser.add_argument("--max-error", type=float, default=1.0)
parser.add_argument("--projector-width", type=int, default=640)
args = parser.parse_args()

input_dir = Path(args.input)
output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

TWO_PI = 2.0 * np.pi


def load_gray(freq, phase):
    path = (
        input_dir
        / f"freq_{freq:02d}"
        / f"phase_{phase:03d}"
        / "0.hdf5"
    )

    with h5py.File(path, "r") as f:
        rgb = f["colors"][:].astype(np.float32)
        depth = f["depth"][:].astype(np.float32)

    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )

    return gray, depth


def decode_frequency(freq):
    i0, depth = load_gray(freq, 0)
    i90, _ = load_gray(freq, 90)
    i180, _ = load_gray(freq, 180)
    i270, _ = load_gray(freq, 270)

    cos_component = i0 - i180
    sin_component = i270 - i90

    wrapped = np.arctan2(
        sin_component,
        cos_component,
    ).astype(np.float32)

    modulation = (
        0.5
        * np.sqrt(
            cos_component**2 + sin_component**2
        )
    ).astype(np.float32)

    average = (
        (i0 + i90 + i180 + i270) / 4.0
    ).astype(np.float32)

    valid = (
        np.isfinite(depth)
        & (depth < 1e9)
        & (modulation >= args.threshold)
        & (average >= args.threshold)
    )

    # 转换到 [0, 2π)
    positive_phase = np.mod(wrapped, TWO_PI)

    return positive_phase, modulation, valid, depth


def wrap_to_pi(value):
    return (value + np.pi) % TWO_PI - np.pi


phi1, modulation1, valid1, depth = decode_frequency(1)
phi4, modulation4, valid4, _ = decode_frequency(4)
phi12, modulation12, valid12, _ = decode_frequency(12)

# 频率1只有一个周期，可作为粗绝对相位
absolute1 = phi1

# 1周期展开4周期
order4 = np.rint(
    (4.0 * absolute1 - phi4) / TWO_PI
)
absolute4 = phi4 + TWO_PI * order4

# 4周期展开12周期
order12 = np.rint(
    (3.0 * absolute4 - phi12) / TWO_PI
)
absolute12 = phi12 + TWO_PI * order12

error4 = np.abs(
    wrap_to_pi(absolute4 - 4.0 * absolute1)
)

error12 = np.abs(
    wrap_to_pi(absolute12 - 3.0 * absolute4)
)

projector_u_normalized = absolute12 / (TWO_PI * 12.0)
projector_u = (
    projector_u_normalized * args.projector_width
).astype(np.float32)

valid_mask = (
    valid1
    & valid4
    & valid12
    & (error4 <= args.max_error)
    & (error12 <= args.max_error)
    & (projector_u_normalized >= 0.0)
    & (projector_u_normalized < 1.0)
)
# 投影仪坐标在局部区域应该连续。
# 使用5×5中值滤波删除错误条纹级次产生的跳变点。
base_valid = valid_mask.copy()
u_for_filter = projector_u.astype(np.float32).copy()
u_for_filter[~base_valid] = 0.0

median_u = cv2.medianBlur(
    u_for_filter,
    5,
)

spatial_error = np.abs(
    projector_u - median_u
)

interior_mask = cv2.erode(
    base_valid.astype(np.uint8),
    np.ones((5, 5), dtype=np.uint8),
    iterations=1,
).astype(bool)

valid_mask = (
    interior_mask
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
    depth=depth,
    modulation_1=modulation1,
    modulation_4=modulation4,
    modulation_12=modulation12,
    error_4=error4,
    error_12=error12,
    spatial_error=spatial_error,
)

u_vis = np.clip(
    projector_u_normalized * 255.0,
    0,
    255,
).astype(np.uint8)
u_vis[~valid_mask] = 0

u_color = cv2.applyColorMap(
    u_vis,
    cv2.COLORMAP_TURBO,
)
u_color[~valid_mask] = 0

absolute_vis = np.clip(
    absolute12 / (TWO_PI * 12.0) * 255.0,
    0,
    255,
).astype(np.uint8)
absolute_vis[~valid_mask] = 0

absolute_color = cv2.applyColorMap(
    absolute_vis,
    cv2.COLORMAP_TURBO,
)
absolute_color[~valid_mask] = 0

cv2.imwrite(
    str(output_dir / "projector_u.png"),
    u_vis,
)
cv2.imwrite(
    str(output_dir / "projector_u_color.png"),
    u_color,
)
cv2.imwrite(
    str(output_dir / "absolute_phase_color.png"),
    absolute_color,
)
cv2.imwrite(
    str(output_dir / "valid_mask.png"),
    valid_mask.astype(np.uint8) * 255,
)

if np.any(valid_mask):
    print(
        "absolute phase range:",
        absolute12[valid_mask].min(),
        absolute12[valid_mask].max(),
    )
    print(
        "projector u range:",
        projector_u[valid_mask].min(),
        projector_u[valid_mask].max(),
    )
    print(
        "mean unwrap error:",
        error4[valid_mask].mean(),
        error12[valid_mask].mean(),
    )

print("valid pixel ratio:", valid_mask.mean())
print("saved to:", output_dir.resolve())
