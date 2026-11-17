import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="outputs/phase_shift")
parser.add_argument("--output", default="outputs/phase_decode")
parser.add_argument("--threshold", type=float, default=5.0)
args = parser.parse_args()

input_dir = Path(args.input)
output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)


def load_capture(phase_degree):
    path = input_dir / f"phase_{phase_degree:03d}" / "0.hdf5"

    with h5py.File(path, "r") as f:
        rgb = f["colors"][:].astype(np.float32)
        depth = f["depth"][:].astype(np.float32)

    # 输入数据是RGB顺序
    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )
    return gray, depth


i0, depth = load_capture(0)
i90, _ = load_capture(90)
i180, _ = load_capture(180)
i270, _ = load_capture(270)

cos_component = i0 - i180
sin_component = i270 - i90

# 范围为 [-pi, pi]
wrapped_phase = np.arctan2(
    sin_component,
    cos_component,
).astype(np.float32)

# 正弦信号幅值
modulation = (
    0.5
    * np.sqrt(
        cos_component**2 + sin_component**2
    )
).astype(np.float32)

average_intensity = (
    (i0 + i90 + i180 + i270) / 4.0
).astype(np.float32)

geometry_mask = np.isfinite(depth) & (depth < 1e9)
valid_mask = (
    geometry_mask
    & (modulation >= args.threshold)
    & (average_intensity >= args.threshold)
)

# 保存原始数值，后续相位展开和三角测量使用
np.savez_compressed(
    output_dir / "phase_data.npz",
    wrapped_phase=wrapped_phase,
    modulation=modulation,
    average_intensity=average_intensity,
    valid_mask=valid_mask,
    depth=depth,
)

phase_gray = np.clip(
    (wrapped_phase + np.pi) / (2.0 * np.pi) * 255.0,
    0,
    255,
).astype(np.uint8)

phase_color = cv2.applyColorMap(
    phase_gray,
    cv2.COLORMAP_TURBO,
)
phase_color[~valid_mask] = 0

modulation_vis = np.zeros_like(phase_gray)
if np.any(valid_mask):
    upper = np.percentile(modulation[valid_mask], 99)
    modulation_vis = np.clip(
        modulation / max(upper, 1e-6) * 255.0,
        0,
        255,
    ).astype(np.uint8)

modulation_vis[~valid_mask] = 0

cv2.imwrite(
    str(output_dir / "wrapped_phase_color.png"),
    phase_color,
)
cv2.imwrite(
    str(output_dir / "modulation.png"),
    modulation_vis,
)
cv2.imwrite(
    str(output_dir / "valid_mask.png"),
    valid_mask.astype(np.uint8) * 255,
)

print("phase range:", wrapped_phase[valid_mask].min(),
      wrapped_phase[valid_mask].max())
print("modulation range:", modulation[valid_mask].min(),
      modulation[valid_mask].max())
print("valid pixel ratio:", valid_mask.mean())
print("saved to:", output_dir.resolve())
