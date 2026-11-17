"""统一计算主体网格的原始体积、表面积校正体积和误差。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


parser = argparse.ArgumentParser()
parser.add_argument("--mesh", required=True)
parser.add_argument("--truth-cm3", type=float, required=True)
parser.add_argument(
    "--calibration",
    default="configs/tsdf_volume_calibration.npz",
)
parser.add_argument("--output", default=None)
args = parser.parse_args()

loaded = trimesh.load_mesh(args.mesh, process=False)
components = list(loaded.split(only_watertight=False))
mesh = max(components, key=lambda component: len(component.faces))
trimesh.repair.fix_winding(mesh)
trimesh.repair.fix_inversion(mesh, multibody=True)

calibration = np.load(args.calibration)
surface_offset = float(calibration["surface_offset_m"])
raw_cm3 = abs(float(mesh.volume)) * 1e6
corrected_cm3 = (
    abs(float(mesh.volume)) + surface_offset * float(mesh.area)
) * 1e6

_, edge_counts = np.unique(
    mesh.edges_sorted,
    axis=0,
    return_counts=True,
)

result = {
    "component_count": len(components),
    "main_face_count": len(mesh.faces),
    "main_component_face_ratio": len(mesh.faces) / len(loaded.faces),
    "watertight": bool(mesh.is_watertight),
    "boundary_edge_count": int(np.sum(edge_counts == 1)),
    "extent_m": [float(value) for value in mesh.extents],
    "surface_area_m2": float(mesh.area),
    "surface_offset_m": surface_offset,
    "truth_cm3": args.truth_cm3,
    "raw_volume_cm3": raw_cm3,
    "raw_error_percent": abs(raw_cm3 - args.truth_cm3)
    / args.truth_cm3
    * 100.0,
    "corrected_volume_cm3": corrected_cm3,
    "corrected_error_percent": abs(corrected_cm3 - args.truth_cm3)
    / args.truth_cm3
    * 100.0,
}

text = json.dumps(result, ensure_ascii=False, indent=2)
print(text)
if args.output:
    Path(args.output).write_text(text + "\n", encoding="utf-8")

