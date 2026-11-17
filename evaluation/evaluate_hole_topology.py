"""评估网格指定截面中的闭合孔曲线。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


parser = argparse.ArgumentParser()
parser.add_argument("--mesh", required=True)
parser.add_argument("--output", default=None)
parser.add_argument("--plane-x", type=float, default=0.0)
parser.add_argument("--plane-z", type=float, default=0.05)
parser.add_argument("--expected-y", type=float, default=0.04)
parser.add_argument("--expected-z", type=float, default=0.04)
parser.add_argument("--tolerance", type=float, default=0.012)
args = parser.parse_args()

loaded = trimesh.load_mesh(args.mesh, process=False)
components = list(loaded.split(only_watertight=False))
mesh = max(components, key=lambda component: len(component.faces))

section = mesh.section(
    plane_origin=[args.plane_x, 0.0, args.plane_z],
    plane_normal=[1.0, 0.0, 0.0],
)
curves = [] if section is None else list(section.discrete)
curves = sorted(
    curves,
    key=lambda curve: np.prod(np.ptp(curve[:, [1, 2]], axis=0)),
    reverse=True,
)

curve_results = []
for curve in curves:
    extent = np.ptp(curve[:, [1, 2]], axis=0)
    center = curve[:, [1, 2]].mean(axis=0)
    gap = float(np.linalg.norm(curve[0] - curve[-1]))
    expected = np.array([args.expected_y, args.expected_z])
    is_target_hole = bool(
        np.all(np.abs(extent - expected) <= args.tolerance)
    )
    curve_results.append(
        {
            "extent_y_m": float(extent[0]),
            "extent_z_m": float(extent[1]),
            "center_y_m": float(center[0]),
            "center_z_m": float(center[1]),
            "closed_gap_m": gap,
            "is_target_hole": is_target_hole,
        }
    )

result = {
    "mesh": str(Path(args.mesh)),
    "component_count": len(components),
    "main_face_count": len(mesh.faces),
    "watertight": bool(mesh.is_watertight),
    "section_is_none": section is None,
    "section_entity_count": 0 if section is None else len(section.entities),
    "closed_curve_count": len(curves),
    "target_hole_found": any(
        curve["is_target_hole"] for curve in curve_results
    ),
    "curves": curve_results,
}

text = json.dumps(result, ensure_ascii=False, indent=2)
print(text)
if args.output:
    Path(args.output).write_text(text + "\n", encoding="utf-8")

