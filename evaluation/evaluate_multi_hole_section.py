"""Evaluate closed hole loops on an arbitrary axis-aligned mesh section."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


parser = argparse.ArgumentParser()
parser.add_argument("--mesh", required=True)
parser.add_argument("--axis", choices=["x", "y", "z"], required=True)
parser.add_argument("--position", type=float, required=True)
parser.add_argument("--expected-a", type=float, required=True)
parser.add_argument("--expected-b", type=float, required=True)
parser.add_argument("--min-extent", type=float, default=0.01)
parser.add_argument("--max-extent", type=float, default=0.06)
parser.add_argument("--output", default=None)
args = parser.parse_args()

axis_index = {"x": 0, "y": 1, "z": 2}[args.axis]
section_indices = {
    "x": [1, 2],
    "y": [0, 2],
    "z": [0, 1],
}[args.axis]

origin = np.zeros(3, dtype=np.float64)
normal = np.zeros(3, dtype=np.float64)
origin[axis_index] = args.position
normal[axis_index] = 1.0

loaded = trimesh.load_mesh(args.mesh, process=False)
components = list(loaded.split(only_watertight=False))
mesh = max(components, key=lambda component: len(component.faces))
section = mesh.section(plane_origin=origin, plane_normal=normal)
curves = [] if section is None else list(section.discrete)

expected = np.array(
    [args.expected_a, args.expected_b],
    dtype=np.float64,
)
curve_results = []
for curve in curves:
    coordinates = curve[:, section_indices]
    extent = np.ptp(coordinates, axis=0)
    center = coordinates.mean(axis=0)
    gap = float(np.linalg.norm(curve[0] - curve[-1]))
    is_candidate = bool(
        np.all(extent >= args.min_extent)
        and np.all(extent <= args.max_extent)
    )
    curve_results.append(
        {
            "extent_m": [float(value) for value in extent],
            "center_m": [float(value) for value in center],
            "closed_gap_m": gap,
            "is_candidate": is_candidate,
            "size_error_percent": [
                float(value)
                for value in np.abs(extent - expected) / expected * 100.0
            ],
        }
    )

curve_results.sort(
    key=lambda item: item["center_m"][0],
)
candidates = [
    curve for curve in curve_results if curve["is_candidate"]
]

result = {
    "mesh": str(Path(args.mesh)),
    "axis": args.axis,
    "position_m": args.position,
    "expected_extent_m": [float(value) for value in expected],
    "component_count": len(components),
    "main_face_count": len(mesh.faces),
    "main_extent_m": [float(value) for value in mesh.extents],
    "watertight": bool(mesh.is_watertight),
    "section_entity_count": 0 if section is None else len(section.entities),
    "closed_curve_count": len(curves),
    "candidate_count": len(candidates),
    "candidates": candidates,
    "all_curves": curve_results,
}

text = json.dumps(result, ensure_ascii=False, indent=2)
print(text)
if args.output:
    Path(args.output).write_text(text + "\n", encoding="utf-8")
