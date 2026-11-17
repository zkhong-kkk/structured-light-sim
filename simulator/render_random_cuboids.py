import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--output-root", required=True)
parser.add_argument("--count", type=int, default=30)
parser.add_argument("--seed", type=int, default=20261116)
parser.add_argument("--start-index", type=int, default=0)
parser.add_argument("--end-index", type=int, default=None)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

output_root = Path(args.output_root).resolve()
output_root.mkdir(parents=True, exist_ok=True)

capture_script = (
    Path(__file__).resolve().parent
    / "single_phase_capture_cuboid.py"
)

blenderproc_command = shutil.which("blenderproc")
if blenderproc_command is None:
    raise RuntimeError("blenderproc command was not found")

if args.count <= 0:
    raise ValueError("--count must be positive")

rng = np.random.default_rng(args.seed)

# 单位均为米。固定种子保证每次运行生成同一批物体。
dimensions = rng.uniform(
    low=np.array([0.06, 0.06, 0.05]),
    high=np.array([0.15, 0.15, 0.16]),
    size=(args.count, 3),
)

# 固定到 0.1 mm，便于记录和复现实验。
dimensions = np.round(dimensions, 4)

manifest_path = output_root / "cases.csv"

with manifest_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "case_id",
            "size_x_m",
            "size_y_m",
            "size_z_m",
            "ground_truth_cm3",
        ],
    )
    writer.writeheader()

    for index, (size_x, size_y, size_z) in enumerate(dimensions):
        writer.writerow(
            {
                "case_id": f"cuboid_{index:03d}",
                "size_x_m": f"{size_x:.4f}",
                "size_y_m": f"{size_y:.4f}",
                "size_z_m": f"{size_z:.4f}",
                "ground_truth_cm3": (
                    f"{size_x * size_y * size_z * 1e6:.6f}"
                ),
            }
        )

end_index = (
    args.count
    if args.end_index is None
    else min(args.end_index, args.count)
)

if args.start_index < 0 or args.start_index >= end_index:
    raise ValueError("Invalid start/end index")

views = [0, 60, 120, 180, 240, 300]
frequencies = [1, 4, 12]
phases = [0, 90, 180, 270]

environment = os.environ.copy()

for case_index in range(args.start_index, end_index):
    size_x, size_y, size_z = dimensions[case_index]
    case_id = f"cuboid_{case_index:03d}"

    print(
        f"\n=== {case_id}: "
        f"{size_x:.4f} x {size_y:.4f} x {size_z:.4f} m ===",
        flush=True,
    )

    for view in views:
        for periods in frequencies:
            for phase in phases:
                capture_output = (
                    output_root
                    / case_id
                    / f"view_{view:03d}"
                    / f"freq_{periods:02d}"
                    / f"phase_{phase:03d}"
                )

                result_file = capture_output / "0.hdf5"

                if result_file.exists():
                    print(f"skip: {result_file}", flush=True)
                    continue

                command = [
                    blenderproc_command,
                    "run",
                    str(capture_script),
                    "--periods",
                    str(periods),
                    "--phase-degree",
                    str(phase),
                    "--turntable-angle",
                    str(view),
                    "--size-x",
                    str(size_x),
                    "--size-y",
                    str(size_y),
                    "--size-z",
                    str(size_z),
                    "--output",
                    str(capture_output),
                ]

                print(" ".join(command), flush=True)

                if not args.dry_run:
                    subprocess.run(
                        command,
                        check=True,
                        env=environment,
                    )

print(f"\nManifest: {manifest_path}")
print("Completed.")
