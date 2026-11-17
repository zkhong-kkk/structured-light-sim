import argparse
import os
from pathlib import Path
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0")
parser.add_argument(
    "--views",
    type=int,
    nargs="+",
    default=[0, 60, 120, 180, 240, 300],
)
parser.add_argument(
    "--output",
    default="outputs/multiview",
)
args = parser.parse_args()

phases = [0, 90, 180, 270]
frequencies = [1, 4, 12]

script_path = (
    Path(__file__).resolve().parent
    / "single_phase_capture_through_hole.py"
)
output_root = Path(args.output).resolve()

environment = os.environ.copy()
environment["CUDA_VISIBLE_DEVICES"] = args.gpu

total = len(args.views) * len(frequencies) * len(phases)
completed = 0

for view in args.views:
    for frequency in frequencies:
        for phase in phases:
            completed += 1

            output_dir = (
                output_root
                / f"view_{view:03d}"
                / f"freq_{frequency:02d}"
                / f"phase_{phase:03d}"
            )

            hdf5_path = output_dir / "0.hdf5"

            if hdf5_path.exists():
                print(
                    f"[{completed}/{total}] skip: {hdf5_path}"
                )
                continue

            command = [
                "blenderproc",
                "run",
                str(script_path),
                "--periods",
                str(frequency),
                "--phase-degree",
                str(phase),
                "--turntable-angle",
                str(view),
                "--output",
                str(output_dir),
            ]

            print(
                f"[{completed}/{total}] "
                f"view={view}, frequency={frequency}, phase={phase}"
            )

            subprocess.run(
                command,
                check=True,
                env=environment,
            )

print("All multiview captures completed.")
