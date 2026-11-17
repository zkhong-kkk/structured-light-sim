import argparse
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, default=15.0)
parser.add_argument("--max-error", type=float, default=0.5)
parser.add_argument("--spatial-threshold", type=float, default=2.0)
parser.add_argument(
    "--views",
    type=int,
    nargs="+",
    default=[0, 60, 120, 180, 240, 300],
)
parser.add_argument(
    "--input",
    default="outputs/multiview",
)
parser.add_argument(
    "--output",
    default="outputs/multiview_reconstruction",
)
parser.add_argument(
    "--calibration",
    default="configs/sim_calibration.npz",
)
args = parser.parse_args()

project_root = Path(__file__).resolve().parents[1]
input_root = Path(args.input).resolve()
output_root = Path(args.output).resolve()
calibration = Path(args.calibration).resolve()

unwrap_script = project_root / "reconstruction" / "unwrap_multifrequency.py"
triangulate_script = project_root / "reconstruction" / "triangulate.py"

for index, view in enumerate(args.views, start=1):
    print(f"\n[{index}/{len(args.views)}] Processing view {view} degrees")

    view_input = input_root / f"view_{view:03d}"
    view_output = output_root / f"view_{view:03d}"

    phase_output = view_output / "phase"
    reconstruction_output = view_output / "reconstruction"

    phase_file = phase_output / "unwrapped_phase_data.npz"
    reconstruction_file = reconstruction_output / "reconstruction_data.npz"

    if not phase_file.exists():
        subprocess.run(
            [
                sys.executable,
                str(unwrap_script),
                "--input",
                str(view_input),
                "--output",
                str(phase_output),
                "--threshold",
                str(args.threshold),
                "--max-error",
                str(args.max_error),
                "--spatial-threshold",
                str(args.spatial_threshold),
            ],
            check=True,
        )
    else:
        print("Phase already decoded, skipping.")

    if not reconstruction_file.exists():
        subprocess.run(
            [
                sys.executable,
                str(triangulate_script),
                "--phase",
                str(phase_file),
                "--calibration",
                str(calibration),
                "--capture",
                str(
                    view_input
                    / "freq_12"
                    / "phase_000"
                    / "0.hdf5"
                ),
                "--output",
                str(reconstruction_output),
                "--flip-projector-u",
            ],
            check=True,
        )
    else:
        print("Point cloud already reconstructed, skipping.")

print("\nAll views reconstructed.")
