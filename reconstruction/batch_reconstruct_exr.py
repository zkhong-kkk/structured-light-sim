import argparse
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser()
parser.add_argument(
    "--views",
    type=int,
    nargs="+",
    default=[0, 60, 120, 180, 240, 300],
)
parser.add_argument(
    "--input",
    default="outputs/multiview_16bit",
)
parser.add_argument(
    "--output",
    default="outputs/multiview_reconstruction_16bit",
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

unwrap_script = (
    project_root
    / "reconstruction"
    / "unwrap_multifrequency_16bit.py"
)

triangulate_script = (
    project_root
    / "reconstruction"
    / "triangulate.py"
)

for index, view in enumerate(args.views, start=1):
    print(f"\n[{index}/{len(args.views)}] view={view}")

    view_input = input_root / f"view_{view:03d}"
    view_output = output_root / f"view_{view:03d}"

    phase_output = view_output / "phase"
    reconstruction_output = view_output / "reconstruction"

    phase_file = phase_output / "unwrapped_phase_data.npz"
    point_file = reconstruction_output / "reconstruction_data.npz"

    if not phase_file.exists():
        subprocess.run(
            [
                sys.executable,
                str(unwrap_script),
                "--input",
                str(view_input),
                "--output",
                str(phase_output),
                "--min-average",
                "0.001",
                "--min-mod-ratio",
                "0.05",
                "--saturation-level",
                "1000000000",                
                "--max-error",
                "0.5",
                "--spatial-threshold",
                "3.0",
                "--depth-cutoff",
                "2.0",
            ],
            check=True,
        )

    if not point_file.exists():
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
                "--depth-cutoff",
                "2.0",
                "--local-range-threshold",
                "0.01",
                "--local-kernel",
                "5",
                "--roi-radius",
                "0.14",
                "--roi-min-z",
                "0.003",
                "--roi-max-z",
                "0.25",
            ],
            check=True,
        )

print("\nAll 16-bit views reconstructed.")
