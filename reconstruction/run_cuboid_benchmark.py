import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECONSTRUCTION_DIR = PROJECT_ROOT / "reconstruction"

VIEWS = [0, 60, 120, 180, 240, 300]
EXPECTED_CAPTURE_COUNT = 72
EXPECTED_VIEW_COUNT = 6


def resolve_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


parser = argparse.ArgumentParser()

parser.add_argument(
    "--capture-root",
    default="outputs/random_cuboids",
)
parser.add_argument(
    "--reconstruction-root",
    default="outputs/random_cuboids_reconstruction",
)
parser.add_argument(
    "--fusion-root",
    default="outputs/random_cuboids_fusion",
)
parser.add_argument(
    "--tsdf-root",
    default="outputs/random_cuboids_tsdf",
)
parser.add_argument(
    "--calibration",
    default="configs/sim_calibration.npz",
)
parser.add_argument(
    "--volume-calibration",
    default="configs/tsdf_volume_calibration.npz",
)
parser.add_argument(
    "--start-index",
    type=int,
    default=0,
)
parser.add_argument(
    "--end-index",
    type=int,
    default=None,
)
parser.add_argument(
    "--with-fusion",
    action="store_true",
)
parser.add_argument(
    "--force",
    action="store_true",
)
parser.add_argument(
    "--fail-fast",
    action="store_true",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
)

args = parser.parse_args()

capture_root = resolve_path(args.capture_root)
reconstruction_root = resolve_path(args.reconstruction_root)
fusion_root = resolve_path(args.fusion_root)
tsdf_root = resolve_path(args.tsdf_root)
calibration_path = resolve_path(args.calibration)
volume_calibration_path = resolve_path(
    args.volume_calibration
)

manifest_path = capture_root / "cases.csv"
results_path = capture_root / "benchmark_results.csv"

if not manifest_path.exists():
    raise FileNotFoundError(manifest_path)

if not calibration_path.exists():
    raise FileNotFoundError(calibration_path)

if not volume_calibration_path.exists():
    raise FileNotFoundError(volume_calibration_path)

volume_calibration = np.load(volume_calibration_path)

if "surface_offset_m" not in volume_calibration:
    raise KeyError(
        "surface_offset_m is missing from "
        f"{volume_calibration_path}"
    )

surface_offset_m = float(
    volume_calibration["surface_offset_m"]
)

with manifest_path.open(
    "r",
    newline="",
    encoding="utf-8",
) as file:
    cases = list(csv.DictReader(file))

end_index = (
    len(cases)
    if args.end_index is None
    else min(args.end_index, len(cases))
)

selected_cases = cases[args.start_index:end_index]

if not selected_cases:
    raise ValueError("No cases selected")


result_fields = [
    "case_id",
    "size_x_m",
    "size_y_m",
    "size_z_m",
    "ground_truth_cm3",
    "capture_count",
    "reconstructed_view_count",
    "component_count",
    "main_face_count",
    "component_face_ratio",
    "watertight",
    "boundary_edge_count",
    "extent_x_m",
    "extent_y_m",
    "extent_z_m",
    "surface_area_m2",
    "raw_volume_cm3",
    "raw_error_percent",
    "corrected_volume_cm3",
    "corrected_error_percent",
    "surface_offset_m",
    "status",
    "error_message",
]


existing_results = {}

if results_path.exists():
    with results_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        for row in csv.DictReader(file):
            existing_results[row["case_id"]] = row


def save_results():
    ordered = sorted(
        existing_results.values(),
        key=lambda row: row["case_id"],
    )

    with results_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=result_fields,
        )
        writer.writeheader()

        for row in ordered:
            complete_row = {
                field: row.get(field, "")
                for field in result_fields
            }
            writer.writerow(complete_row)


def run_command(command):
    print("\n" + " ".join(str(item) for item in command))
    print(flush=True)

    if args.dry_run:
        return

    subprocess.run(
        [str(item) for item in command],
        check=True,
        cwd=PROJECT_ROOT,
    )


def load_main_component(mesh_path):
    loaded = trimesh.load_mesh(
        mesh_path,
        process=False,
    )

    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())

        if not geometries:
            raise RuntimeError(
                f"Empty scene: {mesh_path}"
            )

        mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = loaded

    components = list(
        mesh.split(only_watertight=False)
    )

    if not components:
        raise RuntimeError(
            f"No mesh components: {mesh_path}"
        )

    main = max(
        components,
        key=lambda component: len(component.faces),
    )

    main.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(main)
    trimesh.repair.fix_inversion(
        main,
        multibody=True,
    )

    return mesh, components, main


for case in selected_cases:
    case_id = case["case_id"]

    print("\n" + "=" * 72)
    print(f"Processing {case_id}")
    print("=" * 72, flush=True)

    result = {
        "case_id": case_id,
        "size_x_m": case["size_x_m"],
        "size_y_m": case["size_y_m"],
        "size_z_m": case["size_z_m"],
        "ground_truth_cm3": case[
            "ground_truth_cm3"
        ],
        "surface_offset_m": (
            f"{surface_offset_m:.12f}"
        ),
        "status": "running",
        "error_message": "",
    }

    try:
        case_capture = capture_root / case_id
        case_reconstruction = (
            reconstruction_root / case_id
        )
        case_fusion = fusion_root / case_id
        case_tsdf = tsdf_root / case_id

        capture_files = list(
            case_capture.glob(
                "view_*/freq_*/phase_*/0.hdf5"
            )
        )

        capture_count = len(capture_files)
        result["capture_count"] = str(
            capture_count
        )

        if capture_count != EXPECTED_CAPTURE_COUNT:
            raise RuntimeError(
                f"{case_id}: expected "
                f"{EXPECTED_CAPTURE_COUNT} captures, "
                f"found {capture_count}"
            )

        reconstruction_files = list(
            case_reconstruction.rglob(
                "reconstruction_data.npz"
            )
        )

        if (
            args.force
            or len(reconstruction_files)
            != EXPECTED_VIEW_COUNT
        ):
            run_command(
                [
                    sys.executable,
                    RECONSTRUCTION_DIR
                    / "batch_reconstruct_exr.py",
                    "--views",
                    *[str(view) for view in VIEWS],
                    "--input",
                    case_capture,
                    "--output",
                    case_reconstruction,
                    "--calibration",
                    calibration_path,
                ]
            )

        if args.dry_run:
            result["status"] = "dry_run"
            existing_results[case_id] = result
            save_results()
            continue

        reconstruction_files = list(
            case_reconstruction.rglob(
                "reconstruction_data.npz"
            )
        )

        result["reconstructed_view_count"] = str(
            len(reconstruction_files)
        )

        if (
            len(reconstruction_files)
            != EXPECTED_VIEW_COUNT
        ):
            raise RuntimeError(
                f"{case_id}: reconstruction produced "
                f"{len(reconstruction_files)} views"
            )

        if args.with_fusion:
            fused_path = (
                case_fusion / "fused_object.npz"
            )

            size_x = float(case["size_x_m"])
            size_y = float(case["size_y_m"])
            size_z = float(case["size_z_m"])

            object_radius = (
                0.5
                * np.hypot(size_x, size_y)
                + 0.02
            )

            if args.force or not fused_path.exists():
                run_command(
                    [
                        sys.executable,
                        RECONSTRUCTION_DIR
                        / "fuse_multiview_consistent.py",
                        "--views",
                        *[
                            str(view)
                            for view in VIEWS
                        ],
                        "--input",
                        case_reconstruction,
                        "--output",
                        case_fusion,
                        "--radius",
                        str(object_radius),
                        "--min-z",
                        "0.003",
                        "--max-z",
                        str(size_z + 0.02),
                        "--support-radius",
                        "0.003",
                        "--min-views",
                        "2",
                        "--voxel-size",
                        "0.0015",
                    ]
                )

        raw_mesh_path = (
            case_tsdf / "tsdf_mesh_raw.ply"
        )

        if args.force or not raw_mesh_path.exists():
            run_command(
                [
                    sys.executable,
                    RECONSTRUCTION_DIR
                    / "tsdf_fusion.py",
                    "--reconstruction-root",
                    case_reconstruction,
                    "--capture-root",
                    case_capture,
                    "--calibration",
                    calibration_path,
                    "--output",
                    case_tsdf,
                    "--voxel-size",
                    "0.001",
                    "--sdf-trunc",
                    "0.004",
                    "--fill-kernel",
                    "5",
                    "--fill-iterations",
                    "1",
                ]
            )

        if not raw_mesh_path.exists():
            raise FileNotFoundError(raw_mesh_path)

        full_mesh, components, main = (
            load_main_component(raw_mesh_path)
        )

        unique_edges, edge_counts = np.unique(
            main.edges_sorted,
            axis=0,
            return_counts=True,
        )

        boundary_edge_count = int(
            np.sum(edge_counts == 1)
        )

        raw_volume_m3 = abs(float(main.volume))
        surface_area_m2 = float(main.area)

        corrected_volume_m3 = (
            raw_volume_m3
            + surface_offset_m * surface_area_m2
        )

        truth_cm3 = float(
            case["ground_truth_cm3"]
        )
        raw_volume_cm3 = raw_volume_m3 * 1e6
        corrected_volume_cm3 = (
            corrected_volume_m3 * 1e6
        )

        raw_error_percent = (
            abs(raw_volume_cm3 - truth_cm3)
            / truth_cm3
            * 100.0
        )

        corrected_error_percent = (
            abs(corrected_volume_cm3 - truth_cm3)
            / truth_cm3
            * 100.0
        )

        component_face_ratio = (
            len(main.faces)
            / max(len(full_mesh.faces), 1)
        )

        extent = np.asarray(
            main.extents,
            dtype=np.float64,
        )

        result.update(
            {
                "component_count": str(
                    len(components)
                ),
                "main_face_count": str(
                    len(main.faces)
                ),
                "component_face_ratio": (
                    f"{component_face_ratio:.9f}"
                ),
                "watertight": str(
                    bool(main.is_watertight)
                ),
                "boundary_edge_count": str(
                    boundary_edge_count
                ),
                "extent_x_m": f"{extent[0]:.9f}",
                "extent_y_m": f"{extent[1]:.9f}",
                "extent_z_m": f"{extent[2]:.9f}",
                "surface_area_m2": (
                    f"{surface_area_m2:.9f}"
                ),
                "raw_volume_cm3": (
                    f"{raw_volume_cm3:.6f}"
                ),
                "raw_error_percent": (
                    f"{raw_error_percent:.6f}"
                ),
                "corrected_volume_cm3": (
                    f"{corrected_volume_cm3:.6f}"
                ),
                "corrected_error_percent": (
                    f"{corrected_error_percent:.6f}"
                ),
                "status": "success",
                "error_message": "",
            }
        )

        case_tsdf.mkdir(
            parents=True,
            exist_ok=True,
        )

        metrics_path = (
            case_tsdf / "benchmark_metrics.txt"
        )

        metrics_path.write_text(
            "\n".join(
                [
                    f"case_id: {case_id}",
                    (
                        "ground_truth_cm3: "
                        f"{truth_cm3:.6f}"
                    ),
                    (
                        "raw_volume_cm3: "
                        f"{raw_volume_cm3:.6f}"
                    ),
                    (
                        "raw_error_percent: "
                        f"{raw_error_percent:.6f}"
                    ),
                    (
                        "corrected_volume_cm3: "
                        f"{corrected_volume_cm3:.6f}"
                    ),
                    (
                        "corrected_error_percent: "
                        f"{corrected_error_percent:.6f}"
                    ),
                    (
                        "surface_area_m2: "
                        f"{surface_area_m2:.9f}"
                    ),
                    (
                        "surface_offset_m: "
                        f"{surface_offset_m:.12f}"
                    ),
                    (
                        "boundary_edge_count: "
                        f"{boundary_edge_count}"
                    ),
                    (
                        "watertight: "
                        f"{main.is_watertight}"
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            f"\n{case_id}: "
            f"truth={truth_cm3:.3f} cm3, "
            f"raw={raw_volume_cm3:.3f} cm3 "
            f"({raw_error_percent:.3f}%), "
            f"corrected={corrected_volume_cm3:.3f} "
            f"cm3 "
            f"({corrected_error_percent:.3f}%)"
        )

    except Exception as error:
        result["status"] = "failed"
        result["error_message"] = (
            f"{type(error).__name__}: {error}"
        )

        print(
            f"\nFAILED {case_id}: "
            f"{result['error_message']}",
            file=sys.stderr,
        )

        if args.fail_fast:
            existing_results[case_id] = result
            save_results()
            raise

    existing_results[case_id] = result
    save_results()


successful_errors = []

for row in existing_results.values():
    if row.get("status") != "success":
        continue

    value = row.get(
        "corrected_error_percent",
        "",
    )

    if value:
        successful_errors.append(float(value))

if successful_errors:
    errors = np.asarray(
        successful_errors,
        dtype=np.float64,
    )

    print("\nBenchmark summary")
    print("-----------------")
    print("successful_cases:", len(errors))
    print("mean_error_percent:", errors.mean())
    print("median_error_percent:", np.median(errors))
    print(
        "p95_error_percent:",
        np.percentile(errors, 95),
    )
    print("max_error_percent:", errors.max())
    print(
        "within_1_percent:",
        np.mean(errors <= 1.0),
    )
    print(
        "within_2_percent:",
        np.mean(errors <= 2.0),
    )
    print(
        "within_3_percent:",
        np.mean(errors <= 3.0),
    )

print("\nResults:", results_path)
