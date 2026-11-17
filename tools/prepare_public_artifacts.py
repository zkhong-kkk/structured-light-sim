"""从本地实验目录生成可公开的脱敏日志和结果副本。"""

from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_LOG_DIR = ROOT / "artifacts" / "logs"
PUBLIC_RESULT_DIR = ROOT / "artifacts" / "results"

LOG_NAMES = [
    "hole_camera2_pipeline.log",
    "upper_lower_hole_pipeline.log",
    "dual_tsdf_sweep.log",
    "direct_poisson.log",
    "direct_poisson_tight.log",
    "tsdf_dual.log",
    "tsdf_fillfix.log",
    "upper_lower_roi.log",
    "fourcam_reconstruction.log",
    "fourcam_poisson_tight.log",
    "complex_fourcam_reconstruction.log",
    "complex_poisson_tight.log",
    "complex_poisson_balanced.log",
    "curved_fourcam_reconstruction.log",
    "curved_steep_reconstruction.log",
]

RESULT_COPIES = {
    "outputs/random_cuboids/benchmark_results.csv": (
        "random_cuboids_benchmark.csv"
    ),
    "outputs/tsdf_horizontal_hole_upper_lower_12views/"
    "upper_lower_metrics.txt": "upper_lower_metrics.txt",
    "outputs/poisson_horizontal_hole_direct_dual_tight/"
    "poisson_metrics.txt": "poisson_tight_metrics.txt",
    "outputs/tsdf_horizontal_hole_camera2_12views/"
    "camera2_volume_metrics.txt": "camera2_volume_metrics.txt",
    "outputs/tsdf_horizontal_hole_camera2_12views/"
    "camera2_topology_metrics.txt": "camera2_topology_metrics.txt",
    "outputs/tsdf_horizontal_hole_fourcam_12views/"
    "tsdf_metrics.txt": "fourcam_horizontal_tsdf_metrics.txt",
    "outputs/poisson_horizontal_hole_fourcam_tight/"
    "poisson_metrics.txt": "fourcam_horizontal_poisson_metrics.txt",
    "outputs/tsdf_complex_fourcam_12views/"
    "tsdf_metrics.txt": "fourcam_complex_tsdf_metrics.txt",
    "outputs/poisson_complex_fourcam_direct/"
    "poisson_metrics.txt": "fourcam_complex_poisson_metrics.txt",
    "outputs/tsdf_curved_fourcam_12views/"
    "tsdf_metrics.txt": "fourcam_curved_tsdf_metrics.txt",
    "outputs/poisson_curved_fourcam_direct/"
    "poisson_metrics.txt": "fourcam_curved_poisson_metrics.txt",
    "outputs/tsdf_curved_steep_fourcam_12views/"
    "tsdf_metrics.txt": "fourcam_curved_steep_tsdf_metrics.txt",
    "outputs/poisson_curved_steep_fourcam_direct/"
    "poisson_metrics.txt": "fourcam_curved_steep_poisson_metrics.txt",
    "outputs/fourcam_render_status.txt": "fourcam_render_status.txt",
    "outputs/fourcam_reconstruction_status.txt": (
        "fourcam_reconstruction_status.txt"
    ),
    "outputs/complex_fourcam_render_status.txt": (
        "fourcam_complex_render_status.txt"
    ),
    "outputs/complex_fourcam_reconstruction_status.txt": (
        "fourcam_complex_reconstruction_status.txt"
    ),
    "outputs/curved_fourcam_render_status.txt": (
        "fourcam_curved_render_status.txt"
    ),
    "outputs/curved_fourcam_reconstruction_status.txt": (
        "fourcam_curved_reconstruction_status.txt"
    ),
    "outputs/curved_steep_render_status.txt": (
        "fourcam_curved_steep_render_status.txt"
    ),
    "outputs/curved_steep_reconstruction_status.txt": (
        "fourcam_curved_steep_reconstruction_status.txt"
    ),
}

REDACTIONS = [
    (
        re.compile(
            r"/mnt/disk2/[A-Za-z0-9._-]+/projects/structured-light-sim"
        ),
        "${PROJECT_ROOT}",
    ),
    (
        re.compile(r"/mnt/disk2/[A-Za-z0-9._-]+"),
        "${USER_HOME}",
    ),
    (
        re.compile(r"/home_local/[A-Za-z0-9._-]+"),
        "${LEGACY_HOME}",
    ),
]


def percentile(values: list[float], probability: float) -> float:
    """NumPy 风格线性分位数，避免生成报告时强制依赖 NumPy。"""

    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * fraction


def sanitize_logs() -> None:
    PUBLIC_LOG_DIR.mkdir(parents=True, exist_ok=True)
    for name in LOG_NAMES:
        source = ROOT / "logs" / name
        if not source.exists():
            continue
        text = source.read_text(encoding="utf-8", errors="replace")
        for pattern, replacement in REDACTIONS:
            text = pattern.sub(replacement, text)
        (PUBLIC_LOG_DIR / name).write_text(text, encoding="utf-8")


def copy_results() -> None:
    PUBLIC_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for relative_source, public_name in RESULT_COPIES.items():
        source = ROOT / relative_source
        if source.exists():
            shutil.copyfile(source, PUBLIC_RESULT_DIR / public_name)


def summarize_cuboids() -> None:
    csv_path = PUBLIC_RESULT_DIR / "random_cuboids_benchmark.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["status"] == "success"
        ]

    errors = [float(row["corrected_error_percent"]) for row in rows]
    summary = {
        "successful_cases": len(errors),
        "mean_error_percent": mean(errors),
        "median_error_percent": median(errors),
        "p95_error_percent": percentile(errors, 0.95),
        "max_error_percent": max(errors),
        "within_1_percent": sum(value <= 1.0 for value in errors)
        / len(errors),
        "within_2_percent": sum(value <= 2.0 for value in errors)
        / len(errors),
        "within_3_percent": sum(value <= 3.0 for value in errors)
        / len(errors),
    }
    (PUBLIC_RESULT_DIR / "random_cuboids_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def summarize_sweep() -> None:
    rows = []
    for path in sorted(
        (ROOT / "outputs" / "dual_tsdf_sweep").glob(
            "*/sweep_metrics.txt"
        )
    ):
        values = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
        rows.append(
            {
                "name": values.get("name", path.parent.name),
                "boundary_edges": values.get("boundary_edges", ""),
                "corrected_error_percent": values.get(
                    "corrected_error_percent", ""
                ),
                "closed_curve_count": values.get(
                    "closed_curve_count", ""
                ),
                "inner_loop_found": values.get("inner_loop_found", ""),
                "topology_pass": values.get("topology_pass", ""),
            }
        )

    if not rows:
        return

    output = PUBLIC_RESULT_DIR / "dual_tsdf_sweep.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "boundary_edges",
                "corrected_error_percent",
                "closed_curve_count",
                "inner_loop_found",
                "topology_pass",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_four_camera_experiments() -> None:
    """Write the audited cross-object metrics used by the report.

    Mesh-section measurements are produced by the evaluation scripts rather
    than the raw TSDF/Poisson metric files, so they are recorded here next to
    the volume values and linked to the public raw logs.
    """

    rows = [
        {
            "case_id": "horizontal_hole",
            "display_name": "水平单方孔",
            "truth_volume_cm3": 1008.0,
            "raw_volume_cm3": 968.4254428003654,
            "raw_error_percent": 3.9260473412335886,
            "corrected_volume_cm3": 1015.3763691176924,
            "corrected_error_percent": 0.7317826505647256,
            "boundary_edges": 1376,
            "expected_holes": 1,
            "recovered_holes": 1,
            "hole_measurements_mm": "40.326x40.506",
            "topology_status": "pass",
            "notes": "Tight Poisson; mean dimension error 1.04%",
        },
        {
            "case_id": "double_hole_notch",
            "display_name": "双方孔顶部凹槽",
            "truth_volume_cm3": 1124.5,
            "raw_volume_cm3": 1081.8799343455532,
            "raw_error_percent": 3.7901347847440423,
            "corrected_volume_cm3": 1141.1600863903216,
            "corrected_error_percent": 1.4815550369338877,
            "boundary_edges": 4862,
            "expected_holes": 2,
            "recovered_holes": 2,
            "hole_measurements_mm": "25.898x33.653;22.719x33.843",
            "topology_status": "pass_with_bias",
            "notes": "Direct Poisson; mean dimension error 4.97%",
        },
        {
            "case_id": "stepped_cylinder_original",
            "display_name": "阶梯圆柱双深孔（原布局）",
            "truth_volume_cm3": 858.1271496927825,
            "raw_volume_cm3": 930.8886472448966,
            "raw_error_percent": 8.479104475154218,
            "corrected_volume_cm3": 963.8799616548882,
            "corrected_error_percent": 12.323676275708818,
            "boundary_edges": 732,
            "expected_holes": 2,
            "recovered_holes": 0,
            "hole_measurements_mm": "",
            "topology_status": "fail",
            "notes": "TSDF and Poisson both filled the vertical bores",
        },
        {
            "case_id": "stepped_cylinder_steep",
            "display_name": "阶梯圆柱双深孔（陡俯视）",
            "truth_volume_cm3": 858.1271496927825,
            "raw_volume_cm3": 920.6197864124955,
            "raw_error_percent": 7.282444885013363,
            "corrected_volume_cm3": 953.2301272221765,
            "corrected_error_percent": 11.082620747221641,
            "boundary_edges": 1894,
            "expected_holes": 2,
            "recovered_holes": 1,
            "hole_measurements_mm": "18.712x16.369",
            "topology_status": "fail",
            "notes": "Only one undersized candidate; expected diameter 24 mm",
        },
    ]

    csv_path = PUBLIC_RESULT_DIR / "four_camera_experiments.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "date": "2026-11-17",
        "capture_protocol": {
            "camera_count": 4,
            "turntable_views": 12,
            "frequencies": [1, 4, 12],
            "phase_degrees": [0, 90, 180, 270],
            "frames_per_object": 576,
        },
        "experiments": rows,
        "interpretation": (
            "Four-camera dual-branch reconstruction succeeds on horizontal "
            "and paired through-holes, but does not reliably preserve "
            "high-aspect-ratio vertical bores. Surface-offset correction is "
            "not trusted when topology is already wrong."
        ),
    }
    (PUBLIC_RESULT_DIR / "four_camera_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    sanitize_logs()
    copy_results()
    summarize_cuboids()
    summarize_sweep()
    summarize_four_camera_experiments()
    print("public artifacts prepared:", ROOT / "artifacts")


if __name__ == "__main__":
    main()

