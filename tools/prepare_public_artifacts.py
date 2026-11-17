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
}

REDACTIONS = [
    (
        re.compile(
            r"/mnt/disk2/[^/]+/projects/structured-light-sim"
        ),
        "${PROJECT_ROOT}",
    ),
    (
        re.compile(r"/mnt/disk2/[^/]+"),
        "${USER_HOME}",
    ),
    (
        re.compile(r"/home_local/[^/.\s]+"),
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


def main() -> None:
    sanitize_logs()
    copy_results()
    summarize_cuboids()
    summarize_sweep()
    print("public artifacts prepared:", ROOT / "artifacts")


if __name__ == "__main__":
    main()

