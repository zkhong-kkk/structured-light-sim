"""只使用标准库和公开 CSV 生成 GitHub 可显示的 SVG 图表。"""

from __future__ import annotations

import csv
from html import escape
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"
RESULT_DIR = ROOT / "artifacts" / "results"


def svg_document(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
        '<style>'
        'text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#172033}'
        '.title{font-size:22px;font-weight:700}'
        '.label{font-size:12px}'
        '.value{font-size:13px;font-weight:700}'
        '.grid{stroke:#d9e0ea;stroke-width:1}'
        '</style>\n'
        f'{body}\n</svg>\n'
    )


def cuboid_chart() -> None:
    with (RESULT_DIR / "random_cuboids_benchmark.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["status"] == "success"
        ]

    values = [float(row["corrected_error_percent"]) for row in rows]
    width, height = 1000, 460
    left, top, right, bottom = 70, 55, 25, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = 3.5
    bar_width = plot_width / len(values)
    parts = [
        '<text x="500" y="30" text-anchor="middle" class="title">'
        '30 个随机长方体校正后体积误差</text>'
    ]

    for tick in range(0, 4):
        y = top + plot_height * (1 - tick / maximum)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" '
            f'y2="{y:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left-10}" y="{y+4:.1f}" '
            f'text-anchor="end" class="label">{tick}%</text>'
        )

    for index, (row, value) in enumerate(zip(rows, values)):
        x = left + index * bar_width + 1
        bar_height = plot_height * value / maximum
        y = top + plot_height - bar_height
        color = "#d95d39" if value > 2.0 else "#277da1"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" '
            f'width="{max(bar_width-2, 1):.1f}" height="{bar_height:.1f}" '
            f'fill="{color}"><title>{escape(row["case_id"])}: '
            f'{value:.3f}%</title></rect>'
        )
        if index % 2 == 0:
            parts.append(
                f'<text x="{x+bar_width/2:.1f}" y="{height-bottom+18}" '
                f'text-anchor="middle" class="label" '
                f'transform="rotate(55 {x+bar_width/2:.1f} '
                f'{height-bottom+18})">{index:02d}</text>'
            )

    average = mean(values)
    y_mean = top + plot_height * (1 - average / maximum)
    parts.append(
        f'<line x1="{left}" y1="{y_mean:.1f}" x2="{width-right}" '
        f'y2="{y_mean:.1f}" stroke="#f9a826" stroke-width="3" '
        f'stroke-dasharray="7 5"/>'
    )
    parts.append(
        f'<text x="{width-right-4}" y="{y_mean-7:.1f}" '
        f'text-anchor="end" class="value">平均 {average:.3f}%</text>'
    )
    parts.append(
        '<text x="500" y="450" text-anchor="middle" class="label">'
        '样本编号（蓝色 ≤2%，橙色 >2%）</text>'
    )
    (ASSET_DIR / "random_cuboid_errors.svg").write_text(
        svg_document(width, height, "\n".join(parts)),
        encoding="utf-8",
    )


def method_chart() -> None:
    with (RESULT_DIR / "hole_method_comparison.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    width, height = 820, 450
    left, top, right, bottom = 85, 60, 30, 95
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = 1.4
    bar_width = 120
    gap = (plot_width - bar_width * len(rows)) / (len(rows) + 1)
    colors = ["#6c757d", "#577590", "#2a9d8f"]
    parts = [
        '<text x="410" y="32" text-anchor="middle" class="title">'
        '水平通孔：不同观测方案体积误差</text>'
    ]
    for tick in [0.0, 0.5, 1.0, 1.4]:
        y = top + plot_height * (1 - tick / maximum)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" '
            f'y2="{y:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left-12}" y="{y+4:.1f}" '
            f'text-anchor="end" class="label">{tick:.1f}%</text>'
        )

    for index, row in enumerate(rows):
        value = float(row["corrected_error_percent"])
        x = left + gap + index * (bar_width + gap)
        bar_height = plot_height * value / maximum
        y = top + plot_height - bar_height
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" '
            f'height="{bar_height:.1f}" rx="5" fill="{colors[index]}"/>'
        )
        parts.append(
            f'<text x="{x+bar_width/2:.1f}" y="{y-9:.1f}" '
            f'text-anchor="middle" class="value">{value:.3f}%</text>'
        )
        label = row["display_name"]
        parts.append(
            f'<text x="{x+bar_width/2:.1f}" y="{height-bottom+25}" '
            f'text-anchor="middle" class="label">{escape(label)}</text>'
        )
        topology = (
            "恢复闭合孔截面"
            if row["closed_hole_loop"] == "True"
            else "未恢复孔截面"
        )
        parts.append(
            f'<text x="{x+bar_width/2:.1f}" y="{height-bottom+46}" '
            f'text-anchor="middle" class="label">{topology}</text>'
        )
    parts.append(
        '<text x="410" y="440" text-anchor="middle" class="label">'
        '真值体积 1008 cm³；本文拓扑结果来自独立 Poisson 分支</text>'
    )
    (ASSET_DIR / "hole_method_comparison.svg").write_text(
        svg_document(width, height, "\n".join(parts)),
        encoding="utf-8",
    )


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    cuboid_chart()
    method_chart()
    print("report assets generated:", ASSET_DIR)


if __name__ == "__main__":
    main()

