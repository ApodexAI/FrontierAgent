"""Build input-only case-study previews.

The source bundle must contain only agent-visible task inputs. This script never
reads model outputs, scorecards, graders, or reference artifacts.
"""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


INK = "#18324a"
MUTED = "#60788c"
BLUE = "#467fc5"
DEEP = "#3d6281"
PALE = "#edf4fb"
PAPER = "#f8fafb"
LINE = "#d5e0e8"


def read_xy(path: Path, x_name: str, y_name: str) -> list[tuple[float, float]]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return [(float(row[x_name]), float(row[y_name])) for row in rows]


def scale_points(
    points: list[tuple[float, float]],
    bounds: tuple[float, float, float, float],
    box: tuple[float, float, float, float],
) -> str:
    xmin, xmax, ymin, ymax = bounds
    left, top, width, height = box
    coords = []
    for x, y in points:
        px = left + (x - xmin) / (xmax - xmin) * width
        py = top + height - (y - ymin) / (ymax - ymin) * height
        coords.append(f"{px:.2f},{py:.2f}")
    return " ".join(coords)


def chart_svg(
    output: Path,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, str, list[tuple[float, float]]]],
) -> None:
    width, height = 1200, 700
    left, top, plot_w, plot_h = 115, 150, 985, 440
    xs = [x for _, _, values in series for x, _ in values]
    ys = [y for _, _, values in series for _, y in values]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xpad = max((xmax - xmin) * 0.02, 1e-6)
    ypad = max((ymax - ymin) * 0.08, 1e-6)
    bounds = (xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad)

    grid = []
    labels = []
    for index in range(5):
        frac = index / 4
        x = left + frac * plot_w
        y = top + frac * plot_h
        x_value = bounds[0] + frac * (bounds[1] - bounds[0])
        y_value = bounds[3] - frac * (bounds[3] - bounds[2])
        grid.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" />')
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" />')
        labels.append(f'<text x="{x:.1f}" y="{top + plot_h + 34}" text-anchor="middle">{x_value:.2g}</text>')
        labels.append(f'<text x="{left - 18}" y="{y + 5:.1f}" text-anchor="end">{y_value:.2g}</text>')

    polylines = []
    legend = []
    for index, (name, color, values) in enumerate(series):
        points = scale_points(values, bounds, (left, top, plot_w, plot_h))
        polylines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />')
        legend_x = left + index * 250
        legend.append(f'<line x1="{legend_x}" y1="118" x2="{legend_x + 32}" y2="118" stroke="{color}" stroke-width="4" /><text x="{legend_x + 43}" y="124">{html.escape(name)}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{html.escape(title)}</title>
  <desc id="desc">{html.escape(subtitle)}</desc>
  <rect width="{width}" height="{height}" fill="{PAPER}" />
  <text x="{left}" y="55" class="title">{html.escape(title)}</text>
  <text x="{left}" y="84" class="subtitle">{html.escape(subtitle)}</text>
  <g class="grid">{''.join(grid)}</g>
  <g class="ticks">{''.join(labels)}</g>
  <g class="legend">{''.join(legend)}</g>
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis" />
  {''.join(polylines)}
  <text x="{left + plot_w / 2}" y="665" class="axis-label" text-anchor="middle">{html.escape(x_label)}</text>
  <text x="32" y="{top + plot_h / 2}" class="axis-label" text-anchor="middle" transform="rotate(-90 32 {top + plot_h / 2})">{html.escape(y_label)}</text>
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: {MUTED}; font-size: 17px; }}
    .title {{ font-family: Georgia, serif; fill: {INK}; font-size: 34px; }}
    .subtitle {{ font-size: 16px; }}
    .grid line {{ stroke: {LINE}; stroke-width: 1; }}
    .axis {{ stroke: {DEEP}; stroke-width: 1.5; }}
    .axis-label {{ fill: {INK}; font-size: 18px; }}
    .legend text {{ fill: {INK}; font-size: 16px; }}
    .ticks text {{ font-size: 14px; }}
  </style>
</svg>'''
    output.write_text(svg)


def build_xrd(tasks: Path, output: Path) -> None:
    root = tasks / "task_005_xrd_duplex_phase_quant" / "input" / "patterns"
    files = [
        ("LPBF fast", BLUE, "LPBF_FAST_R1.csv"),
        ("Dual laser", DEEP, "DUAL_LASER_R1.csv"),
        ("Solution annealed", "#8ca0ae", "SA_1100C_R1.csv"),
    ]
    series = [(name, color, read_xy(root / filename, "two_theta_deg", "intensity_counts")) for name, color, filename in files]
    chart_svg(output, "Raw diffraction patterns", "One input scan from each processing condition", "2θ (degrees)", "Intensity (counts)", series)


def build_eis(tasks: Path, output: Path) -> None:
    root = tasks / "task_116_eis_equivalent_circuit_analysis" / "input"
    series = [
        ("SOC 100 · scan 1", BLUE, read_xy(root / "Cell_1_GEIS_SOC100_scan1.csv", "Re(Ztot) [Ohm]", "-Im(Ztot) [Ohm]")),
        # The SOC 70 file contains two consecutive sweeps. Showing one avoids a
        # misleading connector between the end of one sweep and the next.
        ("SOC 70 · scan 1", DEEP, read_xy(root / "Cell_2_GEIS_SOC70.csv", "Re(Ztot) [Ohm]", "-Im(Ztot) [Ohm]")[:61]),
    ]
    chart_svg(output, "Raw impedance spectra", "Agent-visible Nyquist inputs before model selection or fitting", "Z′ (Ω)", "−Z″ (Ω)", series)


def build_cell_strip(tasks: Path, output: Path) -> None:
    root = tasks / "task_011_cell_migration_wound_healing" / "input" / "images"
    names = [("0 h", "0h-C1.jpg"), ("12 h", "12h-C1.jpg"), ("24 h", "24h-C1.jpg")]
    canvas = Image.new("RGB", (1500, 380), PAPER)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    for index, (label, filename) in enumerate(names):
        image = Image.open(root / filename).convert("RGB")
        crop = image.crop((0, 260, image.width, 900))
        crop.thumbnail((476, 270), Image.Resampling.LANCZOS)
        x = 12 + index * 496
        y = 58
        canvas.paste(crop, (x, y))
        draw.text((x, 24), label, fill=INK, font=font)
    draw.text((12, 344), "Agent-visible bright-field inputs · control replicate 1", fill=MUTED, font=font)
    canvas.save(output, quality=88, optimize=True, progressive=True)


def build_cell_card(tasks: Path, output: Path) -> None:
    root = tasks / "task_011_cell_migration_wound_healing" / "input" / "images"
    names = [("0 h", "0h-C1.jpg"), ("12 h", "12h-C1.jpg"), ("24 h", "24h-C1.jpg")]
    canvas = Image.new("RGB", (900, 600), PAPER)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    for index, (label, filename) in enumerate(names):
        image = Image.open(root / filename).convert("RGB")
        crop = image.crop((400, 140, 960, 980))
        crop.thumbnail((276, 470), Image.Resampling.LANCZOS)
        x = 12 + index * 296
        canvas.paste(crop, (x, 62))
        draw.text((x, 24), label, fill=INK, font=font)
    draw.text((12, 562), "Agent-visible bright-field inputs", fill=MUTED, font=font)
    canvas.save(output, quality=88, optimize=True, progressive=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("materials_root", type=Path, help="Path to case_study_materials")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    tasks = args.materials_root / "tasks"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_xrd(tasks, args.output_dir / "xrd-input.svg")
    build_eis(tasks, args.output_dir / "eis-input.svg")
    build_cell_strip(tasks, args.output_dir / "cell-migration-input.jpg")
    build_cell_card(tasks, args.output_dir / "cell-migration-card.jpg")


if __name__ == "__main__":
    main()
