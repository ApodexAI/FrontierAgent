"""Render answer-free PNG previews directly from agent-visible CSV inputs."""

from __future__ import annotations

import argparse
import csv
import html
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


PAPER = "#f8fafb"
INK = "#18324a"
MUTED = "#60788c"
LINE = "#d5e0e8"
BLUE = "#467fc5"
DEEP = "#3d6281"
TEAL = "#168b8b"
GOLD = "#c6842c"
PALETTE = [BLUE, DEEP, TEAL, GOLD]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def scale_points(
    points: list[tuple[float, float]],
    bounds: tuple[float, float, float, float],
    box: tuple[float, float, float, float],
    reverse_x: bool,
) -> str:
    xmin, xmax, ymin, ymax = bounds
    left, top, width, height = box
    coords = []
    for xvalue, yvalue in points:
        xfraction = (xvalue - xmin) / (xmax - xmin)
        if reverse_x:
            xfraction = 1 - xfraction
        x = left + xfraction * width
        y = top + height - (yvalue - ymin) / (ymax - ymin) * height
        coords.append(f"{x:.2f},{y:.2f}")
    return " ".join(coords)


def tick_text(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 10000:
        return f"{value / 1000:.0f}k"
    if magnitude >= 1000:
        return f"{value / 1000:.1f}k"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def chart_svg(
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, str, list[tuple[float, float]]]],
    reverse_x: bool = False,
) -> str:
    width, height = 1400, 820
    left, top, plot_w, plot_h = 135, 205, 1160, 470
    xs = [x for _, _, values in series for x, _ in values]
    ys = [y for _, _, values in series for _, y in values]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xpad = max((xmax - xmin) * .025, 1e-9)
    ypad = max((ymax - ymin) * .09, 1e-9)
    bounds = (xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad)

    grid, labels = [], []
    for index in range(5):
        fraction = index / 4
        x = left + fraction * plot_w
        y = top + fraction * plot_h
        xvalue = bounds[1] - fraction * (bounds[1] - bounds[0]) if reverse_x else bounds[0] + fraction * (bounds[1] - bounds[0])
        yvalue = bounds[3] - fraction * (bounds[3] - bounds[2])
        grid.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" />')
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" />')
        labels.append(f'<text x="{x:.1f}" y="{top + plot_h + 36}" text-anchor="middle">{tick_text(xvalue)}</text>')
        labels.append(f'<text x="{left - 18}" y="{y + 6:.1f}" text-anchor="end">{tick_text(yvalue)}</text>')

    paths, legend = [], []
    cursor_x, cursor_y = left, 154
    for name, color, values in series:
        points = scale_points(values, bounds, (left, top, plot_w, plot_h), reverse_x)
        paths.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round" />')
        label_width = max(175, len(name) * 11 + 62)
        if cursor_x + label_width > left + plot_w:
            cursor_x = left
            cursor_y += 31
        legend.append(f'<line x1="{cursor_x}" y1="{cursor_y}" x2="{cursor_x + 34}" y2="{cursor_y}" stroke="{color}" stroke-width="5" /><text x="{cursor_x + 45}" y="{cursor_y + 7}" class="legend-text">{html.escape(name)}</text>')
        cursor_x += label_width

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="{PAPER}" />
  <text x="{left}" y="62" class="title">{html.escape(title)}</text>
  <text x="{left}" y="99" class="subtitle">{html.escape(subtitle)}</text>
  <g class="legend">{''.join(legend)}</g>
  <g class="grid">{''.join(grid)}</g>
  <g class="ticks">{''.join(labels)}</g>
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis" />
  {''.join(paths)}
  <text x="{left + plot_w / 2}" y="780" class="axis-label" text-anchor="middle">{html.escape(x_label)}</text>
  <text x="{left}" y="{top - 22}" class="y-label">{html.escape(y_label)}</text>
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; fill: {MUTED}; font-size: 20px; }}
    .title {{ font-family: Georgia, 'Times New Roman', serif; fill: {INK}; font-size: 43px; }}
    .subtitle {{ font-size: 21px; }}
    .grid line {{ stroke: {LINE}; stroke-width: 1.4; }}
    .axis {{ stroke: {DEEP}; stroke-width: 2; }}
    .axis-label, .y-label {{ fill: {INK}; font-size: 22px; }}
    .legend-text {{ fill: {INK}; font-size: 19px; font-weight: 600; }}
    .ticks text {{ font-size: 17px; }}
  </style>
</svg>'''


def render_png(output: Path, svg: str) -> None:
    converter = shutil.which("rsvg-convert") or "/opt/homebrew/bin/rsvg-convert"
    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as handle:
        handle.write(svg)
        source = Path(handle.name)
    try:
        subprocess.run([converter, "-w", "1600", "-o", str(output), str(source)], check=True)
    finally:
        source.unlink(missing_ok=True)


def build_qnmr(tasks: Path, output: Path) -> None:
    source = rows(tasks / "task_094_qnmr_purity_qc" / "input" / "nmr_spectra.csv")
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    keep = ["BATCH_A_R1", "BATCH_A_R2", "BATCH_B_R1", "BATCH_B_R2"]
    for row in source:
        if row["spectrum_id"] in keep:
            groups[row["spectrum_id"]].append((float(row["ppm"]), float(row["intensity"])))
    series = [(key.replace("_", " · "), PALETTE[index], sorted(groups[key])) for index, key in enumerate(keep)]
    render_png(output, chart_svg("Raw quantitative ¹H NMR spectra", "Four replicate inputs before reference correction, baseline processing, or integration", "Chemical shift (ppm)", "Raw intensity", series, reverse_x=True))


def build_lfa(tasks: Path, output: Path) -> None:
    source = rows(tasks / "task_104_aln_bn_laser_flash" / "input" / "lfa_temperature_traces.csv")
    keep = ["ALN_0BN_R1", "ALN_10BN_R1", "ALN_20BN_R1"]
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in source:
        if row["run_id"] in keep:
            groups[row["run_id"]].append((float(row["time_ms"]), float(row["detector_signal_V"])))
    series = [(key.replace("_", " · "), PALETTE[index], sorted(groups[key])) for index, key in enumerate(keep)]
    render_png(output, chart_svg("Laser-flash temperature-rise inputs", "One raw detector trace from each BN loading before curve QC or thermal-property calculations", "Time (ms)", "Detector signal (V)", series))


def build_gitt(tasks: Path, output: Path) -> None:
    source = rows(tasks / "task_099_hardcarbon_gitt_diffusion" / "input" / "gitt_timeseries.csv")
    values = sorted(((float(row["time_s"]), float(row["voltage_V"])) for row in source if row["cell_id"] == "HC_1100_C1"))
    origin = values[0][0]
    values = [((x - origin) / 3600, y) for x, y in values]
    render_png(output, chart_svg("Raw GITT voltage sequence", "Pulse, rest, and relaxation measurements from one complete input cell record", "Elapsed time (h)", "Voltage (V)", [("HC 1100 · cell 1", BLUE, values)]))


def build_photoredox(tasks: Path, output: Path) -> None:
    source = rows(tasks / "task_028_photoredox_quantum_yield" / "input" / "led_spectra_raw.csv")
    keep = [("Q_STANDARD_PRE", "Standard · pre", BLUE), ("Q_FILTER50_PRE", "50% filter · pre", DEEP), ("Q_SHORTPATH_PRE", "Short path · pre", TEAL)]
    series = []
    for session_id, label, color in keep:
        values = sorted(
            (float(row["wavelength_nm"]), float(row["raw_counts"]))
            for row in source
            if row["session_id"] == session_id and row["spectrum_replicate"] == "1"
        )
        series.append((label, color, values))
    render_png(output, chart_svg("Raw blue-LED spectra", "Three pre-reaction input spectra before dark correction, detector-response correction, or photon weighting", "Wavelength (nm)", "Raw detector counts", series))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("materials_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    tasks = args.materials_root / "tasks"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_qnmr(tasks, args.output_dir / "qnmr-input.png")
    build_lfa(tasks, args.output_dir / "lfa-input.png")
    build_gitt(tasks, args.output_dir / "gitt-input.png")
    build_photoredox(tasks, args.output_dir / "photoredox-input.png")


if __name__ == "__main__":
    main()
