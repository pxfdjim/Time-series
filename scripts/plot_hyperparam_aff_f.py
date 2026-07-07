#!/usr/bin/env python3
"""Plot hyperparameter sensitivity using Aff-F, V-PR, V-ROC (mean-threshold)."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "result"
OUTPUT_DIR = PROJECT_ROOT / "anomaly_segment_visualization" / "figures" / "qwen3_1p7b" / "hyperparameter_analysis"

FONT_REGULAR = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf")
FONT_ITALIC = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf")

DATASETS = ["KR", "EWJ", "MDT", "Environment", "Energy", "Weather"]
DATASET_COLORS = {
    "KR": "#e41a1c",
    "EWJ": "#377eb8",
    "MDT": "#4daf4a",
    "Environment": "#984ea3",
    "Energy": "#ff7f00",
    "Weather": "#a65628",
}
METRICS = ["Aff-F", "V-PR", "V-ROC"]
METRIC_LABELS = {"Aff-F": "Aff-F (%)", "V-PR": "V-PR (%)", "V-ROC": "V-ROC (%)"}
SENSITIVITY_PANELS = [
    {
        "hparam": "mask_ratio",
        "xlabel": "Mask Ratio",
        "left": {"dataset": "Environment", "metric": "Aff-F", "color": "#2f7d45", "marker": "^"},
        "right": {"dataset": "EWJ", "metric": "V-PR", "color": "#c96b1c", "marker": "s"},
    },
    {
        "hparam": "lamda1",
        "xlabel": r"Alignment Weight $\lambda_1$",
        "left": {"dataset": "Environment", "metric": "Aff-F", "color": "#2f7d45", "marker": "^"},
        "right": {"dataset": "EWJ", "metric": "V-PR", "color": "#c96b1c", "marker": "s"},
    },
]


def configure_matplotlib() -> None:
    for fp in [FONT_REGULAR, FONT_BOLD, FONT_ITALIC]:
        if fp.exists():
            font_manager.fontManager.addfont(str(fp))
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "axes.titlesize": 11.2,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 9.2,
        "ytick.labelsize": 9.2,
        "legend.fontsize": 8.6,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def slug_to_float(value: str) -> float:
    return float(value.replace("m", "-", 1).replace("p", "."))


def extract_metrics(threshold_file: Path) -> dict[str, float]:
    """Extract mean-threshold Aff-F, V-PR, V-ROC from a threshold_metrics.txt file."""
    text = threshold_file.read_text()
    results = {}
    for metric in METRICS:
        # Match lines like "Aff-F: 0.7939..." (mean across thresholds)
        pattern = rf"{re.escape(metric)}:\s*([\d.]+)"
        m = re.search(pattern, text)
        if m:
            results[metric] = float(m.group(1)) * 100  # to percentage
    return results


def collect_hparam_data(hparam: str) -> dict:
    """Collect data for a given hyperparameter (mask_ratio or lamda1)."""
    if hparam == "mask_ratio":
        pattern = "label_hparam_qwen3_1p7b_mask_ratio_m*"
        slug_prefix = "label_hparam_qwen3_1p7b_mask_ratio_m"
    else:
        pattern = "label_hparam_qwen3_1p7b_lamda1_l*"
        slug_prefix = "label_hparam_qwen3_1p7b_lamda1_l"

    data = {}  # value -> {dataset -> {metric -> value}}

    for subdir in sorted(RESULT_DIR.glob(pattern)):
        if not subdir.is_dir():
            continue
        slug = subdir.name
        value_str = slug[len(slug_prefix):]
        value = slug_to_float(value_str)

        data[value] = {}
        for ds in DATASETS:
            ds_dir = subdir / ds
            if not ds_dir.exists():
                continue
            # Find the threshold_metrics.txt file
            txt_files = list(ds_dir.glob("*.threshold_metrics.txt"))
            if not txt_files:
                continue
            metrics = extract_metrics(txt_files[0])
            if metrics:
                data[value][ds] = metrics

    return data


def plot_hyperparameter(data: dict, hparam: str, hparam_label: str, axes_row: list) -> None:
    """Plot three metrics for a hyperparameter in one row of subplots."""
    values = sorted(data.keys())

    for col_idx, metric in enumerate(METRICS):
        ax = axes_row[col_idx]

        for ds in DATASETS:
            y_vals = []
            for v in values:
                if ds in data[v] and metric in data[v][ds]:
                    y_vals.append(data[v][ds][metric])
                else:
                    y_vals.append(np.nan)
            ax.plot(
                range(len(values)),
                y_vals,
                marker="o",
                markersize=4.8,
                linewidth=1.75,
                color=DATASET_COLORS[ds],
                label=ds,
            )

        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([f"{v:g}" for v in values])
        ax.set_xlabel(hparam_label)
        if col_idx == 0:
            ax.set_ylabel(f"{hparam_label}\n{METRIC_LABELS[metric]}", fontweight="bold")
        else:
            ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(axis="y", linestyle=(0, (3, 2)), linewidth=0.62, color="0.72", alpha=0.72)
        ax.grid(axis="x", linestyle=":", linewidth=0.42, color="0.82", alpha=0.58)
        ax.tick_params(length=3.0, pad=2.2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)

        ax.set_title(METRIC_LABELS[metric].replace(" (%)", ""), fontweight="bold", pad=5)


def series_from_data(data: dict, dataset: str, metric: str) -> tuple[list[float], list[float]]:
    values = sorted(data.keys())
    y_vals = []
    for value in values:
        if dataset in data[value] and metric in data[value][dataset]:
            y_vals.append(data[value][dataset][metric])
        else:
            y_vals.append(np.nan)
    return values, y_vals


def padded_limits(values: list[float], min_pad: float = 0.45) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    ymin, ymax = float(arr.min()), float(arr.max())
    span = max(ymax - ymin, min_pad)
    pad = span * 0.26
    return ymin - pad, ymax + pad


def plot_dual_axis_panel(ax: plt.Axes, data: dict, panel: dict) -> None:
    left = panel["left"]
    right = panel["right"]
    values, left_y = series_from_data(data, left["dataset"], left["metric"])
    _, right_y = series_from_data(data, right["dataset"], right["metric"])
    x = np.arange(len(values))

    left_line = ax.plot(
        x,
        left_y,
        color=left["color"],
        marker=left["marker"],
        markersize=4.0,
        linewidth=1.05,
        linestyle=(0, (3.5, 2.4)),
        label=left["dataset"],
    )[0]
    ax.set_ylabel(METRIC_LABELS[left["metric"]], color=left["color"], labelpad=4)
    ax.tick_params(axis="y", colors=left["color"], width=0.7, length=2.8, pad=2.0)
    ax.spines["left"].set_color(left["color"])
    ax.set_ylim(*padded_limits(left_y))

    ax_right = ax.twinx()
    right_line = ax_right.plot(
        x,
        right_y,
        color=right["color"],
        marker=right["marker"],
        markersize=3.8,
        linewidth=1.0,
        linestyle=(0, (1.1, 2.0)),
        label=right["dataset"],
    )[0]
    ax_right.set_ylabel(METRIC_LABELS[right["metric"]], color=right["color"], labelpad=4)
    ax_right.tick_params(axis="y", colors=right["color"], width=0.7, length=2.8, pad=2.0)
    ax_right.spines["right"].set_color(right["color"])
    ax_right.set_ylim(*padded_limits(right_y))

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in values])
    ax.set_xlabel(panel["xlabel"], labelpad=3)
    ax.grid(axis="y", linestyle=(0, (3, 2)), linewidth=0.52, color="0.76", alpha=0.65)
    ax.grid(axis="x", linestyle=":", linewidth=0.36, color="0.85", alpha=0.58)
    ax.tick_params(axis="x", width=0.7, length=2.8, pad=2.0)

    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    for spine in ax_right.spines.values():
        spine.set_linewidth(0.75)

    ax.legend(
        [left_line, right_line],
        [left_line.get_label(), right_line.get_label()],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        frameon=False,
        handlelength=2.1,
        columnspacing=0.9,
        borderaxespad=0.0,
    )


def main() -> None:
    configure_matplotlib()

    mask_data = collect_hparam_data("mask_ratio")
    lamda_data = collect_hparam_data("lamda1")

    print(f"mask_ratio values: {sorted(mask_data.keys())}")
    print(f"lamda1 values: {sorted(lamda_data.keys())}")

    # Save raw data as CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for hparam_name, data in [("mask_ratio", mask_data), ("lamda1", lamda_data)]:
        csv_path = OUTPUT_DIR / f"hyperparam_{hparam_name}_aff_pr_roc.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["value", "dataset", "Aff-F", "V-PR", "V-ROC"])
            for v in sorted(data.keys()):
                for ds in DATASETS:
                    if ds in data[v]:
                        row = data[v][ds]
                        writer.writerow([v, ds, row.get("Aff-F", ""), row.get("V-PR", ""), row.get("V-ROC", "")])
        print(f"Saved {csv_path}")

    # Compact paper-style view: two hyperparameters, two representative datasets,
    # and two metrics separated by left/right y axes.
    data_by_hparam = {"mask_ratio": mask_data, "lamda1": lamda_data}
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.45))

    for ax, panel in zip(axes, SENSITIVITY_PANELS):
        plot_dual_axis_panel(ax, data_by_hparam[panel["hparam"]], panel)

    fig.subplots_adjust(left=0.075, right=0.925, bottom=0.24, top=0.84, wspace=0.43)

    stem = "mindts_hyperparameter_aff_pr_roc"
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {OUTPUT_DIR / f'{stem}.pdf'}")


if __name__ == "__main__":
    main()
