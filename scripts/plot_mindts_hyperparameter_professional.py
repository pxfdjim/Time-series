#!/usr/bin/env python3
"""Draw paper-style MindTS hyperparameter sensitivity figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "anomaly_segment_visualization" / "figures" / "qwen3_1p7b" / "hyperparameter_analysis"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR
FONT_REGULAR = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf")
FONT_ITALIC = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf")
DATASETS = ["KR", "EWJ", "MDT", "Environment", "Energy", "Weather"]
HPARAMS = [
    ("mask_ratio", "Mask Ratio", "mask ratio", 0.4),
    ("lamda1", r"Alignment Weight $\lambda_1$", r"alignment weight $\lambda_1$", 1.0),
]
METRIC_ROWS = [
    ("mean_aff_f", "Aff-F (%)", "#2f7d4a"),
    ("VUS_PR", "VUS-PR (%)", "#c15d27"),
    ("VUS_ROC", "VUS-ROC (%)", "#1f5aa6"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="mindts_hyperparameter_sensitivity_professional")
    return parser.parse_args()


def configure_matplotlib() -> None:
    for font_path in [FONT_REGULAR, FONT_BOLD, FONT_ITALIC]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "axes.titlesize": 11.6,
            "axes.labelsize": 10.4,
            "xtick.labelsize": 9.3,
            "ytick.labelsize": 9.3,
            "axes.linewidth": 0.72,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_records(input_dir: Path) -> pd.DataFrame:
    vus_path = input_dir / "mindts_hyperparameter_vus_records.csv"
    aff_path = input_dir / "mindts_hyperparameter_aff_f_records.csv"
    if not vus_path.exists() or not aff_path.exists():
        raise FileNotFoundError(
            "Missing hyperparameter records. Expected "
            f"{vus_path.name} and {aff_path.name} under {input_dir}."
        )
    vus = pd.read_csv(vus_path)
    aff = pd.read_csv(aff_path)[["hparam", "value", "dataset", "mean_aff_f", "best_aff_f"]]
    records = vus.merge(aff, on=["hparam", "value", "dataset"], how="inner")
    records["dataset"] = pd.Categorical(records["dataset"], categories=DATASETS, ordered=True)
    return records.sort_values(["hparam", "value", "dataset"]).reset_index(drop=True)


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["mean_aff_f", "best_aff_f", "VUS_PR", "VUS_ROC", "R_AUC_PR", "R_AUC_ROC"]
    aggregations = {"dataset_count": ("dataset", "nunique")}
    for metric in metric_columns:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    return records.groupby(["hparam", "value"], observed=False).agg(**aggregations).reset_index()


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="0.86", linewidth=0.55, linestyle=(0, (3, 2)))
    ax.grid(axis="x", color="0.92", linewidth=0.42, linestyle=":")
    ax.tick_params(length=2.8, width=0.72, pad=2.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.72)


def set_tight_ylim(ax: plt.Axes, values: np.ndarray, std: np.ndarray) -> None:
    lower = float(np.nanmin(values - std))
    upper = float(np.nanmax(values + std))
    pad = max(1.0, (upper - lower) * 0.22)
    ax.set_ylim(max(0.0, lower - pad), min(100.0, upper + pad))


def plot_sensitivity(summary: pd.DataFrame, output_dir: Path, stem: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.45, 3.92))
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for row_idx, (hparam, row_title, xlabel, default_value) in enumerate(HPARAMS):
        part = summary[summary["hparam"] == hparam].sort_values("value").reset_index(drop=True)
        x = part["value"].to_numpy(dtype=float)
        x_labels = [f"{value:g}" for value in x]
        x_pos = np.arange(len(x), dtype=float)

        for col_idx, (metric, ylabel, color) in enumerate(METRIC_ROWS):
            ax = axes[row_idx, col_idx]
            mean = part[f"{metric}_mean"].to_numpy(dtype=float) * 100.0
            std = part[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float) * 100.0
            best_idx = int(np.nanargmax(mean))

            ax.fill_between(x_pos, mean - std, mean + std, color=color, alpha=0.13, linewidth=0)
            ax.plot(x_pos, mean, color=color, linewidth=1.45, marker="o", markersize=4.4)
            ax.scatter(
                [x_pos[best_idx]],
                [mean[best_idx]],
                s=46,
                marker="*",
                color="#111111",
                zorder=4,
                linewidth=0.3,
            )

            if default_value in x:
                default_idx = int(np.where(np.isclose(x, default_value))[0][0])
                ax.axvline(
                    default_idx,
                    color="0.25",
                    linewidth=0.78,
                    linestyle=(0, (4, 2)),
                    alpha=0.68,
                )
                if col_idx == 0:
                    ax.text(
                        default_idx + 0.04,
                        0.98,
                        "default",
                        transform=ax.get_xaxis_transform(),
                        ha="left",
                        va="top",
                        fontsize=7.4,
                        color="0.25",
                    )

            ax.text(
                0.015,
                0.92,
                panel_labels[row_idx * 3 + col_idx],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.8,
                fontweight="bold",
            )
            ax.text(
                0.985,
                0.08,
                f"best={x[best_idx]:g}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8.3,
                color="0.18",
            )
            if row_idx == 0:
                ax.set_title(ylabel.replace(" (%)", ""), fontweight="bold", pad=5)
            if col_idx == 0:
                ax.set_ylabel(f"{row_title}\nScore (%)", fontweight="bold")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel(xlabel)
            set_tight_ylim(ax, mean, std)
            style_axis(ax)

    legend_handles = [
        Line2D([0], [0], color="#111111", marker="*", linewidth=0, markersize=8.2, label="best mean"),
        Line2D([0], [0], color="0.25", linestyle=(0, (4, 2)), linewidth=0.8, label="default"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 1.01),
        ncol=2,
        frameon=False,
        fontsize=9.1,
        handlelength=1.8,
        columnspacing=1.7,
    )
    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.13, top=0.86, hspace=0.47, wspace=0.25)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def heatmap_matrix(records: pd.DataFrame, hparam: str, metric: str) -> tuple[np.ndarray, list[str]]:
    part = records[records["hparam"] == hparam].copy()
    values = sorted(part["value"].unique())
    matrix = np.full((len(DATASETS), len(values)), np.nan, dtype=float)
    for row_idx, dataset in enumerate(DATASETS):
        for col_idx, value in enumerate(values):
            match = part[(part["dataset"] == dataset) & np.isclose(part["value"], value)]
            if not match.empty:
                matrix[row_idx, col_idx] = float(match.iloc[0][metric]) * 100.0
    return matrix, [f"{value:g}" for value in values]


def plot_heatmaps(records: pd.DataFrame, output_dir: Path, stem: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.55))
    heat_specs = [
        ("mask_ratio", "mean_aff_f", "Mask Ratio: Aff-F"),
        ("lamda1", "mean_aff_f", r"Alignment Weight $\lambda_1$: Aff-F"),
        ("mask_ratio", "VUS_ROC", "Mask Ratio: VUS-ROC"),
        ("lamda1", "VUS_ROC", r"Alignment Weight $\lambda_1$: VUS-ROC"),
    ]
    for ax, (hparam, metric, title) in zip(axes.flat, heat_specs):
        matrix, labels = heatmap_matrix(records, hparam, metric)
        valid = matrix[np.isfinite(matrix)]
        vmin = float(np.nanpercentile(valid, 5))
        vmax = float(np.nanpercentile(valid, 95))
        image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontweight="bold", pad=4)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(np.arange(len(DATASETS)))
        ax.set_yticklabels(DATASETS)
        ax.tick_params(length=0, pad=2.0)
        best_cols = np.nanargmax(matrix, axis=1)
        for row_idx, col_idx in enumerate(best_cols):
            ax.scatter(col_idx, row_idx, marker="*", s=25, color="#1b1b1b", linewidth=0.25)
        for spine in ax.spines.values():
            spine.set_linewidth(0.72)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.018)
        cbar.ax.tick_params(labelsize=7.4, length=2.4, pad=1.8)
        cbar.outline.set_linewidth(0.6)

    axes[1, 0].set_xlabel("mask ratio")
    axes[1, 1].set_xlabel(r"alignment weight $\lambda_1$")
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.115, top=0.925, hspace=0.36, wspace=0.27)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}_heatmap.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    records = load_records(args.input_dir)
    summary = summarize(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.output_dir / f"{args.stem}_records.csv", index=False)
    summary.to_csv(args.output_dir / f"{args.stem}_summary.csv", index=False)
    plot_sensitivity(summary, args.output_dir, args.stem)
    plot_heatmaps(records, args.output_dir, args.stem)
    print(f"records={len(records)} summary_rows={len(summary)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
