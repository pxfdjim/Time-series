#!/usr/bin/env python3
"""Collect and plot MindTS hyperparameter sweep results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "anomaly_segment_visualization" / "figures" / "qwen3_1p7b" / "hyperparameter_analysis"
FONT_REGULAR = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf")
FONT_ITALIC = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf")
DATASETS = ["KR", "EWJ", "MDT", "Environment", "Energy", "Weather"]
METRICS = ["VUS_ROC", "VUS_PR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-roots",
        nargs="+",
        type=Path,
        default=[PROJECT_ROOT / "result", PROJECT_ROOT / "result_table"],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-tag", default="qwen3_1p7b")
    parser.add_argument("--stem", default="mindts_hyperparameter_vus")
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
            "axes.titlesize": 10.5,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def slug_to_float(value: str) -> float:
    return float(value.replace("m", "-", 1).replace("p", "."))


def hparam_from_root(root_name: str, model_tag: str) -> tuple[str, float] | None:
    mask_match = re.fullmatch(rf"label_hparam_{re.escape(model_tag)}_mask_ratio_m(.+)", root_name)
    if mask_match:
        return "mask_ratio", slug_to_float(mask_match.group(1))
    lamda_match = re.fullmatch(rf"label_hparam_{re.escape(model_tag)}_lamda1_l(.+)", root_name)
    if lamda_match:
        return "lamda1", slug_to_float(lamda_match.group(1))
    return None


def load_hparam_from_config(dataset_dir: Path, fallback: tuple[str, float]) -> tuple[str, float]:
    config_path = dataset_dir / "experiment_config.json"
    if not config_path.exists():
        return fallback
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return str(config.get("hparam", fallback[0])), float(config.get("hparam_value", fallback[1]))


def latest_csv(dataset_dir: Path) -> Path | None:
    files = sorted(dataset_dir.glob("MindTS*.csv"), key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def collect_records(result_roots: list[Path], model_tag: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str, float, str]] = set()
    for result_root in result_roots:
        if not result_root.exists():
            continue
        for sweep_root in sorted(result_root.glob(f"label_hparam_{model_tag}_*")):
            parsed = hparam_from_root(sweep_root.name, model_tag)
            if parsed is None:
                continue
            for dataset in DATASETS:
                dataset_dir = sweep_root / dataset
                csv_path = latest_csv(dataset_dir)
                if csv_path is None:
                    continue
                hparam, value = load_hparam_from_config(dataset_dir, parsed)
                key = (str(csv_path.resolve()), hparam, value, dataset)
                if key in seen:
                    continue
                seen.add(key)
                data = pd.read_csv(csv_path)
                if data.empty:
                    continue
                row = data.iloc[0]
                record = {
                    "result_root": str(result_root),
                    "save_root": sweep_root.name,
                    "dataset": dataset,
                    "hparam": hparam,
                    "value": value,
                    "csv_path": str(csv_path),
                }
                for metric in ["VUS_ROC", "VUS_PR", "R_AUC_ROC", "R_AUC_PR", "auc_roc", "auc_pr"]:
                    if metric in row:
                        record[metric] = float(row[metric])
                records.append(record)
    return pd.DataFrame(records)


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return records
    return (
        records.groupby(["hparam", "value"], as_index=False)
        .agg(
            dataset_count=("dataset", "nunique"),
            VUS_ROC_mean=("VUS_ROC", "mean"),
            VUS_ROC_std=("VUS_ROC", "std"),
            VUS_PR_mean=("VUS_PR", "mean"),
            VUS_PR_std=("VUS_PR", "std"),
            R_AUC_ROC_mean=("R_AUC_ROC", "mean"),
            R_AUC_PR_mean=("R_AUC_PR", "mean"),
        )
        .sort_values(["hparam", "value"])
    )


def plot_summary(summary: pd.DataFrame, output_dir: Path, stem: str) -> None:
    if summary.empty:
        raise ValueError("No completed hyperparameter records were found.")

    colors = {"VUS_ROC": "#1f5aa6", "VUS_PR": "#c15d27"}
    labels = {"VUS_ROC": "VUS-ROC", "VUS_PR": "VUS-PR"}
    hparams = [("mask_ratio", "Mask ratio"), ("lamda1", r"Alignment weight $\lambda_1$")]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), sharey=True)

    for ax, (hparam, title) in zip(axes, hparams):
        part = summary[summary["hparam"] == hparam].sort_values("value").reset_index(drop=True)
        x = np.arange(len(part), dtype=float)
        for metric in METRICS:
            mean = part[f"{metric}_mean"].to_numpy(dtype=float)
            std = part[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(
                x,
                mean,
                marker="o",
                markersize=4.2,
                linewidth=1.35,
                color=colors[metric],
                label=labels[metric],
            )
            ax.fill_between(x, mean - std, mean + std, color=colors[metric], alpha=0.13, linewidth=0)
        ax.set_title(title, fontweight="bold", pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{value:g}" for value in part["value"]])
        ax.set_xlabel(hparam)
        ax.grid(axis="y", linestyle=(0, (3, 2)), linewidth=0.55, color="0.72", alpha=0.72)
        ax.grid(axis="x", linestyle=":", linewidth=0.40, color="0.82", alpha=0.65)
        ax.set_ylim(0.0, 1.02)
        ax.tick_params(length=2.8, pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)

    axes[0].set_ylabel("score")
    axes[1].legend(frameon=False, loc="lower right", fontsize=8.6, handlelength=2.0)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.205, top=0.845, wspace=0.115)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    records = collect_records(args.result_roots, args.model_tag)
    summary = summarize(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.output_dir / f"{args.stem}_records.csv", index=False)
    summary.to_csv(args.output_dir / f"{args.stem}_summary.csv", index=False)
    if not summary.empty:
        plot_summary(summary, args.output_dir, args.stem)
    print(f"records={len(records)} summary_rows={len(summary)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
