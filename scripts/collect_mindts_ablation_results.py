#!/usr/bin/env python3
"""Collect MindTS ablation experiment metrics into compact CSV tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["KR", "EWJ", "MDT", "Environment", "Energy", "Weather"]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "anomaly_segment_visualization" / "figures" / "qwen3_1p7b" / "ablation_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=PROJECT_ROOT / "result")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-tag", default="qwen3_1p7b")
    return parser.parse_args()


def latest_csv(dataset_dir: Path) -> Path | None:
    files = sorted(dataset_dir.glob("MindTS*.csv"), key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def variant_from_root(root_name: str, model_tag: str) -> str | None:
    prefix = f"label_ablation_{model_tag}_"
    if root_name.startswith(prefix):
        return root_name[len(prefix):]
    return None


def collect_records(result_root: Path, model_tag: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for sweep_root in sorted(result_root.glob(f"label_ablation_{model_tag}_*")):
        variant = variant_from_root(sweep_root.name, model_tag)
        if variant is None:
            continue
        for dataset in DATASETS:
            dataset_dir = sweep_root / dataset
            csv_path = latest_csv(dataset_dir)
            if csv_path is None:
                continue
            data = pd.read_csv(csv_path)
            if data.empty:
                continue
            config_path = dataset_dir / "experiment_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
            record = {
                "variant": variant,
                "dataset": dataset,
                "save_root": sweep_root.name,
                "csv_path": str(csv_path),
                "run_id": config.get("run_id", ""),
            }
            threshold_metrics = ["f_score", "adjust_f_score", "affiliation_f"]
            for metric in threshold_metrics:
                if metric in data:
                    values = pd.to_numeric(data[metric], errors="coerce")
                    record[f"{metric}_mean"] = float(values.mean())
                    record[f"{metric}_best"] = float(values.max())
            score_metrics = ["auc_roc", "auc_pr", "R_AUC_ROC", "R_AUC_PR", "VUS_ROC", "VUS_PR"]
            for metric in score_metrics:
                if metric in data:
                    values = pd.to_numeric(data[metric], errors="coerce")
                    record[metric] = float(values.mean())
            records.append(record)
    return pd.DataFrame(records)


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return records
    metric_columns = [
        col
        for col in [
            "f_score_mean",
            "f_score_best",
            "adjust_f_score_mean",
            "adjust_f_score_best",
            "affiliation_f_mean",
            "affiliation_f_best",
            "auc_roc",
            "auc_pr",
            "R_AUC_ROC",
            "R_AUC_PR",
            "VUS_ROC",
            "VUS_PR",
        ]
        if col in records.columns
    ]
    aggregations = {"dataset_count": ("dataset", "nunique")}
    for metric in metric_columns:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    summary = records.groupby("variant").agg(**aggregations).reset_index()
    sort_column = "VUS_ROC_mean" if "VUS_ROC_mean" in summary.columns else "variant"
    return summary.sort_values(sort_column, ascending=sort_column == "variant")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = collect_records(args.result_root, args.model_tag)
    summary = summarize(records)
    records.to_csv(args.output_dir / "mindts_ablation_records.csv", index=False)
    summary.to_csv(args.output_dir / "mindts_ablation_summary.csv", index=False)
    print(f"records={len(records)} summary_rows={len(summary)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
