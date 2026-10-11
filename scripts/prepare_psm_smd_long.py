#!/usr/bin/env python3
"""Convert official PSM and SMD raw splits into the repository's date,data,cols format.

Usage (from Time-series root):
    python scripts/prepare_psm_smd_long.py

Input:
    dataset/anomaly_detect/raw/PSM/{train.csv,test.csv,test_label.csv}
    dataset/anomaly_detect/raw/SMD/{train,test,test_label}/machine-*.txt
Output:
    dataset/anomaly_detect/data/PSM.csv
    dataset/anomaly_detect/data/SMD_machine-*.csv
    29 entries appended/updated in dataset/anomaly_detect/DETECT_META.csv

The official train/test boundary is preserved via metadata.train_lens. Features are
not normalized, resampled, or rounded by explicit precision reduction. PSM NaNs
are reported by channel and filled with zero; Inf is rejected. SMD nonfinite
features are rejected. Raw files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _verify_features(frame: pd.DataFrame, dataset: str, fill_nan: bool = False) -> pd.DataFrame:
    """Check numeric features; zero-fill only the explicitly requested PSM NaNs."""
    result = frame.apply(pd.to_numeric, errors="raise")
    array = result.to_numpy(dtype=np.float64)
    nan_counts = result.isna().sum()
    if fill_nan:
        print(f"[{dataset}] NaN counts before zero fill (total={int(nan_counts.sum()):,}):", flush=True)
        for index, (column, count) in enumerate(nan_counts.items(), start=1):
            print(f"  {column} -> channel{index}: {int(count):,}", flush=True)
    inf_count = int(np.isinf(array).sum())
    if fill_nan:
        print(f"[{dataset}] Inf count before conversion: {inf_count:,}", flush=True)
    if inf_count:
        raise ValueError(f"{dataset}: found {inf_count} Inf feature values")
    if nan_counts.any():
        if not fill_nan:
            raise ValueError(f"{dataset}: found {int(nan_counts.sum())} NaN feature values")
        result = result.fillna(0)
    if not np.isfinite(result.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{dataset}: nonfinite feature values remain after checking")
    return result


def _binary_labels(data: np.ndarray, name: str) -> np.ndarray:
    labels = np.asarray(data).reshape(-1)
    if not np.isfinite(labels.astype(np.float64)).all():
        raise ValueError(f"{name}: non-finite labels")
    uniq = set(np.unique(labels.astype(np.float64)).tolist())
    if not uniq.issubset({0.0, 1.0}):
        raise ValueError(f"{name}: expected binary labels, found {sorted(uniq)[:15]}")
    return labels.astype(np.int8)


def _validate_long(path: Path, values: pd.DataFrame, labels: np.ndarray,
                   train_length: int, test_labels: np.ndarray):
    """Read the generated CSV in chunks and compare every point to its source."""
    length = len(values)
    channel_names = list(values.columns) + ["label"]
    if not 0 < train_length < length:
        raise ValueError(f"{path.name}: invalid train/test boundary {train_length}/{length}")
    if len(test_labels) != length - train_length:
        raise ValueError(f"{path.name}: test label length differs from test length")
    if not np.array_equal(labels[:train_length], np.zeros(train_length, dtype=np.int8)):
        raise ValueError(f"{path.name}: training labels are not all zero")
    if not np.array_equal(labels[train_length:], test_labels):
        raise ValueError(f"{path.name}: official test labels were changed")

    source_columns = [values[col].to_numpy(dtype=np.float64) for col in values.columns]
    source_columns.append(labels.astype(np.float64))
    seen = 0
    for chunk in pd.read_csv(path, chunksize=250_000, float_precision="round_trip"):
        if list(chunk.columns) != ["date", "data", "cols"]:
            raise ValueError(f"{path.name}: expected date,data,cols columns")
        actual_data = pd.to_numeric(chunk["data"], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(actual_data).all():
            raise ValueError(f"{path.name}: generated CSV contains NaN/Inf")
        offset = 0
        while offset < len(chunk):
            channel = seen // length
            position = seen % length
            if channel >= len(channel_names):
                raise ValueError(f"{path.name}: generated CSV has too many rows")
            count = min(len(chunk) - offset, length - position)
            part = chunk.iloc[offset:offset + count]
            if not part["cols"].eq(channel_names[channel]).all():
                raise ValueError(f"{path.name}: wrong channel order at CSV row {seen + 2}")
            expected_dates = np.arange(position + 1, position + count + 1)
            if not np.array_equal(part["date"].to_numpy(), expected_dates):
                raise ValueError(f"{path.name}: wrong date order at CSV row {seen + 2}")
            expected_data = source_columns[channel][position:position + count]
            if not np.array_equal(actual_data[offset:offset + count], expected_data):
                raise ValueError(f"{path.name}: source values differ at CSV row {seen + 2}")
            seen += count
            offset += count
    expected_rows = length * len(channel_names)
    if seen != expected_rows:
        raise ValueError(f"{path.name}: expected {expected_rows} rows, found {seen}")
    print(f"[CHECK] {path.name.removesuffix('.partial')}: {len(values.columns)} channels, train={train_length:,}, "
          f"test={length - train_length:,}, labels={len(test_labels):,}, "
          f"rows={seen:,}, NaN/Inf=0, values and boundary match source", flush=True)


def _write_long(path: Path, values: pd.DataFrame, labels: np.ndarray,
                train_length: int, test_labels: np.ndarray):
    """Write contiguous per-channel groups, and the label group last.

    The existing process_data_df() assumes exactly this group order/length.
    """
    length = len(values)
    if len(labels) != length:
        raise ValueError(f"{path.name}: label length differs from feature length")
    if "label" in values.columns or "date" in values.columns or "cols" in values.columns:
        raise ValueError(f"{path.name}: conflicting feature column name")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".csv.partial")
    dates = np.arange(1, length + 1, dtype=np.int64)
    try:
        with temp.open("w", newline="", encoding="utf-8") as handle:
            for k, col in enumerate(list(values.columns) + ["label"]):
                value_arr = labels if col == "label" else values[col].to_numpy()
                chunk = pd.DataFrame({"date": dates, "data": value_arr, "cols": col})
                chunk.to_csv(handle, header=(k == 0), index=False, lineterminator="\n")
        _validate_long(temp, values, labels, train_length, test_labels)
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    print(f"[OK] {path.name}: timepoints={length:,}, features={values.shape[1]}, "
          f"CSV rows={(values.shape[1] + 1) * length:,}, size={path.stat().st_size / (1024**2):.1f} MiB", flush=True)


def _meta_record(filename: str, dataset: str, train_length: int, total_length: int) -> dict:
    return {
        "file_name": filename,
        "trend": "", "seasonal": "", "stationary": "", "pattern": "", "shifting": "",
        "dataset_name": dataset,
        "train_lens": str(train_length),
        "time_steps": "",
        "if_univariate": "FALSE",
        # Use the exact same size tag as the six existing named datasets.
        "size": "mult_new_new",
        "type_value": "",
        "total_len": str(total_length),
        "train/total": format(train_length / total_length, ".10g"),
    }


def _prepare_psm(root: Path, output: Path) -> dict:
    folder = root / "PSM"
    train = pd.read_csv(folder / "train.csv")
    test = pd.read_csv(folder / "test.csv")
    lab = pd.read_csv(folder / "test_label.csv")
    timestamp_col = "timestamp_(min)"
    if timestamp_col not in train or timestamp_col not in test:
        raise ValueError("PSM: expected timestamp_(min) column in train/test")
    features_train = train.drop(columns=[timestamp_col])
    features_test = test.drop(columns=[timestamp_col])
    if list(features_train.columns) != list(features_test.columns):
        raise ValueError("PSM: training and test feature columns/order differ")
    expected_features = [f"feature_{i}" for i in range(25)]
    if list(features_train.columns) != expected_features:
        raise ValueError(f"PSM: expected feature_0 through feature_24 in order; got {list(features_train.columns)}")
    if len(lab) != len(features_test):
        raise ValueError("PSM: test labels length differs from test length")
    label_cols = [c for c in lab.columns if c.lower().strip() in ("label", "labels", "test_label", "anomaly")]
    label_col = label_cols[0] if len(label_cols) == 1 else lab.columns[-1]
    if timestamp_col in lab:
        if not np.array_equal(lab[timestamp_col].to_numpy(), test[timestamp_col].to_numpy()):
            raise ValueError("PSM: label timestamps differ from test timestamps")
    test_labels = _binary_labels(pd.to_numeric(lab[label_col], errors="raise").to_numpy(), "PSM labels")
    train_features = _verify_features(features_train, "PSM train", fill_nan=True)
    test_features = _verify_features(features_test, "PSM test", fill_nan=True)
    channel_names = [f"channel{i + 1}" for i in range(25)]
    train_features.columns = channel_names
    test_features.columns = channel_names
    data = pd.concat([train_features, test_features], ignore_index=True)
    labels = np.concatenate([np.zeros(len(train_features), dtype=np.int8), test_labels])
    filename = "PSM.csv"
    _write_long(output / filename, data, labels, len(train_features), test_labels)
    return _meta_record(filename, "PSM", len(train_features), len(data))


def _prepare_smd(root: Path, output: Path) -> list[dict]:
    folder = root / "SMD"
    sources = sorted((folder / "train").glob("machine-*.txt"))
    if len(sources) != 28:
        raise ValueError(f"SMD: expected 28 machine files, found {len(sources)}")
    records = []
    for path in sources:
        train = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
        test_path = folder / "test" / path.name
        label_path = folder / "test_label" / path.name
        test = np.loadtxt(test_path, delimiter=",", dtype=np.float64, ndmin=2)
        test_labels = _binary_labels(np.loadtxt(label_path, delimiter=","), str(label_path))
        if train.ndim != 2 or test.ndim != 2 or train.shape[1] != 38 or test.shape[1] != 38:
            raise ValueError(f"SMD {path.name}: expected arrays shaped [length, 38]")
        if len(test) != len(test_labels):
            raise ValueError(f"SMD {path.name}: test/label lengths differ")
        names = [f"channel{i+1}" for i in range(38)]
        # No time-series crosses a machine boundary. Each machine is a separate CSV.
        train_frame = _verify_features(pd.DataFrame(train, columns=names), f"SMD {path.name} train")
        test_frame = _verify_features(pd.DataFrame(test, columns=names), f"SMD {path.name} test")
        data = pd.concat([train_frame, test_frame], ignore_index=True)
        labels = np.concatenate([np.zeros(len(train_frame), dtype=np.int8), test_labels])
        filename = f"SMD_{path.stem}.csv"
        _write_long(output / filename, data, labels, len(train_frame), test_labels)
        records.append(_meta_record(filename, "SMD", len(train_frame), len(data)))
    return records


def _update_meta(meta_path: Path, records: list[dict]):
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing repository metadata: {meta_path}")
    line_ending = "\r\n" if b"\r\n" in meta_path.read_bytes() else "\n"
    with meta_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        fields = reader.fieldnames
        existing = list(reader)
    if fields is None:
        raise ValueError("Empty metadata")
    required = set(records[0])
    if not required.issubset(set(fields)):
        raise ValueError(f"Metadata missing expected columns: {required - set(fields)}")
    records_by_name = {record["file_name"]: record for record in records}
    filenames = set(records_by_name)
    existing_names = {record["file_name"] for record in existing}
    updated = [records_by_name.get(record["file_name"], record) for record in existing]
    updated.extend(record for record in records if record["file_name"] not in existing_names)
    backup_path = meta_path.with_suffix(meta_path.suffix + ".backup")
    if not backup_path.exists():
        shutil.copy2(meta_path, backup_path)
    temporary = meta_path.with_suffix(meta_path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore", lineterminator=line_ending)
        writer.writeheader()
        writer.writerows(updated)
    with temporary.open("r", encoding="utf-8", newline="") as fp:
        actual = list(csv.DictReader(fp))
    actual_by_name = {row["file_name"]: row for row in actual}
    if len(actual) != len(updated):
        raise ValueError("Metadata verification failed: row count changed")
    for record in records:
        if any(actual_by_name.get(record["file_name"], {}).get(key) != value
               for key, value in record.items()):
            raise ValueError(f"Metadata verification failed for {record['file_name']}")
    original = {row["file_name"]: row for row in existing if row["file_name"] not in filenames}
    if any(actual_by_name.get(name) != row for name, row in original.items()):
        raise ValueError("Metadata verification failed for an existing dataset")
    temporary.replace(meta_path)
    print(f"[CHECK] metadata: {len(records)} train_lens entries match generated files; "
          f"{len(original)} existing entries unchanged", flush=True)
    print(f"[OK] metadata updated: {len(records)} entries ({meta_path}), backup: {backup_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", type=Path, help="Time-series repository root")
    parser.add_argument("--dataset", choices=("both", "psm", "smd"), default="both")
    args = parser.parse_args()
    base = args.project_root.resolve() / "dataset" / "anomaly_detect"
    root, out, meta = base / "raw", base / "data", base / "DETECT_META.csv"
    records = []
    if args.dataset in ("both", "psm"):
        records.append(_prepare_psm(root, out))
    if args.dataset in ("both", "smd"):
        records.extend(_prepare_smd(root, out))
    _update_meta(meta, records)
    print("[DONE] Converted to date,data,cols format, preserving official train/test boundaries.")
    print("[NOTE] Original datasets are retained in raw/. Converted large CSVs may need Git LFS or external artifact hosting.")


if __name__ == "__main__":
    main()
