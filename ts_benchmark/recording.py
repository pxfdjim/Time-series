# -*- coding: utf-8 -*-

from __future__ import absolute_import

import io
import itertools
import logging
import os
import os.path
import json
from io import StringIO
from typing import List, Optional

import pandas as pd
from pandas.errors import ParserError

from ts_benchmark.common.constant import ROOT_PATH
from ts_benchmark.utils.compress import (
    get_compress_method_from_ext,
    decompress,
    compress,
    get_compress_file_ext,
)
from ts_benchmark.utils.get_file_name import get_unique_file_suffix
from ts_benchmark.utils.get_file_name import get_model_config_tag

logger = logging.getLogger(__name__)
TRAINING_LOG_FIELD = "training_log"
THRESHOLD_FIELD = "typical_anomaly_ratio"
THRESHOLD_METRIC_FIELDS = ["affiliation_f", "VUS_PR", "VUS_ROC"]


def read_record_file(fn: str) -> pd.DataFrame:
    """
    Reads a single record file.

    The format of the file is currently determined by the extension name.

    :param fn: Path to the record file.
    :return: Benchmarking records in DataFrame format.
    """
    ext = os.path.splitext(fn)[1]
    compress_method = get_compress_method_from_ext(ext)
    if compress_method is None:
        return pd.read_csv(fn)
    else:
        with open(fn, "rb") as fh:
            data = fh.read()
        data = decompress(data, method=compress_method)
        ret = []
        for k, v in data.items():
            ret.append(pd.read_csv(StringIO(v.decode("utf8"))))
        return pd.concat(ret, axis=0)


def write_record_file(
    result_df: pd.DataFrame,
    file_path: str,
    compress_method: Optional[str] = None,
) -> str:
    """
    Write to a single record file.

    :param result_df: Benchmarking records in DataFrame format.
    :param file_path: Path to the record file to save.
    :param compress_method: The format used to compress the record file, if None is given,
        no compression is applied.
    :return: Path to the record file written.
    """
    if compress_method is not None:
        buf = io.StringIO()
        result_df.to_csv(buf, index=False)
        write_data = compress(
            {os.path.basename(file_path): buf.getvalue()}, method=compress_method
        )
        file_path = f"{file_path}.{get_compress_file_ext(compress_method)}"

        with open(file_path, "wb") as fh:
            fh.write(write_data)
    else:
        result_df.to_csv(file_path, index=False)

    return file_path


def _extract_training_log(result_df: pd.DataFrame) -> pd.DataFrame:
    if TRAINING_LOG_FIELD not in result_df.columns:
        return pd.DataFrame()

    records = []
    seen = set()
    for _, row in result_df.iterrows():
        raw_log = row.get(TRAINING_LOG_FIELD)
        if not isinstance(raw_log, str) or not raw_log:
            continue
        key = (row.get("model_name"), row.get("model_params"), row.get("file_name"), raw_log)
        if key in seen:
            continue
        seen.add(key)
        try:
            epoch_records = json.loads(raw_log)
        except json.JSONDecodeError:
            continue
        if not isinstance(epoch_records, list):
            continue
        for epoch_record in epoch_records:
            if not isinstance(epoch_record, dict):
                continue
            cur_record = {
                "model_name": row.get("model_name"),
                "model_params": row.get("model_params"),
                "file_name": row.get("file_name"),
            }
            cur_record.update(epoch_record)
            records.append(cur_record)
    return pd.DataFrame(records)


def _format_training_log_text(training_log_df: pd.DataFrame) -> str:
    lines = ["MindTS training log", ""]
    group_columns = ["model_name", "model_params", "file_name"]
    metric_columns = [
        column for column in training_log_df.columns if column not in group_columns
    ]

    for group_values, group_df in training_log_df.groupby(group_columns, dropna=False):
        model_name, model_params, file_name = group_values
        lines.extend(
            [
                f"model_name: {model_name}",
                f"file_name: {file_name}",
                f"model_params: {model_params}",
                "",
                group_df[metric_columns].to_string(index=False),
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def _write_training_log_file(result_df: pd.DataFrame, result_path: str, record_filename: str) -> Optional[str]:
    training_log_df = _extract_training_log(result_df)
    if training_log_df.empty:
        return None
    base_filename = os.path.splitext(record_filename)[0]
    training_log_path = os.path.join(result_path, f"{base_filename}.training_log.txt")
    with open(training_log_path, "w", encoding="utf-8") as fh:
        fh.write(_format_training_log_text(training_log_df))
    return training_log_path


def _extract_threshold_metrics(result_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [THRESHOLD_FIELD] + THRESHOLD_METRIC_FIELDS
    if any(column not in result_df.columns for column in required_columns):
        return pd.DataFrame()

    group_columns = [
        column
        for column in ["model_name", "model_params", "file_name"]
        if column in result_df.columns
    ]
    columns = group_columns + required_columns
    threshold_df = result_df[columns].copy()
    for column in required_columns:
        threshold_df[column] = pd.to_numeric(threshold_df[column], errors="coerce")
    return threshold_df.dropna(subset=[THRESHOLD_FIELD])


def _format_threshold_metrics_text(threshold_df: pd.DataFrame) -> str:
    lines = ["MindTS threshold metrics", ""]
    group_columns = [
        column
        for column in ["model_name", "model_params", "file_name"]
        if column in threshold_df.columns
    ]
    table_columns = [THRESHOLD_FIELD] + THRESHOLD_METRIC_FIELDS

    if group_columns:
        grouped = threshold_df.groupby(group_columns, dropna=False)
    else:
        grouped = [((), threshold_df)]

    for group_values, group_df in grouped:
        if group_columns:
            if len(group_columns) == 1:
                group_values = (group_values,)
            for column, value in zip(group_columns, group_values):
                lines.append(f"{column}: {value}")
            lines.append("")

        group_df = group_df.sort_values(THRESHOLD_FIELD)
        display_df = group_df[table_columns].rename(
            columns={
                THRESHOLD_FIELD: "threshold",
                "affiliation_f": "Aff-F",
                "VUS_PR": "V-PR",
                "VUS_ROC": "V-ROC",
            }
        )
        lines.extend(
            [
                "Each threshold:",
                display_df.to_string(index=False),
                "",
            ]
        )

        mean_values = group_df[THRESHOLD_METRIC_FIELDS].mean(skipna=True)
        lines.extend(
            [
                "Mean across thresholds:",
                f"Aff-F: {mean_values.get('affiliation_f')}",
                f"V-PR: {mean_values.get('VUS_PR')}",
                f"V-ROC: {mean_values.get('VUS_ROC')}",
                "",
            ]
        )

        aff_f = group_df["affiliation_f"]
        if aff_f.notna().any():
            best_idx = aff_f.idxmax()
            best_row = group_df.loc[best_idx]
            lines.extend(
                [
                    "Best threshold by Aff-F:",
                    f"threshold: {best_row.get(THRESHOLD_FIELD)}",
                    f"Aff-F: {best_row.get('affiliation_f')}",
                    f"V-PR: {best_row.get('VUS_PR')}",
                    f"V-ROC: {best_row.get('VUS_ROC')}",
                    "",
                ]
            )

    return "\n".join(lines).rstrip() + "\n"


def _write_threshold_metrics_file(result_df: pd.DataFrame, result_path: str, record_filename: str) -> Optional[str]:
    threshold_df = _extract_threshold_metrics(result_df)
    if threshold_df.empty:
        return None
    base_filename = os.path.splitext(record_filename)[0]
    threshold_path = os.path.join(result_path, f"{base_filename}.threshold_metrics.txt")
    with open(threshold_path, "w", encoding="utf-8") as fh:
        fh.write(_format_threshold_metrics_text(threshold_df))
    return threshold_path


def _get_record_descriptor(result_df: pd.DataFrame) -> str:
    if "model_params" not in result_df.columns or result_df.empty:
        return ""
    return get_model_config_tag(result_df["model_params"].iloc[0])


def load_record_data(
    record_files: List[str], drop_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Loads benchmarking records from multiple record files.

    :param record_files: The list of paths to the record files. Each item in the list can either
        be the path to a directory or a file. If it is a path to a directory, then all record files
        in the directory are loaded; Otherwise, the file specified by the path is loaded.
    :param drop_columns: The columns to drop during loading.
        This parameter is mainly used to save memory.
    :return: The loaded benchmarking records in DataFrame format.
    """
    record_files = itertools.chain.from_iterable(
        [
            [fn] if not os.path.isdir(fn) else find_record_files(fn)
            for fn in record_files
        ]
    )

    ret = []
    for fn in record_files:
        logger.info("loading log file %s", fn)
        try:
            cur_record = read_record_file(fn)
            if drop_columns:
                cur_record = cur_record.drop(columns=drop_columns, errors="ignore")
            ret.append(cur_record)
        except (FileNotFoundError, PermissionError, KeyError, ParserError):
            # TODO: it is ugly to identify log files by artifact columns...
            logger.info("unrecognized log file format, skipping %s...", fn)
    return pd.concat(ret, axis=0)


def find_record_files(directory: str) -> List[str]:
    """
    Finds records files in a directory.

    :param directory: The path to the directory.
    :return: The list of file paths to the record files that are found in the give directory.
    """
    record_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # TODO: this is a temporary solution, any good methods to identify a log file?
            if (
                file.endswith(".training_log.csv")
                or file.endswith(".training_log.txt")
                or file.endswith(".threshold_metrics.txt")
            ):
                continue
            if file.endswith(".csv") or file.endswith(".tar.gz"):
                record_files.append(os.path.join(root, file))
    return record_files


def save_log(
    result_df: pd.DataFrame,
    save_path,
    file_prefix: str,
    compress_method: Optional[str] = None,
) -> str:
    """
    Save log data.

    Save the evaluation results, model hyperparameters, model evaluation configuration, and model name to a log file.

    :param result_df: Benchmarking records in DataFrame format.
    :param save_path: Path to the directory where the records are saved.
    :param file_prefix: Prefix of the file name to save the records.
    :param compress_method: The compression method for the output file.
    :return: The path to the output file.
    """
    if result_df["log_info"].any():
        error_itr = filter(None, result_df["log_info"])
        for error in itertools.islice(error_itr, 3):
            logger.info(error)
        if any(error_itr):
            logger.info(
                "-------------More error messages can be found in the record files!-------------"
            )

    if save_path is not None:
        result_path = (
            os.path.join(ROOT_PATH, "result", save_path)
            if not os.path.isabs(save_path)
            else save_path
        )
    else:
        result_path = os.path.join(ROOT_PATH, "result")
    os.makedirs(result_path, exist_ok=True)

    record_filename = file_prefix + get_unique_file_suffix(_get_record_descriptor(result_df))
    file_path = os.path.join(result_path, record_filename)

    _write_training_log_file(result_df, result_path, record_filename)
    _write_threshold_metrics_file(result_df, result_path, record_filename)
    result_df_to_write = result_df.drop(columns=[TRAINING_LOG_FIELD], errors="ignore")
    return write_record_file(result_df_to_write, file_path, compress_method)
