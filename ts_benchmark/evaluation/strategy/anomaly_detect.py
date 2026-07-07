# -*- coding: utf-8 -*-
import base64
import json
import pickle
import time
import traceback
from typing import List, Any

import numpy as np
import pandas as pd
import torch
import re

from ts_benchmark.common.constant import ROOT_PATH
from ts_benchmark.data.data_pool import DataPool
from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.metrics import classification_metrics_label
from ts_benchmark.evaluation.metrics import classification_metrics_score
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.strategy import Strategy
from ts_benchmark.models import ModelFactory
from ts_benchmark.utils.data_processing import split_before
from ts_benchmark.utils.random_utils import fix_random_seed
import os

TRAINING_LOG_FIELD = "training_log"


class AnomalyDetect(Strategy):
    """
    异常检测类，用于在时间序列数据上执行异常检测。
    """

    def __init__(self, strategy_config: dict, evaluator: Evaluator):
        """
        初始化子类实例。

        :param strategy_config: 模型评估配置。
        """
        super().__init__(strategy_config, evaluator)
        self.model = None
        self.data_lens = None

    def _visual_export_root(self):
        export_dir = self.strategy_config.get("visual_export_dir") or os.environ.get(
            "MINDTS_VIS_EXPORT_DIR"
        )
        if not export_dir:
            return None
        if os.path.isabs(export_dir):
            return export_dir
        return os.path.join(ROOT_PATH, export_dir)

    def _should_save_checkpoint(self):
        value = self.strategy_config.get(
            "save_checkpoint",
            os.environ.get("MINDTS_SAVE_CHECKPOINT", "false"),
        )
        if isinstance(value, bool):
            return value
        return str(value).lower() in {"1", "true", "yes", "y", "on"}

    def _safe_name(self, value):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
        return safe or "unknown"

    def _series_export_dir(self, series_name, model_factory=None):
        root = self._visual_export_root()
        if root is None:
            return None
        model_name = self._safe_name(getattr(model_factory, "model_name", "model"))
        series = self._safe_name(series_name).replace(".csv", "")
        export_dir = os.path.join(root, model_name, series)
        os.makedirs(export_dir, exist_ok=True)
        return export_dir

    def _align_to_length(self, values, target_length, fill_value=0):
        arr = np.asarray(values).reshape(-1)
        if arr.shape[0] < target_length:
            return np.pad(
                arr,
                (0, target_length - arr.shape[0]),
                mode="constant",
                constant_values=fill_value,
            )
        if arr.shape[0] > target_length:
            return arr[:target_length]
        return arr.copy()

    def _model_artifacts(self):
        if self.model is None:
            return {}
        return getattr(self.model, "last_detection_artifacts", {}) or {}

    def _export_checkpoint(self, series_name, model_factory):
        if not self._should_save_checkpoint():
            return
        export_dir = self._series_export_dir(series_name, model_factory)
        if export_dir is None or self.model is None:
            return
        checkpoint = getattr(self.model, "check_point", None)
        if checkpoint is None:
            return
        payload = {
            "series_name": series_name,
            "model_name": getattr(model_factory, "model_name", None),
            "model_hyper_params": getattr(model_factory, "model_hyper_params", {}),
            "best_valid_loss": getattr(self.model, "best_valid_loss", None),
            "best_checkpoint_epoch": getattr(self.model, "best_checkpoint_epoch", None),
            "trainable_state_dict": checkpoint,
        }
        scaler = getattr(self.model, "scaler", None)
        if scaler is not None and hasattr(scaler, "mean_"):
            payload["scaler_mean"] = scaler.mean_
            payload["scaler_scale"] = scaler.scale_
        torch.save(payload, os.path.join(export_dir, "checkpoint_trainable.pt"))

    def _simple_point_metrics(self, actual_label, predict_label):
        actual = np.asarray(actual_label).reshape(-1).astype(bool)
        pred = np.asarray(predict_label).reshape(-1).astype(bool)
        tp = int(np.logical_and(actual, pred).sum())
        fp = int(np.logical_and(~actual, pred).sum())
        fn = int(np.logical_and(actual, ~pred).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def _export_visual_artifacts(
        self,
        series_name,
        model_factory,
        test_data,
        test_label,
        predict_labels,
        score_raw,
    ):
        export_dir = self._series_export_dir(series_name, model_factory)
        if export_dir is None:
            return

        actual_label = np.asarray(test_label).reshape(-1).astype(float)
        target_length = len(actual_label)
        score_raw = np.asarray(score_raw).reshape(-1).astype(float)
        score = self._align_to_length(score_raw, target_length, fill_value=0).astype(float)

        ratio_values = list(predict_labels.keys())
        ratio_names = [str(ratio) for ratio in ratio_values]
        aligned_predictions = []
        raw_prediction_lengths = []
        summary_rows = []
        for ratio in ratio_values:
            raw_pred = np.asarray(predict_labels[ratio]).reshape(-1).astype(int)
            aligned_pred = self._align_to_length(raw_pred, target_length, fill_value=0).astype(int)
            aligned_predictions.append(aligned_pred)
            raw_prediction_lengths.append(len(raw_pred))
            row = {
                "series_name": series_name,
                "threshold_ratio": ratio,
                "raw_prediction_length": len(raw_pred),
                "aligned_length": target_length,
                "score_raw_length": len(score_raw),
                "actual_anomaly_points": int(actual_label.sum()),
                "predicted_anomaly_points": int(aligned_pred.sum()),
            }
            row.update(self._simple_point_metrics(actual_label, aligned_pred))
            summary_rows.append(row)

        pred_matrix = (
            np.vstack(aligned_predictions)
            if aligned_predictions
            else np.empty((0, target_length), dtype=int)
        )
        test_values = test_data.to_numpy(dtype=float, copy=True)
        test_columns = np.asarray(test_data.columns.astype(str), dtype=str)
        test_index = np.asarray(test_data.index.astype(str), dtype=str)

        artifacts = self._model_artifacts()
        intermediate = artifacts.get("intermediate", {})
        npz_payload = {
            "test_values": test_values,
            "test_columns": test_columns,
            "test_index": test_index,
            "actual_label": actual_label,
            "score": score,
            "score_raw": score_raw,
            "ratios": np.asarray(ratio_names, dtype=str),
            "predictions": pred_matrix,
            "raw_prediction_lengths": np.asarray(raw_prediction_lengths, dtype=int),
        }
        intermediate_shapes = {}
        for key, value in intermediate.items():
            arr = np.asarray(value)
            npz_payload[f"intermediate_{key}"] = arr
            intermediate_shapes[key] = list(arr.shape)
        np.savez_compressed(os.path.join(export_dir, "anomaly_trace.npz"), **npz_payload)

        trace_df = pd.DataFrame({"time_index": test_index, "label": actual_label, "score": score})
        for col_idx, col_name in enumerate(test_columns):
            trace_df[f"value_{col_name}"] = test_values[:, col_idx]
        for ratio_name, aligned_pred in zip(ratio_names, aligned_predictions):
            trace_df[f"pred_T{ratio_name}"] = aligned_pred
        trace_df.to_csv(os.path.join(export_dir, "anomaly_trace.csv"), index=False)
        pd.DataFrame(summary_rows).to_csv(os.path.join(export_dir, "threshold_summary.csv"), index=False)

        thresholds = artifacts.get("thresholds", {})
        metadata = {
            "series_name": series_name,
            "model_name": getattr(model_factory, "model_name", None),
            "model_hyper_params": getattr(model_factory, "model_hyper_params", {}),
            "strategy_config": self.strategy_config,
            "test_length": target_length,
            "score_raw_length": len(score_raw),
            "ratios": ratio_names,
            "thresholds": {str(k): float(v) for k, v in thresholds.items()},
            "best_valid_loss": getattr(self.model, "best_valid_loss", None),
            "best_checkpoint_epoch": getattr(self.model, "best_checkpoint_epoch", None),
            "intermediate_shapes": intermediate_shapes,
        }
        with open(os.path.join(export_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True, default=str)

    def _configure_model_shape_logging(self, series_name: str):
        if self.model is None:
            return
        if hasattr(self.model, "configure_shape_logging"):
            self.model.configure_shape_logging(series_name)

    def _write_shape_log(self, event: str, lines: List[str]):
        if self.model is not None and getattr(self.model, "shape_log_path", None):
            log_path = self.model.shape_log_path
        else:
            os.makedirs("result/shape_logs", exist_ok=True)
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(event)).strip("_")
            log_path = os.path.join("result/shape_logs", f"{safe_name}.shape.log")
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n[{event}]\n")
            for line in lines:
                log_file.write(f"{line}\n")

    def _array_preview(self, values, limit=10):
        arr = np.asarray(values).reshape(-1)
        return arr[:limit].tolist()

    def _label_counts(self, values):
        arr = np.asarray(values).reshape(-1)
        if arr.size == 0:
            return {}
        unique, counts = np.unique(arr, return_counts=True)
        return {str(k): int(v) for k, v in zip(unique, counts)}

    def _log_metric_shapes(
        self,
        series_name,
        ratio,
        actual_label,
        predict_label_before,
        another_before,
        predict_label_after,
        another_after,
        remaining_length,
        remaining_length_another,
    ):
        self._write_shape_log(
            f"metric_inputs.ratio_{ratio}",
            [
                f"dataset: {series_name}",
                f"ratio: {ratio}",
                (
                    f"actual_label shape: {np.asarray(actual_label).shape}; "
                    "meaning: flattened ground-truth anomaly labels from test_label, one value per original timestamp"
                ),
                (
                    f"predict_label before padding shape: {np.asarray(predict_label_before).shape}; "
                    "meaning: binary anomaly predictions produced by thresholding reconstruction energy"
                ),
                (
                    f"another before padding shape: {np.asarray(another_before).shape}; "
                    "meaning: continuous anomaly score/reconstruction energy used by score-style metrics"
                ),
                f"remaining_length for predict_label: {remaining_length}",
                f"remaining_length for another: {remaining_length_another}",
                (
                    f"predict_label passed to evaluator shape: {np.asarray(predict_label_after).shape}; "
                    "meaning: binary predictions after alignment/padding to actual_label length"
                ),
                (
                    f"another passed to evaluator shape: {np.asarray(another_after).shape}; "
                    "meaning: continuous anomaly scores after alignment/padding to actual_label length"
                ),
                f"actual_label value counts: {self._label_counts(actual_label)}",
                f"predict_label value counts: {self._label_counts(predict_label_after)}",
                f"actual_label first values: {self._array_preview(actual_label)}",
                f"predict_label first values: {self._array_preview(predict_label_after)}",
                f"another first values: {self._array_preview(another_after)}",
            ],
        )

    def _get_training_log(self) -> str:
        if self.model is None or not hasattr(self.model, "get_training_log"):
            return ""
        return self.model.get_training_log()

    def execute(self, series_name: str, model_factory: ModelFactory) -> Any:
        """
        执行异常检测策略。

        :param series_name: 要执行异常检测的序列名称。
        :param model_factory: 模型对象的构造/工厂函数。
        :return: 评估结果。
        """
        fix_random_seed()

        model = model_factory()
        try:
            self.model = model
            self._configure_model_shape_logging(series_name)
            train_data, train_label, test_data, test_label = self.split_data(
                series_name
            )
            start_fit_time = time.time()
            if hasattr(model, "detect_fit"):
                self.model.detect_fit(train_data, train_label)  # 在训练数据上拟合模型
            else:
                self.model.fit(train_data, train_label)  # 在训练数据上拟合模型

            end_fit_time = time.time()
            self._export_checkpoint(series_name, model_factory)
            predict_labels, another = self.detect(test_data)

            # 模型打label保存到本地
            for ratio, labels in predict_labels.items():
                pr_label = pd.DataFrame(labels, columns=['Label'])
                folder_path = './Labels/MindTS/KR' + str(int(ratio)) + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                output_file = os.path.join(folder_path, 'test_labels.txt')
                np.savetxt(output_file, pr_label, fmt='%f')

            if not isinstance(predict_labels, dict):
                predict_labels = {"None": predict_labels}

            actual_label = test_label.to_numpy().flatten()
            end_inference_time = time.time()
            score_raw = np.asarray(another).reshape(-1).copy()
            self._export_visual_artifacts(
                series_name,
                model_factory,
                test_data,
                test_label,
                predict_labels,
                score_raw,
            )

            single_series_results_list = []
            for ratio, predict_label in predict_labels.items():
                predict_label_before = np.asarray(predict_label).reshape(-1).copy()
                another_before = score_raw.copy()
                predict_label = predict_label_before.copy()
                another_eval = score_raw.copy()
                remaining_length = len(actual_label) - len(predict_label)
                remaining_length_another = len(actual_label) - len(another_eval)
                print(remaining_length)
                # Pad the predict_label array with zeros at the end
                if remaining_length > 0:
                    predict_label = np.pad(
                        predict_label,
                        (0, remaining_length),
                        mode="constant",
                        constant_values=0,
                    )

                if remaining_length_another > 0:
                    another_eval = np.pad(
                        another_eval,
                        (0, remaining_length_another),
                        mode="constant",
                        constant_values=0,
                    )

                self._log_metric_shapes(
                    series_name,
                    ratio,
                    actual_label,
                    predict_label_before,
                    another_before,
                    predict_label,
                    another_eval,
                    remaining_length,
                    remaining_length_another,
                )

                single_series_results, log_info = self.evaluator.evaluate_with_log(
                    actual=actual_label.astype(float),
                    predicted=predict_label.astype(float),
                    another=another_eval.astype(float),
                )
                print(single_series_results)

                inference_data = [predict_label, another_eval]
                actual_data_pickle = pickle.dumps(test_label)
                actual_data_pickle = base64.b64encode(actual_data_pickle).decode("utf-8")

                inference_data_pickle = pickle.dumps(inference_data)
                inference_data_pickle = base64.b64encode(inference_data_pickle).decode(
                    "utf-8"
                )

                single_series_results += [
                    series_name,
                    ratio,
                    '',
                    '',
                    log_info,
                    self._get_training_log(),
                ]

                single_series_results_list.append(single_series_results)
        except Exception as e:
            # log = f"{traceback.format_exc()}\n{e}"
            log = f"The error series is: {series_name}\n{traceback.format_exc()}\n{e}"
            single_series_results_list = [self.get_default_result(
                **{FieldNames.LOG_INFO: log}
            )]
        return single_series_results_list


    def multi_execute(self, series_name: str, text_name: str, model_factory: ModelFactory) -> Any:
        """
        执行异常检测策略。

        :param series_name: 要执行异常检测的序列名称。
        :param model_factory: 模型对象的构造/工厂函数。
        :return: 评估结果。
        """
        fix_random_seed()

        model = model_factory()
        try:
            self.model = model
            self._configure_model_shape_logging(series_name)
            train_data, train_text, train_label, test_data, test_text, test_label = self.split_multi_data(
                series_name,
                text_name
            )
            start_fit_time = time.time()

            torch.cuda.empty_cache()

            torch.cuda.reset_peak_memory_stats()

            total_allocated = 0.0
            total_peak_allocated = 0.0

            self.model.detect_multi_fit(train_data, train_text, train_label)

            end_fit_time = time.time()
            self._export_checkpoint(series_name, model_factory)

            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                peak_allocated = torch.cuda.max_memory_allocated(i) / 1024**3

                total_allocated += allocated
                total_peak_allocated += peak_allocated

            predict_labels, another = self.multi_detect(test_data, test_text)

            if not isinstance(predict_labels, dict):
                predict_labels = {"None": predict_labels}

            if not isinstance(predict_labels, dict):
                predict_labels = {"None": predict_labels}

            actual_label = test_label.to_numpy().flatten()
            end_inference_time = time.time()
            score_raw = np.asarray(another).reshape(-1).copy()
            self._export_visual_artifacts(
                series_name,
                model_factory,
                test_data,
                test_label,
                predict_labels,
                score_raw,
            )

            single_series_results_list = []
            for ratio, predict_label in predict_labels.items():
                predict_label_before = np.asarray(predict_label).reshape(-1).copy()
                another_before = score_raw.copy()
                predict_label = predict_label_before.copy()
                another_eval = score_raw.copy()
                remaining_length = len(actual_label) - len(predict_label)
                remaining_length_another = len(actual_label) - len(another_eval)
                print(remaining_length)
                # Pad the predict_label array with zeros at the end
                if remaining_length > 0:
                    predict_label = np.pad(
                        predict_label,
                        (0, remaining_length),
                        mode="constant",
                        constant_values=0,
                    )

                if remaining_length_another > 0:
                    another_eval = np.pad(
                        another_eval,
                        (0, remaining_length_another),
                        mode="constant",
                        constant_values=0,
                    )

                self._log_metric_shapes(
                    series_name,
                    ratio,
                    actual_label,
                    predict_label_before,
                    another_before,
                    predict_label,
                    another_eval,
                    remaining_length,
                    remaining_length_another,
                )

                single_series_results, log_info = self.evaluator.evaluate_with_log(
                    actual=actual_label.astype(float),
                    predicted=predict_label.astype(float),
                    another=another_eval.astype(float),
                )
                print(single_series_results)

                inference_data = [predict_label, another_eval]
                actual_data_pickle = pickle.dumps(test_label)
                actual_data_pickle = base64.b64encode(actual_data_pickle).decode("utf-8")

                inference_data_pickle = pickle.dumps(inference_data)
                inference_data_pickle = base64.b64encode(inference_data_pickle).decode(
                    "utf-8"
                )

                single_series_results += [
                    series_name,
                    ratio,
                    '',
                    '',
                    log_info,
                    self._get_training_log(),
                ]

                single_series_results_list.append(single_series_results)
        except Exception as e:
            # log = f"{traceback.format_exc()}\n{e}"
            log = f"The error series is: {series_name}\n{traceback.format_exc()}\n{e}"
            single_series_results_list = [self.get_default_result(
                **{FieldNames.LOG_INFO: log}
            )]
        return single_series_results_list
    

    def mmd_execute(self, series_name: str, text_name: str, model_factory: ModelFactory) -> Any:
        """
        执行异常检测策略。

        :param series_name: 要执行异常检测的序列名称。
        :param model_factory: 模型对象的构造/工厂函数。
        :return: 评估结果。
        """
        fix_random_seed()

        model = model_factory()
        try:
            self.model = model
            self._configure_model_shape_logging(series_name)
            train_data, train_text, train_label, test_data, test_text, test_label = self.split_multi_data(
                series_name,
                text_name
            )
            start_fit_time = time.time()

            torch.cuda.empty_cache()
            self.model.detect_timeMMD_fit(train_data, train_text, train_label)

            end_fit_time = time.time()
            self._export_checkpoint(series_name, model_factory)
            predict_labels, another = self.mmd_detect(test_data, test_text)

            # 模型打label保存到本地
            for ratio, labels in predict_labels.items():
                pr_label = pd.DataFrame(labels, columns=['Label'])
                folder_path = './Labels/MindTS/KR' + str(int(ratio)) + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                output_file = os.path.join(folder_path, 'test_labels.txt')
                np.savetxt(output_file, pr_label, fmt='%f')

            if not isinstance(predict_labels, dict):
                predict_labels = {"None": predict_labels}

            if not isinstance(predict_labels, dict):
                predict_labels = {"None": predict_labels}

            actual_label = test_label.to_numpy().flatten()
            end_inference_time = time.time()
            score_raw = np.asarray(another).reshape(-1).copy()
            self._export_visual_artifacts(
                series_name,
                model_factory,
                test_data,
                test_label,
                predict_labels,
                score_raw,
            )

            single_series_results_list = []
            for ratio, predict_label in predict_labels.items():
                predict_label_before = np.asarray(predict_label).reshape(-1).copy()
                another_before = score_raw.copy()
                predict_label = predict_label_before.copy()
                another_eval = score_raw.copy()
                remaining_length = len(actual_label) - len(predict_label)
                remaining_length_another = len(actual_label) - len(another_eval)
                print(remaining_length)
                # Pad the predict_label array with zeros at the end
                if remaining_length > 0:
                    predict_label = np.pad(
                        predict_label,
                        (0, remaining_length),
                        mode="constant",
                        constant_values=0,
                    )

                if remaining_length_another > 0:
                    another_eval = np.pad(
                        another_eval,
                        (0, remaining_length_another),
                        mode="constant",
                        constant_values=0,
                    )

                self._log_metric_shapes(
                    series_name,
                    ratio,
                    actual_label,
                    predict_label_before,
                    another_before,
                    predict_label,
                    another_eval,
                    remaining_length,
                    remaining_length_another,
                )

                single_series_results, log_info = self.evaluator.evaluate_with_log(
                    actual=actual_label.astype(float),
                    predicted=predict_label.astype(float),
                    another=another_eval.astype(float),
                )
                print(single_series_results)

                inference_data = [predict_label, another_eval]
                actual_data_pickle = pickle.dumps(test_label)
                actual_data_pickle = base64.b64encode(actual_data_pickle).decode("utf-8")

                inference_data_pickle = pickle.dumps(inference_data)
                inference_data_pickle = base64.b64encode(inference_data_pickle).decode(
                    "utf-8"
                )

                single_series_results += [
                    series_name,
                    ratio,
                    '',
                    '',
                    log_info,
                    self._get_training_log(),
                ]

                single_series_results_list.append(single_series_results)
        except Exception as e:
            # log = f"{traceback.format_exc()}\n{e}"
            log = f"The error series is: {series_name}\n{traceback.format_exc()}\n{e}"
            single_series_results_list = [self.get_default_result(
                **{FieldNames.LOG_INFO: log}
            )]
        return single_series_results_list


    def split_data(self, data: str):
        raise NotImplementedError

    def detect(self, test_data: pd.DataFrame):
        raise NotImplementedError

    def multi_detect(self, test_data: pd.DataFrame, test_text: pd.DataFrame):
        raise NotImplementedError
    
    def mmd_detect(self, test_data: pd.DataFrame, test_text: pd.DataFrame):
        raise NotImplementedError

    @staticmethod
    def accepted_metrics():
        raise NotImplementedError

    @property
    def field_names(self) -> List[str]:
        return self.evaluator.metric_names + [
            FieldNames.FILE_NAME,
            FieldNames.ANOMALY_RATIO,
            FieldNames.ACTUAL_DATA,
            FieldNames.INFERENCE_DATA,
            FieldNames.LOG_INFO,
            TRAINING_LOG_FIELD,
        ]


class FixedDetectScore(AnomalyDetect):
    REQUIRED_FIELDS = ["train_test_split"]

    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        self.data_lens = len(data)
        train_length = int(self.strategy_config["train_test_split"] * self.data_lens)
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class FixedDetectLabel(AnomalyDetect):
    REQUIRED_FIELDS = ["train_test_split"]

    def split_data(self, series_name: str):
        data = DataPool().get_pool().get_series(series_name)
        self.data_lens = len(data)
        train_length = int(self.strategy_config["train_test_split"] * self.data_lens)
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_label(test_data)

    def multi_detect(self, test_data, test_text):
        return self.model.detect_multi_label(test_data, test_text)
    
    def mmd_detect(self, test_data, test_text):
        return self.model.detect_timeMMD_label(test_data, test_text)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__


class UnFixedDetectScore(AnomalyDetect):
    def split_data(self, series_name: str):
        data = DataPool().get_pool().get_series(series_name)
        data = data.reset_index(drop=True)
        train_length = int(
            DataPool().get_pool().get_series_meta_info(series_name)["train_lens"].item()
        )
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )

        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    def multi_detect(self, test_data, test_text):
        return self.model.detect_multi_score(test_data, test_text)
    
    def mmd_detect(self, test_data, test_text):
        return self.model.detect_timeMMD_score(test_data, test_text)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class UnFixedDetectLabel(AnomalyDetect):
    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        data = data.reset_index(drop=True)
        train_length = int(
            DataPool().get_pool().get_series_meta_info(series_name)["train_lens"].item()
        )
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def split_multi_data(self, series_name, text_name):
        data = DataPool().get_pool().get_series(series_name)
        data = data.reset_index(drop=True)

        train_length = int(
            DataPool().get_pool().get_series_meta_info(series_name)["train_lens"].item()
        )
        train_time, test_time = split_before(data, train_length)
        if text_name is None:
            train_text, test_text = None, None
        else:
            text = DataPool().get_pool().get_text(text_name)
            text = text.reset_index(drop=True)
            train_text, test_text = split_before(text, train_length)

        train_data, train_label = (
            train_time.loc[:, train_time.columns != "label"],
            train_time.loc[:, ["label"]],
        )
        test_data, test_label = (
            test_time.loc[:, test_time.columns != "label"],
            test_time.loc[:, ["label"]],
        )
        return train_data, train_text, train_label, test_data, test_text, test_label

    def detect(self, test_data):
        return self.model.detect_label(test_data)

    def multi_detect(self, test_data, test_text):
        return self.model.detect_multi_label(test_data, test_text)
    
    def mmd_detect(self, test_data, test_text):
        return self.model.detect_timeMMD_label(test_data, test_text)


    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__


class AllDetectScore(AnomalyDetect):
    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        train = data
        test = data
        train_data, train_label = train.loc[:, train.columns != "label"], None
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, None, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    def multi_detect(self, test_data, test_text):
        return self.model.detect_multi_score(test_data, test_text)
    
    def mmd_detect(self, test_data, test_text):
        return self.model.detect_timeMMD_score(test_data, test_text)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class AllDetectLabel(AnomalyDetect):
    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        train = data
        test = data
        train_data, train_label = train.loc[:, train.columns != "label"], None
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, None, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_label(test_data)

    def multi_detect(self, test_data, test_text):
        return self.model.detect_multi_score(test_data, test_text)
    
    def mmd_detect(self, test_data, test_text):
        return self.model.detect_timeMMD_score(test_data, test_text)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__
