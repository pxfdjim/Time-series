# -*- coding: utf-8 -*-
import base64
import pickle
import time
import traceback
from typing import List, Any

import numpy as np
import pandas as pd
import torch
import re

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

            single_series_results_list = []
            for ratio, predict_label in predict_labels.items():
                predict_label_before = predict_label.copy()
                another_before = another.copy()
                remaining_length = len(actual_label) - len(predict_label)
                remaining_length_another = len(actual_label) - len(another)
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
                    another = np.pad(
                        another,
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
                    another,
                    remaining_length,
                    remaining_length_another,
                )

                single_series_results, log_info = self.evaluator.evaluate_with_log(
                    actual=actual_label.astype(float),
                    predicted=predict_label.astype(float),
                    another=another.astype(float),
                )
                print(single_series_results)

                inference_data = [predict_label, another]
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

            single_series_results_list = []
            for ratio, predict_label in predict_labels.items():
                predict_label_before = predict_label.copy()
                another_before = another.copy()
                remaining_length = len(actual_label) - len(predict_label)
                remaining_length_another = len(actual_label) - len(another)
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
                    another = np.pad(
                        another,
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
                    another,
                    remaining_length,
                    remaining_length_another,
                )

                single_series_results, log_info = self.evaluator.evaluate_with_log(
                    actual=actual_label.astype(float),
                    predicted=predict_label.astype(float),
                    another=another.astype(float),
                )
                print(single_series_results)

                inference_data = [predict_label, another]
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

            single_series_results_list = []
            for ratio, predict_label in predict_labels.items():
                predict_label_before = predict_label.copy()
                another_before = another.copy()
                remaining_length = len(actual_label) - len(predict_label)
                remaining_length_another = len(actual_label) - len(another)
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
                    another = np.pad(
                        another,
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
                    another,
                    remaining_length,
                    remaining_length_another,
                )

                single_series_results, log_info = self.evaluator.evaluate_with_log(
                    actual=actual_label.astype(float),
                    predicted=predict_label.astype(float),
                    another=another.astype(float),
                )
                print(single_series_results)

                inference_data = [predict_label, another]
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
        text = DataPool().get_pool().get_text(text_name)
        data = data.reset_index(drop=True)
        text = text.reset_index(drop=True)

        train_length = int(
            DataPool().get_pool().get_series_meta_info(series_name)["train_lens"].item()
        )
        train_time, test_time = split_before(data, train_length)
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
