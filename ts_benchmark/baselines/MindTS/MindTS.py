
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler
import torch.nn.functional as F
from ts_benchmark.baselines.MindTS.models.MindTS_model import MINDTSModel
from ts_benchmark.baselines.utils import anomaly_detection_data_provider, anomaly_detection_multi_data_provider, anomaly_detection_timeMMD_data_provider
from ts_benchmark.baselines.utils import train_val_split
from ts_benchmark.baselines.MindTS.utils.tools import adjust_learning_rate
from torch import optim
from tqdm.auto import tqdm
import time
import gc
import os
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_MINDTS_BASED_HYPER_PARAMS = {
    "top_k": 3,
    "enc_in": 4,
    "dec_in": 4,
    "c_out": 4,
    "e_layers": 1,
    "d_layers": 1,
    "d_model": 256,
    "d_ff": 256,
    "embed": "timeF",
    "freq": "h",
    "lradj": "type1",
    "moving_avg": 25,
    "num_kernels": 6,
    "factor": 1,
    "n_heads": 8,
    "seg_len": 6,
    "win_size": 72,
    "activation": "gelu",
    "output_attention": 0,
    "patch_len": 6,
    "patch_size": 6,
    "stride": 6,
    "dropout": 0.1,
    "batch_size": 16,
    "lr": 0.0001,
    "num_epochs": 3,
    "num_workers": 0,
    "loss": "MSE",
    "itr": 1,
    "distil": True,
    "patience": 3,
    "task_name": "anomaly_detection",
    "p_hidden_dims": [128, 128],
    "p_hidden_layers": 2,
    "mem_dim": 32,
    "anomaly_ratio": [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
    "conv_kernel": [12, 16],
    "use_norm": True,
    "parallel_strategy": "DP",
    "num_epochs": 3,
    "mask_ratio": 0.5,
    "r": 0.5,
    "lamda": 1.0,
    "enc_in_time": 1,
    "lamda1": 1.0,
    "lamda2": 1.0,
    "stl_period": None,
    "stl_weight": 0.001,
    "dataset_description": "A generic time-series dataset.",
    "main_device": None,
    "llm_device": None,
    "llm_device_map": "balanced_low_0",
    "llm_prompt_batch_size": 32,
    "use_de_stationary_cross_view": False,
    "use_information_condenser": True,
    "align_loss_type": "contrastive",
    "align_detach_target": True,
    "align_logvar_min": -6.0,
    "align_logvar_max": 2.0,
    "recon_loss_type": "mse",
    "recon_logvar_min": -6.0,
    "recon_logvar_max": 2.0,
}

def clip_loss(logits_per_time, logits_per_text):
    loss_device = logits_per_time.device
    labels = torch.arange(logits_per_time.shape[1], device=loss_device).long()
    total_loss = torch.zeros((), device=loss_device, dtype=logits_per_time.dtype)
    for i in range(logits_per_time.shape[0]):
        total_loss += (F.cross_entropy(logits_per_time[i], labels) + F.cross_entropy(logits_per_text[i], labels)) / 2
    return total_loss / logits_per_time.shape[0]

def gaussian_nll_reconstruction_loss(target, mu, logvar, logvar_min=-6.0, logvar_max=2.0):
    logvar = torch.clamp(logvar, logvar_min, logvar_max)
    inv_var = torch.exp(-logvar)
    nll = 0.5 * (logvar + (target - mu).pow(2) * inv_var)
    return nll.mean()

def reconstruction_loss(target, mu, logvar, config, criterion):
    if config.recon_loss_type == "mse":
        return criterion(mu, target)
    if config.recon_loss_type == "gaussian_nll":
        return gaussian_nll_reconstruction_loss(
            target,
            mu,
            logvar,
            config.recon_logvar_min,
            config.recon_logvar_max,
        )
    raise ValueError(f"Unknown recon_loss_type: {config.recon_loss_type}")

def reconstruction_anomaly_score(target, mu, logvar, config):
    if config.recon_loss_type == "mse":
        return torch.mean((target - mu) ** 2, dim=-1)
    if config.recon_loss_type == "gaussian_nll":
        logvar = torch.clamp(logvar, config.recon_logvar_min, config.recon_logvar_max)
        score = logvar + (target - mu).pow(2) * torch.exp(-logvar)
        return torch.mean(score, dim=-1)
    raise ValueError(f"Unknown recon_loss_type: {config.recon_loss_type}")

def Bottleneck_loss(total_mask, r, lamda):
    compress_loss, connect_loss = 0., 0.
    for i in range(total_mask.shape[0]):
        temp = total_mask[i]
        compress_loss += (temp * torch.log(temp/(r + 1e-6) + 1e-6) + (1-temp) * torch.log((1-temp)/(1-r+1e-6) + 1e-6)).mean()
        shift1 = temp[1:,:]
        shift2 = temp[:-1,:]
        connect_loss += torch.sum((shift1 - shift2).norm(p=2)) / shift1.flatten().shape[0]
    connect_loss /= total_mask.shape[0]
    compress_loss /= total_mask.shape[0]

    mask_loss = compress_loss + lamda * connect_loss
    return mask_loss


class MINDTSConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_MINDTS_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.parallel_strategy not in [None, 'DP']:
            raise ValueError("Invalid value for parallel_strategy. Supported values are 'DP' and None.")

    @property
    def pred_len(self):
        # return self.seq_len
        return 0

    @property
    def learning_rate(self):
        return self.lr

    @property
    def model_name(self):
        return "MindTS"


class MindTS:
    def __init__(self, **kwargs):
        super(MindTS, self).__init__()
        self.config = MINDTSConfig(**kwargs)
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.seq_len = self.config.win_size
        self.lamda1 = self.config.lamda1
        self.lamda2 = self.config.lamda2
        self.stl_weight = self.config.stl_weight
        self.training_logs = []
        self.check_point = None
        self.best_valid_loss = None
        self.best_checkpoint_epoch = None
        self.shape_log_path = None
        self._shape_log_events = set()

    def configure_shape_logging(self, series_name, log_dir="result/shape_logs"):
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(series_name)).strip("_")
        if not safe_name:
            safe_name = "unknown_series"
        os.makedirs(log_dir, exist_ok=True)
        self.shape_log_path = os.path.join(log_dir, f"{safe_name}.shape.log")
        setattr(self.config, "shape_log_path", self.shape_log_path)
        setattr(self.config, "shape_log_dataset", series_name)
        self._shape_log_events = set()
        with open(self.shape_log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"Shape log for dataset: {series_name}\n")
            log_file.write("=" * 80 + "\n")

    def _write_shape_log(self, event, lines, once=True):
        if self.shape_log_path is None:
            return
        if once and event in self._shape_log_events:
            return
        self._shape_log_events.add(event)
        with open(self.shape_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n[{event}]\n")
            for line in lines:
                log_file.write(f"{line}\n")

    def _shape_of(self, value):
        if hasattr(value, "shape"):
            return tuple(value.shape)
        return None

    def _log_model_batch_shapes(self, event, inputs, outputs, scores=None):
        lines = [
            f"seq_len/window length: {self.config.seq_len}",
            f"batch_size config: {self.config.batch_size}",
        ]
        for name, value, meaning in inputs:
            lines.append(f"{name} shape: {self._shape_of(value)}; meaning: {meaning}")
        for name, value, meaning in outputs:
            lines.append(f"{name} shape: {self._shape_of(value)}; meaning: {meaning}")
        if scores is not None:
            for name, value, meaning in scores:
                lines.append(f"{name} shape: {self._shape_of(value)}; meaning: {meaning}")
        self._write_shape_log(event, lines)

    def _base_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _get_main_device(self):
        model = self._base_model()
        return getattr(model, "main_device", self.device)

    def _uses_manual_device_split(self):
        return bool(getattr(self._base_model(), "manual_device_split", False))

    def _prepare_model_for_device(self):
        if self._uses_manual_device_split():
            self._base_model().prepare_devices()
            self.device = self._get_main_device()
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

    def _load_best_checkpoint(self):
        checkpoint = self.check_point
        if checkpoint is None and hasattr(self, "early_stopping"):
            checkpoint = self.early_stopping.check_point
        if checkpoint is None:
            return
        if isinstance(self.model, nn.DataParallel):
            self.model.load_state_dict(checkpoint)
        elif hasattr(self._base_model(), "load_trainable_state_dict"):
            self._base_model().load_trainable_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)

    def _save_current_checkpoint(self):
        if hasattr(self._base_model(), "trainable_state_dict"):
            self.check_point = self._base_model().trainable_state_dict()
        else:
            self.check_point = {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            }

    def _save_best_checkpoint_if_needed(self, valid_loss, epoch):
        if not np.isfinite(valid_loss):
            return
        if self.best_valid_loss is None or valid_loss < self.best_valid_loss:
            self.best_valid_loss = float(valid_loss)
            self.best_checkpoint_epoch = int(epoch)
            self._save_current_checkpoint()
            print(
                f"Best validation loss updated at epoch {epoch}: {valid_loss:.6f}. "
                "Saving checkpoint."
            )

    def _reset_training_logs(self):
        self.training_logs = []
        self.check_point = None
        self.best_valid_loss = None
        self.best_checkpoint_epoch = None

    def _record_training_log(self, epoch, train_loss, valid_loss, learning_rate, epoch_time):
        self.training_logs.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "valid_loss": float(valid_loss),
                "learning_rate": float(learning_rate),
                "epoch_time_sec": float(epoch_time),
            }
        )

    def get_training_log(self):
        return json.dumps(self.training_logs, sort_keys=True)

    def _bottleneck_loss(self, total_mask):
        if not getattr(self.config, "use_information_condenser", True):
            return torch.zeros((), device=total_mask.device, dtype=total_mask.dtype)
        return Bottleneck_loss(total_mask, self.config.r, self.config.lamda)

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by model.

        :return: An empty dictionary indicating that model does not require additional hyperparameters.
        """
        return {}

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        try:
            freq = pd.infer_freq(train_data.index)
        except Exception as ignore:
            freq = 'S'
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

    def detect_validate(self, valid_data_loader, criterion):
        config = self.config
        total_loss = []
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for input, _ in valid_data_loader:
                input = input.to(device)

                outputs = self.model(input)

                outputs = outputs[:, :, :]

                outputs = outputs.detach().cpu()
                true = input.detach().cpu()

                loss = criterion(outputs, true).detach().cpu().numpy()

                total_loss.append(loss)

        total_loss = np.mean(total_loss)
        self.model.train()
        return total_loss

    def detect_multi_validate(self, valid_data_loader, criterion, epoch=None):
        config = self.config
        total_loss = []
        self.model.eval()
        description = "Validate" if epoch is None else f"Validate epoch {epoch}/{config.num_epochs}"

        with torch.no_grad():
            valid_progress = tqdm(
                valid_data_loader,
                desc=description,
                unit="batch",
                leave=True,
                dynamic_ncols=True,
            )
            for batch_x_time, batch_input_ids, batch_attention_mask, _, trend_stl, season_stl, residual_stl in valid_progress:
                batch_x_time = batch_x_time.float().to(self._get_main_device())
                trend_stl = trend_stl.float().to(self._get_main_device())
                season_stl = season_stl.float().to(self._get_main_device())
                residual_stl = residual_stl.float().to(self._get_main_device())
                outputs_mu, outputs_logvar, logits_per_time, logits_per_text, total_mask, loss_stl, align_loss = self.model(
                    batch_x_time,
                    batch_input_ids,
                    batch_attention_mask,
                    trend_stl,
                    season_stl,
                    residual_stl,
                )
                f_dim = -1 if self.config.enc_in == 1 else 0
                outputs_mu = outputs_mu[:, :, f_dim:]
                outputs_logvar = outputs_logvar[:, :, f_dim:]

                # Reconstruction loss
                loss1 = reconstruction_loss(
                    batch_x_time,
                    outputs_mu,
                    outputs_logvar,
                    self.config,
                    criterion,
                ).detach().cpu().numpy()

                # Comparison Loss
                loss2 = align_loss.detach().cpu().numpy()

                # Bottleneck loss
                loss3 = self._bottleneck_loss(total_mask).detach().cpu().numpy()

                loss = loss1 + self.lamda1*loss2 + self.lamda2*loss3 + self.stl_weight*loss_stl.detach().cpu().numpy()
                total_loss.append(loss)
                valid_progress.set_postfix(loss=f"{float(loss):.6f}")

        total_loss = np.mean(total_loss)
        self.model.train()
        return total_loss

    def detect_fit(self, train_data: pd.DataFrame, train_label: pd.DataFrame):
        self.detect_hyper_param_tune(train_data)
        setattr(self.config, "task_name", "anomaly_detection")
        self.model = MINDTSModel(self.config)

        device_ids = np.arange(torch.cuda.device_count()).tolist()
        if len(device_ids) > 1 and self.config.parallel_strategy == "DP" and not self.model.manual_device_split:
            self.model = nn.DataParallel(self.model, device_ids=device_ids)

        config = self.config
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        self.scaler.fit(train_data_value.values)

        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )

        valid_data = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )

        self.valid_data_loader = anomaly_detection_data_provider(
            valid_data,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="val",
        )

        self.train_data_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="train",
        )

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

        self._prepare_model_for_device()
        self._reset_training_logs()
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        for epoch in range(config.num_epochs):
            epoch_start_time = time.time()
            train_loss_values = []
            epoch_lr = optimizer.param_groups[0]["lr"]
            self.model.train()
            for i, (input, target) in enumerate(self.train_data_loader):
                optimizer.zero_grad()
                input = input.float().to(self.device)
                outputs = self.model(input)
                self._log_model_batch_shapes(
                    "detect_fit.train_first_batch",
                    [
                        (
                            "input",
                            input,
                            "[batch, seq_len, channels]; normalized sliding-window time-series batch",
                        ),
                    ],
                    [
                        (
                            "outputs",
                            outputs,
                            "[batch, seq_len, channels]; reconstructed time-series window",
                        ),
                    ],
                )
                outputs = outputs[:, :, :]
                loss = criterion(outputs, input)
                train_loss_values.append(loss.detach().cpu().item())
                loss.backward()
                optimizer.step()
            valid_loss = self.detect_validate(self.valid_data_loader, criterion)
            train_loss = float(np.mean(train_loss_values)) if train_loss_values else np.nan
            self._record_training_log(epoch + 1, train_loss, valid_loss, epoch_lr, time.time() - epoch_start_time)
            print(f"\tepoch: {epoch + 1}, train_loss: {train_loss:.6f}, valid_loss: {valid_loss:.6f}")
            self._save_best_checkpoint_if_needed(valid_loss, epoch + 1)

            adjust_learning_rate(optimizer, epoch + 1, config)
        if self.check_point is None:
            self._save_current_checkpoint()


    def detect_multi_fit(self, train_data: pd.DataFrame, train_text: pd.DataFrame, train_label: pd.DataFrame):
        self.detect_hyper_param_tune(train_data)
        setattr(self.config, "task_name", "anomaly_detection")
        config = self.config
        if config.stl_period is None:
            raise ValueError("stl_period must be set for TEMPO-style STL supervision")
        print("Initializing MindTS model and DeepSeek encoder...", flush=True)
        self.model = MINDTSModel(self.config)
        print("MindTS model initialized.", flush=True)
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        train_data_text, valid_text = train_val_split(train_text, 0.8, None)
        self.scaler.fit(train_data_value.values)

        device_ids = np.arange(torch.cuda.device_count()).tolist()
        if len(device_ids) > 1 and self.config.parallel_strategy == "DP" and not self.model.manual_device_split:
            self.model = nn.DataParallel(self.model, device_ids=device_ids)

        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )

        valid_data = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )

        train_data_text = pd.DataFrame(
            train_data_text,
            columns=train_data_text.columns,
            index=train_data_text.index,
        )

        valid_text = pd.DataFrame(
            valid_text,
            columns=valid_text.columns,
            index=valid_text.index,
        )

        print("Preparing validation STL windows...", flush=True)
        self.valid_data_loader = anomaly_detection_multi_data_provider(
            valid_data,
            valid_text,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="val",
            stl_period=config.stl_period,
        )
        print(f"Validation loader ready: {len(self.valid_data_loader)} batches.", flush=True)

        print("Preparing training STL windows...", flush=True)
        self.train_data_loader = anomaly_detection_multi_data_provider(
            train_data_value,
            train_data_text,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="train",
            stl_period=config.stl_period,
        )
        print(f"Training loader ready: {len(self.train_data_loader)} batches.", flush=True)

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.lr,
        )

        self._prepare_model_for_device()
        self._reset_training_logs()
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Trainable parameters: {total_params:,}", flush=True)

        for epoch in range(config.num_epochs):
            epoch_start_time = time.time()
            train_loss_values = []
            epoch_lr = optimizer.param_groups[0]["lr"]
            self.model.train()
            print(f"Epoch {epoch + 1}/{config.num_epochs} started.", flush=True)
            train_progress = tqdm(
                self.train_data_loader,
                desc=f"Train epoch {epoch + 1}/{config.num_epochs}",
                unit="batch",
                leave=True,
                dynamic_ncols=True,
            )
            for batch_x_time, batch_input_ids, batch_attention_mask, batch_y, trend_stl, season_stl, residual_stl in train_progress:
                optimizer.zero_grad()
                batch_x_time = batch_x_time.float().to(self._get_main_device())
                trend_stl = trend_stl.float().to(self._get_main_device())
                season_stl = season_stl.float().to(self._get_main_device())
                residual_stl = residual_stl.float().to(self._get_main_device())
                outputs_mu, outputs_logvar, logits_per_time, logits_per_text, total_mask, loss_stl, align_loss = self.model(
                    batch_x_time,
                    batch_input_ids,
                    batch_attention_mask,
                    trend_stl,
                    season_stl,
                    residual_stl,
                )
                self._log_model_batch_shapes(
                    "detect_multi_fit.train_first_batch",
                    [
                        (
                            "batch_x_time",
                            batch_x_time,
                            "[batch, seq_len, channels]; normalized sliding-window time-series batch",
                        ),
                        (
                            "trend_stl",
                            trend_stl,
                            "[batch, seq_len, channels]; STL trend supervision aligned to each window",
                        ),
                        (
                            "season_stl",
                            season_stl,
                            "[batch, seq_len, channels]; STL seasonal supervision aligned to each window",
                        ),
                        (
                            "residual_stl",
                            residual_stl,
                            "[batch, seq_len, channels]; STL residual supervision aligned to each window",
                        ),
                    ],
                    [
                        (
                            "outputs_mu",
                            outputs_mu,
                            "[batch, seq_len, channels]; reconstructed mean time-series window",
                        ),
                        (
                            "outputs_logvar",
                            outputs_logvar,
                            "[batch, seq_len, channels]; reconstructed log variance",
                        ),
                        (
                            "logits_per_time",
                            logits_per_time,
                            "[batch*channels, patches, patches]; time-to-semantic similarity logits",
                        ),
                        (
                            "logits_per_text",
                            logits_per_text,
                            "[batch*channels, patches, patches]; semantic-to-time similarity logits",
                        ),
                        (
                            "total_mask",
                            total_mask,
                            "[batch*channels, patches, 1]; information condenser keep probability",
                        ),
                    ],
                )
                f_dim = -1 if self.config.enc_in == 1 else 0
                outputs_mu = outputs_mu[:, :, f_dim:]
                outputs_logvar = outputs_logvar[:, :, f_dim:]

                # Reconstruction loss
                loss1 = reconstruction_loss(
                    batch_x_time,
                    outputs_mu,
                    outputs_logvar,
                    self.config,
                    criterion,
                )

                # Comparison Loss
                loss2 = align_loss

                # Bottleneck loss
                loss3 = self._bottleneck_loss(total_mask)

                loss = loss1 + self.lamda1*loss2 + self.lamda2*loss3 + self.stl_weight*loss_stl
                loss_value = loss.detach().cpu().item()
                train_loss_values.append(loss_value)
                train_progress.set_postfix(loss=f"{loss_value:.6f}", lr=f"{epoch_lr:.2e}")
                loss.backward()
                optimizer.step()
            valid_loss = self.detect_multi_validate(self.valid_data_loader, criterion, epoch + 1)
            train_loss = float(np.mean(train_loss_values)) if train_loss_values else np.nan
            self._record_training_log(epoch + 1, train_loss, valid_loss, epoch_lr, time.time() - epoch_start_time)
            print(f"Epoch {epoch + 1}/{config.num_epochs} finished: train_loss={train_loss:.6f}, valid_loss={valid_loss:.6f}", flush=True)
            self._save_best_checkpoint_if_needed(valid_loss, epoch + 1)

            adjust_learning_rate(optimizer, epoch + 1, config)
        if self.check_point is None:
            self._save_current_checkpoint()

    @torch.no_grad()
    def detect_score(self, test: pd.DataFrame) -> np.ndarray:
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        self._load_best_checkpoint()

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        self._prepare_model_for_device()
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        attens_energy = []
        test_labels = []

        for batch_x, batch_y in tqdm(
            self.thre_loader,
            desc="Score windows",
            unit="batch",
            leave=True,
            dynamic_ncols=True,
        ):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        return test_energy, test_energy

    @torch.no_grad()
    def detect_multi_score(self, test_data: pd.DataFrame, test_text: pd.DataFrame) -> np.ndarray:
        test_data = pd.DataFrame(
            self.scaler.transform(test_data.values), columns=test_data.columns, index=test_data.index
        )
        test_text = pd.DataFrame(
            test_text.values, columns=test_text.columns, index=test_text.index
        )
        self._load_best_checkpoint()

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        self.thre_loader = anomaly_detection_multi_data_provider(
            test_data,
            test_text,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        self._prepare_model_for_device()
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        attens_energy = []
        test_labels = []
        for batch_x_time, batch_input_ids, batch_attention_mask, batch_y, _, _, _ in tqdm(
            self.thre_loader,
            desc="Score windows",
            unit="batch",
            leave=True,
            dynamic_ncols=True,
        ):
            batch_x_time = batch_x_time.float().to(self._get_main_device())
            # reconstruction
            outputs_mu, outputs_logvar, logits_per_time, logits_per_text, total_mask, _, _ = self.model(batch_x_time, batch_input_ids, batch_attention_mask)
            # criterion
            score = reconstruction_anomaly_score(batch_x_time, outputs_mu, outputs_logvar, self.config)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        return test_energy, test_energy

    @torch.no_grad()
    def detect_label(self, test: pd.DataFrame) -> np.ndarray:
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        self._load_best_checkpoint()

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        self.test_data_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="test",
        )

        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        attens_energy = []

        self._prepare_model_for_device()
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        for batch_x, batch_y in tqdm(
            self.train_data_loader,
            desc="Train threshold energy",
            unit="batch",
            leave=True,
            dynamic_ncols=True,
        ):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            self._log_model_batch_shapes(
                "detect_label.train_threshold_first_batch",
                [
                    (
                        "batch_x",
                        batch_x,
                        "[batch, seq_len, channels]; training windows used to estimate threshold energy",
                    ),
                    (
                        "batch_y",
                        batch_y,
                        "[batch, seq_len, channels]; loader reconstruction target placeholder",
                    ),
                ],
                [
                    (
                        "outputs",
                        outputs,
                        "[batch, seq_len, channels]; reconstructed training windows",
                    ),
                ],
                [
                    (
                        "score",
                        score,
                        "[batch, seq_len]; per-time-step reconstruction error averaged over channels",
                    ),
                ],
            )
            score = score.detach().cpu().numpy()
            attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []

        for batch_x, batch_y in tqdm(
            self.test_data_loader,
            desc="Test energy",
            unit="batch",
            leave=True,
            dynamic_ncols=True,
        ):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            self._log_model_batch_shapes(
                "detect_label.test_first_batch",
                [
                    (
                        "batch_x",
                        batch_x,
                        "[batch, seq_len, channels]; test windows used for combined threshold calibration",
                    ),
                    (
                        "batch_y",
                        batch_y,
                        "[batch, seq_len, channels]; test window labels/targets from loader",
                    ),
                ],
                [
                    (
                        "outputs",
                        outputs,
                        "[batch, seq_len, channels]; reconstructed test windows",
                    ),
                ],
                [
                    (
                        "score",
                        score,
                        "[batch, seq_len]; per-time-step reconstruction error averaged over channels",
                    ),
                ],
            )
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        attens_energy = []
        test_labels = []

        for batch_x, batch_y in tqdm(
            self.thre_loader,
            desc="Threshold energy",
            unit="batch",
            leave=True,
            dynamic_ncols=True,
        ):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            self._log_model_batch_shapes(
                "detect_label.threshold_first_batch",
                [
                    (
                        "batch_x",
                        batch_x,
                        "[batch, seq_len, channels]; threshold-mode windows that produce final anomaly scores",
                    ),
                    (
                        "batch_y",
                        batch_y,
                        "[batch, seq_len, channels]; threshold-mode labels/targets from loader",
                    ),
                ],
                [
                    (
                        "outputs",
                        outputs,
                        "[batch, seq_len, channels]; reconstructed threshold-mode windows",
                    ),
                ],
                [
                    (
                        "score",
                        score,
                        "[batch, seq_len]; per-time-step reconstruction error averaged over channels",
                    ),
                ],
            )
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        preds = {}
        for ratio in self.config.anomaly_ratio:
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)

        return preds, test_energy

    @torch.no_grad()
    def detect_multi_label(self, test_data: pd.DataFrame, test_text: pd.DataFrame) -> np.ndarray:
        test_data = pd.DataFrame(
            self.scaler.transform(test_data.values), columns=test_data.columns, index=test_data.index
        )

        test_text = pd.DataFrame(
            test_text.values, columns=test_text.columns, index=test_text.index
        )
        self._load_best_checkpoint()

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        self.test_data_loader = anomaly_detection_multi_data_provider(
            test_data,
            test_text,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="test",
        )

        self.thre_loader = anomaly_detection_multi_data_provider(
            test_data,
            test_text,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        attens_energy = []

        self._prepare_model_for_device()
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        for batch_x_time, batch_input_ids, batch_attention_mask, batch_y, _, _, _ in tqdm(
            self.train_data_loader,
            desc="Train threshold energy",
            unit="batch",
            leave=True,
            dynamic_ncols=True,
        ):
            batch_x_time = batch_x_time.float().to(self._get_main_device())
            # reconstruction
            outputs_mu, outputs_logvar, logits_per_time, logits_per_text, total_mask, _, _ = self.model(batch_x_time, batch_input_ids, batch_attention_mask)
            # criterion
            score = reconstruction_anomaly_score(batch_x_time, outputs_mu, outputs_logvar, self.config)
            self._log_model_batch_shapes(
                "detect_multi_label.train_threshold_first_batch",
                [
                    (
                        "batch_x_time",
                        batch_x_time,
                        "[batch, seq_len, channels]; training windows used to estimate threshold energy",
                    ),
                    (
                        "batch_y",
                        batch_y,
                        "[batch, seq_len, channels]; loader reconstruction target placeholder",
                    ),
                ],
                [
                    (
                        "outputs_mu",
                        outputs_mu,
                        "[batch, seq_len, channels]; reconstructed mean training windows",
                    ),
                    (
                        "outputs_logvar",
                        outputs_logvar,
                        "[batch, seq_len, channels]; reconstructed log variance",
                    ),
                    (
                        "logits_per_time",
                        logits_per_time,
                        "[batch*channels, patches, patches]; time-to-semantic similarity logits",
                    ),
                    (
                        "logits_per_text",
                        logits_per_text,
                        "[batch*channels, patches, patches]; semantic-to-time similarity logits",
                    ),
                    (
                        "total_mask",
                        total_mask,
                        "[batch*channels, patches, 1]; information condenser keep probability",
                    ),
                ],
                [
                    (
                        "score",
                        score,
                        "[batch, seq_len]; per-time-step reconstruction error averaged over channels",
                    ),
                ],
            )
            score = score.detach().cpu().numpy()
            attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for batch_x_time, batch_input_ids, batch_attention_mask, batch_y, _, _, _ in tqdm(
            self.test_data_loader,
            desc="Test energy",
            unit="batch",
            leave=True,
            dynamic_ncols=True,
        ):
            batch_x_time = batch_x_time.float().to(self._get_main_device())
            # reconstruction
            outputs_mu, outputs_logvar, logits_per_time, logits_per_text, total_mask, _, _ = self.model(batch_x_time, batch_input_ids, batch_attention_mask)
            # criterion
            score = reconstruction_anomaly_score(batch_x_time, outputs_mu, outputs_logvar, self.config)
            self._log_model_batch_shapes(
                "detect_multi_label.test_first_batch",
                [
                    (
                        "batch_x_time",
                        batch_x_time,
                        "[batch, seq_len, channels]; test windows used for combined threshold calibration",
                    ),
                    (
                        "batch_y",
                        batch_y,
                        "[batch, seq_len, channels]; test window labels/targets from loader",
                    ),
                ],
                [
                    (
                        "outputs_mu",
                        outputs_mu,
                        "[batch, seq_len, channels]; reconstructed mean test windows",
                    ),
                    (
                        "outputs_logvar",
                        outputs_logvar,
                        "[batch, seq_len, channels]; reconstructed log variance",
                    ),
                    (
                        "logits_per_time",
                        logits_per_time,
                        "[batch*channels, patches, patches]; time-to-semantic similarity logits",
                    ),
                    (
                        "logits_per_text",
                        logits_per_text,
                        "[batch*channels, patches, patches]; semantic-to-time similarity logits",
                    ),
                    (
                        "total_mask",
                        total_mask,
                        "[batch*channels, patches, 1]; information condenser keep probability",
                    ),
                ],
                [
                    (
                        "score",
                        score,
                        "[batch, seq_len]; per-time-step reconstruction error averaged over channels",
                    ),
                ],
            )
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        attens_energy = []
        test_labels = []
        for batch_x_time, batch_input_ids, batch_attention_mask, batch_y, _, _, _ in tqdm(
            self.thre_loader,
            desc="Threshold energy",
            unit="batch",
            leave=True,
            dynamic_ncols=True,
        ):
            batch_x_time = batch_x_time.float().to(self._get_main_device())
            # reconstruction
            outputs_mu, outputs_logvar, logits_per_time, logits_per_text, total_mask, _, _ = self.model(batch_x_time, batch_input_ids, batch_attention_mask)
            # criterion
            score = reconstruction_anomaly_score(batch_x_time, outputs_mu, outputs_logvar, self.config)
            self._log_model_batch_shapes(
                "detect_multi_label.threshold_first_batch",
                [
                    (
                        "batch_x_time",
                        batch_x_time,
                        "[batch, seq_len, channels]; threshold-mode windows that produce final anomaly scores",
                    ),
                    (
                        "batch_y",
                        batch_y,
                        "[batch, seq_len, channels]; threshold-mode labels/targets from loader",
                    ),
                ],
                [
                    (
                        "outputs_mu",
                        outputs_mu,
                        "[batch, seq_len, channels]; reconstructed mean threshold-mode windows",
                    ),
                    (
                        "outputs_logvar",
                        outputs_logvar,
                        "[batch, seq_len, channels]; reconstructed log variance",
                    ),
                    (
                        "logits_per_time",
                        logits_per_time,
                        "[batch*channels, patches, patches]; time-to-semantic similarity logits",
                    ),
                    (
                        "logits_per_text",
                        logits_per_text,
                        "[batch*channels, patches, patches]; semantic-to-time similarity logits",
                    ),
                    (
                        "total_mask",
                        total_mask,
                        "[batch*channels, patches, 1]; information condenser keep probability",
                    ),
                ],
                [
                    (
                        "score",
                        score,
                        "[batch, seq_len]; per-time-step reconstruction error averaged over channels",
                    ),
                ],
            )
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        preds = {}
        for ratio in self.config.anomaly_ratio:
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)

        return preds, test_energy

    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        return self.model_name
