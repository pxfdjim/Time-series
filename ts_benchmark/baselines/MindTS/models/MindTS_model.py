import torch
import time
import os
import importlib.util
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from ts_benchmark.baselines.MindTS.layers.Embed import WarriorsEmbedding, DataEmbedding_inverted
from ts_benchmark.baselines.MindTS.layers.Transformer_EncDec import Encoder, EncoderLayer
from ts_benchmark.baselines.MindTS.layers.SelfAttention_Family import FullAttention, AttentionLayer
from einops import rearrange
from transformers import AutoTokenizer, AutoModel, AutoConfig


DEEPSEEK_PATH = "DeepSeek"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_MEMORY_GIB = 1024 ** 3


def _resolve_device(device_name, fallback):
    if device_name is None:
        return fallback
    resolved = torch.device(device_name)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        visible_gpu_count = torch.cuda.device_count()
        if resolved.index is not None and resolved.index >= visible_gpu_count:
            raise ValueError(
                f"Requested device {resolved}, but only {visible_gpu_count} CUDA "
                "device(s) are visible. Check --gpus or CUDA_VISIBLE_DEVICES."
            )
    return resolved


def _has_accelerate():
    return importlib.util.find_spec("accelerate") is not None


def _gpu_max_memory(visible_gpu_count, main_gpu_index=0):
    if visible_gpu_count <= 0:
        return None

    main_fraction = float(os.environ.get("MINDTS_MAIN_GPU_LLM_FRACTION", "0.35"))
    llm_fraction = float(os.environ.get("MINDTS_LLM_GPU_MEMORY_FRACTION", "0.90"))
    max_memory = {}
    for gpu_index in range(visible_gpu_count):
        free_bytes, _ = torch.cuda.mem_get_info(gpu_index)
        fraction = main_fraction if gpu_index == main_gpu_index else llm_fraction
        memory_gib = max(1, int((free_bytes * fraction) // GPU_MEMORY_GIB))
        max_memory[gpu_index] = f"{memory_gib}GiB"
    max_memory["cpu"] = "64GiB"
    return max_memory


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class MovingAverage(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        padding = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, padding, 1)
        end = x[:, -1:, :].repeat(1, padding, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1, factor=1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, prompt, semantic_features):
        attn_output, _ = self.attention(prompt, semantic_features, semantic_features)
        x = self.norm1(prompt + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        out = self.norm2(x + self.dropout(ff_output))

        return out


class MultiTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(MultiTransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, llm_features, time_features_patch_mask):
        self_attn_output, _ = self.self_attn(llm_features, llm_features, llm_features)
        self_attn_output = self.dropout(self_attn_output)
        cross_attn_output, _ = self.cross_attn(query=time_features_patch_mask, key=self_attn_output, value=self_attn_output)
        cross_attn_output = self.dropout(cross_attn_output)
        x = self.norm1(time_features_patch_mask + cross_attn_output)
        ff_output = self.ffn(x)
        ff_output = self.dropout(ff_output)
        output = self.norm2(x + ff_output)

        return output


class MINDTSModel(nn.Module):
    def __init__(self, configs):
        super(MINDTSModel, self).__init__()
        default_device = device
        main_device_name = getattr(configs, "main_device", None)
        llm_device_name = getattr(configs, "llm_device", None)
        explicit_device_split = main_device_name is not None or llm_device_name is not None
        visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.manual_device_split = explicit_device_split or visible_gpu_count > 1
        if explicit_device_split:
            if main_device_name is None:
                main_device_name = "cuda:0" if visible_gpu_count else str(default_device)
            if llm_device_name is None:
                llm_device_name = main_device_name
        elif visible_gpu_count > 1:
            main_device_name = "cuda:0"
            llm_device_name = "cuda:1"
        self.main_device = _resolve_device(main_device_name, default_device)
        self.llm_device = _resolve_device(llm_device_name, self.main_device)
        self.visible_gpu_count = visible_gpu_count
        self.llm_device_map = getattr(configs, "llm_device_map", "balanced_low_0")
        self.llm_uses_device_map = False
        self.llm_prompt_batch_size = int(getattr(configs, "llm_prompt_batch_size", 32))
        self.llm_empty_cache_between_chunks = os.environ.get(
            "MINDTS_LLM_EMPTY_CACHE_BETWEEN_CHUNKS",
            "false",
        ).lower() == "true"
        self.device = self.main_device    # Device for trainable time-series modules
        self.configs = configs
        self.batch_size = configs.batch_size
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1    # Number of patches
        self.d_model = configs.d_model
        self.channel_time = configs.enc_in_time    # Number of input time channels
        self.mask_ratio = configs.mask_ratio   # Masking ratio for sequence
        self.align_loss_type = getattr(configs, "align_loss_type", "contrastive")# "text_gaussian_nll"、"symmetric_gaussian_kl"、"none"
        self.align_detach_target = getattr(configs, "align_detach_target", True)
        self.align_logvar_min = getattr(configs, "align_logvar_min", -6.0)
        self.align_logvar_max = getattr(configs, "align_logvar_max", 2.0)
        self.recon_loss_type = getattr(configs, "recon_loss_type", "mse")
        self.recon_logvar_min = getattr(configs, "recon_logvar_min", -6.0)
        self.recon_logvar_max = getattr(configs, "recon_logvar_max", 2.0)
        self.llm_model_path = getattr(configs, "llm_model_path", DEEPSEEK_PATH)
        default_llm_name = os.path.basename(str(self.llm_model_path).rstrip(os.sep)) or str(self.llm_model_path)
        self.llm_model_name = getattr(configs, "llm_model_name", default_llm_name)
        self.llm_layers = int(getattr(configs, "llm_layers", 6))
        self.llm_config = AutoConfig.from_pretrained(self.llm_model_path, trust_remote_code=True)
        self.llm_hidden_size = int(
            getattr(
                self.llm_config,
                "hidden_size",
                getattr(self.llm_config, "n_embd", 1536),
            )
        )
        allowed_align_losses = {
            "contrastive",
            "text_gaussian_nll",
            "symmetric_gaussian_kl",
            "none",
        }
        if self.align_loss_type not in allowed_align_losses:
            raise ValueError(
                f"Invalid align_loss_type={self.align_loss_type!r}. "
                f"Supported values are {sorted(allowed_align_losses)}."
            )
        self.shape_log_path = getattr(configs, "shape_log_path", None)
        self._shape_log_events = set()
        self.description = getattr(
            configs,
            "dataset_description",
            "A generic time-series dataset.",
        )

        # Embedding layers
        self.windows_data_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.proj_patch = nn.Linear(self.patch_num * configs.d_model, configs.seq_len, bias=True)
        self.proj_patch_logvar = nn.Linear(self.patch_num * configs.d_model, configs.seq_len, bias=True)
        self.prompt_proj_hidden = nn.Linear(self.llm_hidden_size, configs.d_model, bias=True)
        self.text_proj_hidden = nn.Linear(self.llm_hidden_size, configs.d_model, bias=True)
        self.proj_text = nn.Linear(1024 * configs.d_model, configs.d_model, bias=True)
        self.proj_prompt = nn.Linear(128 * configs.d_model, configs.d_model, bias=True)
        self.patch_embedding = WarriorsEmbedding(configs.d_model, self.patch_size, self.stride, self.stride, configs.dropout)
        self.component_fusion = nn.Linear(3 * configs.d_model, configs.d_model, bias=True)
        self.moving_avg = MovingAverage(configs.moving_avg)
        self.map_trend = nn.Linear(configs.seq_len, configs.seq_len)
        self.map_season = nn.Sequential(
            nn.Linear(configs.seq_len, 4 * configs.seq_len),
            nn.ReLU(),
            nn.Linear(4 * configs.seq_len, configs.seq_len),
        )

        # Time patch encoder with stacked attention layers
        self.time_patch_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                             output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )
        self.time_windows_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )
        self.layer = configs.e_layers    # Number of encoder layers
        self.layer_norm = nn.LayerNorm(configs.d_model)    # Layer normalization
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))    # Logit scaling factor
        self.top_k = self.patch_size    # Top-k value for selection
        self.d_ff = configs.d_ff    # Feedforward hidden dimension
        self.num_heads = configs.n_heads    # Number of attention heads

        # Load LLM config
        self.llm_config.num_hidden_layers = min(
            self.llm_layers,
            int(getattr(self.llm_config, "num_hidden_layers", self.llm_layers)),
        )
        self.llm_config.output_attentions = True
        self.llm_config.output_hidden_states = True
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.llm_config.pad_token_id = self.tokenizer.pad_token_id
        model_load_kwargs = {
            "trust_remote_code": True,
            "config": self.llm_config,
            "attn_implementation": "eager",
        }
        llm_device_map_mode = str(self.llm_device_map).lower()
        use_automatic_llm_device_map = (
            llm_device_map_mode not in {"none", "false", "off"}
            and visible_gpu_count > 1
            and not explicit_device_split
        )
        if use_automatic_llm_device_map:
            if _has_accelerate():
                main_gpu_index = self.main_device.index
                if main_gpu_index is None:
                    main_gpu_index = 0
                self.llm_uses_device_map = True
                model_load_kwargs.update(
                    {
                        "device_map": self.llm_device_map,
                        "max_memory": _gpu_max_memory(
                            visible_gpu_count,
                            main_gpu_index=main_gpu_index,
                        ),
                        "low_cpu_mem_usage": True,
                    }
                )
                print(
                    f"Loading {self.llm_model_name} with HuggingFace device_map={self.llm_device_map!r} "
                    f"across {visible_gpu_count} visible GPU(s).",
                    flush=True,
                )
            else:
                print(
                    f"accelerate is not installed; loading {self.llm_model_name} on a single "
                    "LLM device. Install accelerate to enable automatic LLM "
                    "layer sharding across multiple GPUs.",
                    flush=True,
                )
        self.model = AutoModel.from_pretrained(
            self.llm_model_path,
            **model_load_kwargs,
        )
        if len(self.tokenizer) > self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.transformer_block = TransformerBlock(
            self.d_model,
            self.num_heads,
            self.d_ff,
            factor=configs.factor,
        )
        self.multimodal_Transformer_Block = MultiTransformerBlock(self.d_model, self.num_heads, self.d_ff)
        if self.align_loss_type == "text_gaussian_nll":
            self.align_mu_head = nn.Linear(configs.d_model, configs.d_model)
            self.align_logvar_head = nn.Linear(configs.d_model, configs.d_model)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.register_buffer("role_prompt_embeddings", torch.empty(0), persistent=False)
        if self.manual_device_split:
            self.prepare_devices()

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

    def prepare_devices(self):
        for name, module in self.named_children():
            if name == "model":
                if not self.llm_uses_device_map:
                    module.to(self.llm_device)
            else:
                module.to(self.main_device)
        self.logit_scale.data = self.logit_scale.data.to(self.main_device)
        if self.logit_scale.grad is not None:
            self.logit_scale.grad = self.logit_scale.grad.to(self.main_device)
        self.device = self.main_device
        return self

    def trainable_state_dict(self):
        return {
            key: value.detach().cpu().clone()
            for key, value in self.state_dict().items()
            if not key.startswith("model.")
        }

    def load_trainable_state_dict(self, state_dict):
        if state_dict is None:
            return
        self.load_state_dict(state_dict, strict=False)
        if self.manual_device_split:
            self.prepare_devices()

    def _check_token_ids(self, input_ids):
        vocab_size = self.model.get_input_embeddings().num_embeddings
        max_token_id = int(input_ids.max().item())
        if max_token_id >= vocab_size:
            raise ValueError(f"input id out of range: {max_token_id} >= {vocab_size}")

    def _runtime_main_device(self, x_enc_time):
        return self.main_device if self.manual_device_split else x_enc_time.device

    def _runtime_llm_device(self):
        if self.llm_uses_device_map:
            return self.model.get_input_embeddings().weight.device
        if self.manual_device_split:
            return self.llm_device
        return next(self.model.parameters()).device

    def _encode_prompts_with_llm(self, prompts, main_device, max_length=128):
        chunk_size = max(1, self.llm_prompt_batch_size)
        embeddings = []
        for start in range(0, len(prompts), chunk_size):
            prompt_chunk = prompts[start:start + chunk_size]
            prompt_tokens = self.tokenizer(
                prompt_chunk,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = prompt_tokens["input_ids"].to(self._runtime_llm_device())
            attention_mask = prompt_tokens["attention_mask"].to(self._runtime_llm_device())
            self._check_token_ids(input_ids)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                embeddings.append(outputs.hidden_states[-1].detach().to(main_device))
            del input_ids, attention_mask, prompt_tokens, outputs
            if self.llm_empty_cache_between_chunks and torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(embeddings, dim=0)

    def random_masking(self, xb, mask_ratio):
        bs_nvars, L, d_model = xb.shape
        device = xb.device
        x = xb.clone()
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(bs_nvars, L, device=device)
        ids_shuffle = torch.argsort(noise, dim=1).to(device)
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(device)
        ids_keep = ids_shuffle[:, :len_keep]
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d_model))
        x_removed = torch.zeros(bs_nvars, L - len_keep, d_model, device=device)
        x_ = torch.cat([x_kept, x_removed], dim=1)
        x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, d_model))
        mask = torch.ones([bs_nvars, L], device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, x_kept, mask, ids_restore

    def _apply_patch_mask(self, x, mask):
        return x.masked_fill(mask.unsqueeze(-1).bool(), 0.0)

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        _, lags = torch.topk(corr, self.top_k, dim=-1)
        return lags

    def _normalize_component(self, x):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        return x / stdev

    def _contrastive_align_loss(self, logits_per_time, logits_per_text):
        labels = torch.arange(logits_per_time.shape[1], device=logits_per_time.device).long()
        total_loss = torch.zeros((), device=logits_per_time.device, dtype=logits_per_time.dtype)
        for i in range(logits_per_time.shape[0]):
            total_loss = total_loss + (
                F.cross_entropy(logits_per_time[i], labels)
                + F.cross_entropy(logits_per_text[i], labels)
            ) / 2
        return total_loss / logits_per_time.shape[0]

    def _text_gaussian_nll_align_loss(self, time_features, llm_features):
        target = time_features.detach() if self.align_detach_target else time_features
        mu_txt = self.align_mu_head(llm_features)
        logvar_txt = self.align_logvar_head(llm_features).clamp(
            self.align_logvar_min,
            self.align_logvar_max,
        )
        inv_var = torch.exp(-logvar_txt)
        nll = 0.5 * (logvar_txt + (target - mu_txt).pow(2) * inv_var)
        return nll.mean()

    def _symmetric_gaussian_kl_align_loss(self, time_features, llm_features):
        eps = 1e-5
        time_norm = F.layer_norm(time_features, (time_features.shape[-1],))
        llm_norm = F.layer_norm(llm_features, (llm_features.shape[-1],))

        mu_t = time_norm.mean(dim=1)
        mu_x = llm_norm.mean(dim=1)
        var_t = time_norm.var(dim=1, unbiased=False) + eps
        var_x = llm_norm.var(dim=1, unbiased=False) + eps
        logvar_t = torch.log(var_t).clamp(self.align_logvar_min, self.align_logvar_max)
        logvar_x = torch.log(var_x).clamp(self.align_logvar_min, self.align_logvar_max)
        var_t = torch.exp(logvar_t)
        var_x = torch.exp(logvar_x)

        kl_t_to_x = 0.5 * (
            logvar_x
            - logvar_t
            + (var_t + (mu_t - mu_x).pow(2)) / var_x
            - 1
        )
        kl_x_to_t = 0.5 * (
            logvar_t
            - logvar_x
            + (var_x + (mu_x - mu_t).pow(2)) / var_t
            - 1
        )
        return 0.5 * (kl_t_to_x + kl_x_to_t).mean()

    def _alignment_loss(self, time_features, llm_features, logits_per_time, logits_per_text):
        if self.align_loss_type == "contrastive":
            return self._contrastive_align_loss(logits_per_time, logits_per_text)
        if self.align_loss_type == "text_gaussian_nll":
            return self._text_gaussian_nll_align_loss(time_features, llm_features)
        if self.align_loss_type == "symmetric_gaussian_kl":
            return self._symmetric_gaussian_kl_align_loss(time_features, llm_features)
        return torch.zeros((), device=time_features.device, dtype=time_features.dtype)

    def _decompose_local(self, x):
        trend = self.moving_avg(x)
        trend = self.map_trend(trend.transpose(1, 2)).transpose(1, 2)
        season = x - trend
        season = self.map_season(season.transpose(1, 2)).transpose(1, 2)
        residual = x - trend - season
        return trend, season, residual

    def _get_role_prompt_embeddings(self, main_device):
        if self.role_prompt_embeddings.numel() == 0:
            dataset_context = self.description.strip()
            role_prompts = [
                f"Dataset context: {dataset_context} Reconstruct the time series using its trend component.",
                f"Dataset context: {dataset_context} Reconstruct the time series using its seasonal component.",
                f"Dataset context: {dataset_context} Reconstruct the time series using its residual component.",
            ]
            self.role_prompt_embeddings = self._encode_prompts_with_llm(
                role_prompts,
                main_device,
                max_length=128,
            )
        return self.prompt_proj_hidden(self.role_prompt_embeddings.to(main_device).float())

    def _encode_intrinsic_prompts(self, x_enc_time, main_device):
        x_enc_time = x_enc_time.permute(0, 2, 1).contiguous()
        x_enc_time = rearrange(x_enc_time, 'b c l -> (b c) l')
        x_enc_time = x_enc_time.unfold(1, self.patch_size, self.stride)

        min_values = torch.min(x_enc_time, dim=2)[0]
        max_values = torch.max(x_enc_time, dim=2)[0]
        medians = torch.median(x_enc_time, dim=2).values
        lags = self.calcute_lags(x_enc_time)
        trends = x_enc_time.diff(dim=2)
        prompt_list = []
        for b in range(x_enc_time.shape[0]):
            prompt = []
            for c in range(x_enc_time.shape[1]):
                min_values_str = str(min_values[b][c].tolist())
                max_values_str = str(max_values[b][c].tolist())
                median_values_str = str(medians[b][c].tolist())
                lags_values_str = str(lags[b][c].tolist())
                patch_num_middle = self.patch_num // 2
                first_half = trends[b][:patch_num_middle]
                second_half = trends[b][patch_num_middle:]
                first_half_mean = first_half.mean()
                second_half_mean = second_half.mean()
                first_half_std = first_half.std()
                second_half_std = second_half.std()
                if first_half_mean > 0 and second_half_mean < 0:
                    trend = 'first upward then downward'
                elif first_half_mean < 0 and second_half_mean > 0:
                    trend = 'first downward then upward'
                elif first_half_mean > 0 and second_half_mean > 0:
                    trend = 'upwarding'
                elif first_half_mean < 0 and second_half_mean < 0:
                    trend = 'downwarding'
                elif first_half_std < 0.01 and second_half_std < 0.01:
                    trend = 'balanced'
                else:
                    trend = 'uncertain'

                prompt_ = (
                    f"<|start_prompt|>Dataset description: {self.description}"
                    f"Task description: reconstruct the {str(self.seq_len)} steps given the previous {str(self.seq_len)} steps information; "
                    "Input statistics: "
                    f"min value {min_values_str}, "
                    f"max value {max_values_str}, "
                    f"median value {median_values_str}, "
                    f"the trend of input is {trend}, "
                    f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
                )
                prompt.append(prompt_)
            prompt_list.append(prompt)

        all_prompts = [prompt for batch in prompt_list for prompt in batch]
        embeddings = self._encode_prompts_with_llm(
            all_prompts,
            main_device,
            max_length=128,
        )

        batch_size_prompt = len(prompt_list[0])
        prompt_feature = embeddings.view(
            -1,
            batch_size_prompt,
            embeddings.size(1),
            embeddings.size(2),
        ).float()
        prompt_feature = self.prompt_proj_hidden(prompt_feature)
        prompt_feature = rearrange(
            prompt_feature,
            'v n m d -> v n (m d)',
            n=self.patch_num,
            m=128,
            d=self.d_model,
        )
        return self.proj_prompt(prompt_feature)

    def _encode_component(self, component, role_prompt, batch_channels):
        component = self._normalize_component(component)
        component_patch, _ = self.patch_embedding(component.permute(0, 2, 1))
        prompt = role_prompt.unsqueeze(0).expand(batch_channels, -1, -1)
        component_with_prompt = torch.cat([prompt, component_patch], dim=1)
        component_features, _ = self.time_windows_encoder(component_with_prompt)
        return component_features[:, prompt.shape[1]:, :]

    def _calculate_stl_loss(
        self,
        trend_local,
        season_local,
        residual_local,
        trend_stl,
        season_stl,
        residual_stl,
    ):
        if trend_stl is None or season_stl is None or residual_stl is None:
            return torch.zeros((), device=trend_local.device)
        trend_target = trend_stl.to(trend_local.device)
        season_target = season_stl.to(season_local.device)
        residual_target = residual_stl.to(residual_local.device)
        return (
            F.mse_loss(trend_local, trend_target)
            + F.mse_loss(season_local, season_target)
            + F.mse_loss(residual_local, residual_target)
        )


    def Multimodal_Time_Series(
        self,
        x_enc_time,
        trend_stl=None,
        season_stl=None,
        residual_stl=None,
    ):
        main_device = self._runtime_main_device(x_enc_time)
        x_enc_time = x_enc_time.to(main_device)
        x_enc_time_raw = x_enc_time.clone().detach()
        # -------------------------------------------------------------Input data normalization--------------------------------------------------------------------
        means = x_enc_time.mean(1, keepdim=True).detach()
        x_enc_time = x_enc_time - means
        stdev = torch.sqrt(torch.var(x_enc_time, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc_time /= stdev
        B, T, N = x_enc_time.size()
        self._write_shape_log(
            "llm_model.first_batch",
            [
                f"llm_model_name: {self.llm_model_name}",
                f"llm_model_path: {self.llm_model_path}",
                f"llm_hidden_size: {self.llm_hidden_size}",
                f"llm_layers: {self.llm_config.num_hidden_layers}",
            ],
        )

        # -------------------------------------------------------------Series patching and masking-----------------------------------------------------------------
        x_enc_time_patch_normal, _ = self.patch_embedding(x_enc_time.permute(0, 2, 1))
        x_enc_time_patch_mask, _, patch_mask, _ = self.random_masking(x_enc_time_patch_normal, self.mask_ratio)

        # -------------------------------------------------------------Time Encoder--------------------------------------------------------------------------------
        time_features_patch_normal, attns = self.time_patch_encoder(
            x_enc_time_patch_normal,
        )    #[B*C, N, D]
        time_features_patch_mask, attns = self.time_patch_encoder(
            x_enc_time_patch_mask,
        )    #[B*C, N, D]
        reconstruction_patch_features = time_features_patch_mask

        # -------------------------------------------------------------Intrinsic prompt reasoning-------------------------------------------------------------------
        prompt_feature = self._encode_intrinsic_prompts(x_enc_time, main_device)

        # -------------------------------------------------------------TEMPO-style component semantics------------------------------------------------------------
        trend_local, season_local, residual_local = self._decompose_local(x_enc_time_raw)
        loss_stl = self._calculate_stl_loss(
            trend_local,
            season_local,
            residual_local,
            trend_stl,
            season_stl,
            residual_stl,
        )
        role_prompts = self._get_role_prompt_embeddings(main_device)
        batch_channels = B * N
        trend_features = self._encode_component(trend_local, role_prompts[0], batch_channels)
        season_features = self._encode_component(season_local, role_prompts[1], batch_channels)
        residual_features = self._encode_component(residual_local, role_prompts[2], batch_channels)
        semantic_features = self.component_fusion(
            torch.cat([trend_features, season_features, residual_features], dim=-1)
        )

        # -------------------------------------------------------------prompt and component Cross-view Attention---------------------------------------------------
        cross_query_features = prompt_feature
        cross_key_value_features = semantic_features
        cross_direction = (
            "prompt_feature query semantic_features as key/value; "
            "fixed best direction"
        )
        llm_features = self.transformer_block(
            cross_query_features,
            cross_key_value_features,
        )
        self._write_shape_log(
            "cross_view_attention.first_batch",
            [
                (
                    f"prompt_feature shape: {self._shape_of(prompt_feature)}; "
                    "meaning: intrinsic prompt reasoning features from LLM prompts, [batch*channels, patches, d_model]"
                ),
                (
                    f"semantic_features shape: {self._shape_of(semantic_features)}; "
                    "meaning: fused trend/seasonal/residual component semantics, [batch*channels, patches, d_model]"
                ),
                (
                    f"attention direction: {cross_direction}; "
                    "output keeps the query side as residual backbone"
                ),
                (
                    f"llm_features shape: {self._shape_of(llm_features)}; "
                    "meaning: component-guided semantic features enriched by intrinsic prompt information, [batch*channels, patches, d_model]"
                ),
            ],
        )

        # -------------------------------------------------------------time-text Similarity matrix----------------------------------------------------------------
        time_norm = F.normalize(time_features_patch_normal, p=2, dim=-1)
        llm_norm = F.normalize(llm_features, p=2, dim=-1)
        logit_scale = self.logit_scale.exp()
        logits_per_time = logit_scale * torch.bmm(time_norm, llm_norm.transpose(1, 2))
        logits_per_text = logits_per_time.transpose(1, 2)
        align_loss = self._alignment_loss(
            time_features_patch_normal,
            llm_features,
            logits_per_time,
            logits_per_text,
        )

        # -------------------------------------------------------------Reconstruction-----------------------------------------------------------------------------
        multi_features = self.multimodal_Transformer_Block(reconstruction_patch_features, llm_features)
        reconstruction_direction = (
            "reconstruction_patch_features as first/self-attended side; llm_features as second/query side; "
            "fixed best reconstruction direction"
        )
        self._write_shape_log(
            "reconstruction_attention.first_batch",
            [
                (
                    f"reconstruction_patch_features shape: {self._shape_of(reconstruction_patch_features)}; "
                    "meaning: masked time/fused patch features before reconstruction"
                ),
                (
                    f"llm_features shape: {self._shape_of(llm_features)}; "
                    "meaning: cross-view text/component features"
                ),
                f"reconstruction direction: {reconstruction_direction}",
                (
                    f"multi_features shape: {self._shape_of(multi_features)}; "
                    "meaning: features projected back to the reconstructed sequence"
                ),
            ],
        )
        flat_features = rearrange(multi_features, '(b c) n d -> (b c) (n d)', c = self.channel_time, n = self.patch_num, d = self.d_model)
        output_mu = self.proj_patch(flat_features)
        output_logvar = self.proj_patch_logvar(flat_features)
        output_mu = rearrange(output_mu, '(b c) t -> b t c', t = self.seq_len, c = self.channel_time)
        output_logvar = rearrange(output_logvar, '(b c) t -> b t c', t = self.seq_len, c = self.channel_time)

        # -------------------------------------------------------------Inverse normalization-----------------------------------------------------------------------
        stdev_repeated = stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        means_repeated = means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        output_mu = output_mu * stdev_repeated
        output_mu = output_mu + means_repeated
        output_logvar = output_logvar + 2 * torch.log(stdev_repeated + 1e-6)
        output_logvar = torch.clamp(output_logvar, self.recon_logvar_min, self.recon_logvar_max)
        return output_mu, output_logvar, logits_per_time, logits_per_text, loss_stl, align_loss


    def forward(
        self,
        x_enc_time,
        trend_stl=None,
        season_stl=None,
        residual_stl=None,
    ):
        return self.Multimodal_Time_Series(
            x_enc_time,
            trend_stl,
            season_stl,
            residual_stl,
        )
