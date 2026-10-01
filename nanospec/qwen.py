# Portions adapted from Transformers: Copyright 2025 The Qwen Team and
# The HuggingFace Inc. team. Licensed under Apache 2.0; see
# licenses/Transformers-Apache-2.0.txt and THIRD_PARTY_NOTICES.md.
# Modified for explicit checkpoint tensors and synchronous pipeline inference.
"""Qwen3.5 checkpoint and registered weight containers; execution is in train.py.

FP8 storage and dynamic 1x128 activation quantization are preserved. The initial
PyTorch-only matmul dequantizes to FP32; it deliberately does not claim native
FP8 tensor-core throughput. Weights are dequantized one projection at a time.
No KV cache: full sequences supply the training teacher in synchronous phases.
"""

import json
from pathlib import Path

from safetensors import safe_open
from torch import nn


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.config = json.loads((self.path / "config.json").read_text())
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]

    def load(self, keys, device):
        out = {}
        files = {}
        for key in keys:
            files.setdefault(self.index[key], []).append(key)
        for filename, names in files.items():
            with safe_open(self.path / filename, framework="pt", device="cpu") as f:
                for name in names:
                    out[name] = f.get_tensor(name).to(device)
        return out

    def frozen(self, device="cpu"):
        return self.load(["model.language_model.embed_tokens.weight", "lm_head.weight"], device)


class FP8Linear(nn.Linear):
    """Registered checkpoint projection; scales are buffers and weights stay in FP8."""

    def __init__(self, weight, scale=None):
        out_features, in_features = weight.shape
        super().__init__(in_features, out_features, bias=False, device="meta")
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_buffer("weight_scale_inv", scale)

    @classmethod
    def from_checkpoint(cls, weights, prefix):
        return cls(weights[prefix + ".weight"], weights.get(prefix + ".weight_scale_inv"))


class QwenRMSNorm(nn.Module):
    def __init__(self, weight, eps):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.eps = eps


class QwenGatedRMSNorm(nn.Module):
    def __init__(self, weight, eps):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.eps = eps


class QwenAttention(nn.Module):
    def __init__(self, config, weights, prefix):
        super().__init__()
        self.head_dim = config["head_dim"]
        self.num_heads = config["num_attention_heads"]
        self.num_key_value_heads = config["num_key_value_heads"]
        rope = config["rope_parameters"]
        self.rotary_dim = int(self.head_dim * rope["partial_rotary_factor"])
        self.rope_theta = rope["rope_theta"]
        self.q_proj = FP8Linear.from_checkpoint(weights, prefix + ".q_proj")
        self.k_proj = FP8Linear.from_checkpoint(weights, prefix + ".k_proj")
        self.v_proj = FP8Linear.from_checkpoint(weights, prefix + ".v_proj")
        self.o_proj = FP8Linear.from_checkpoint(weights, prefix + ".o_proj")
        self.q_norm = QwenRMSNorm(weights[prefix + ".q_norm.weight"], config["rms_norm_eps"])
        self.k_norm = QwenRMSNorm(weights[prefix + ".k_norm.weight"], config["rms_norm_eps"])


class QwenGatedDeltaNet(nn.Module):
    def __init__(self, config, weights, prefix):
        super().__init__()
        self.num_key_heads = config["linear_num_key_heads"]
        self.num_value_heads = config["linear_num_value_heads"]
        self.key_head_dim = config["linear_key_head_dim"]
        self.value_head_dim = config["linear_value_head_dim"]
        self.in_proj_qkv = FP8Linear.from_checkpoint(weights, prefix + ".in_proj_qkv")
        self.in_proj_z = FP8Linear.from_checkpoint(weights, prefix + ".in_proj_z")
        self.in_proj_b = FP8Linear.from_checkpoint(weights, prefix + ".in_proj_b")
        self.in_proj_a = FP8Linear.from_checkpoint(weights, prefix + ".in_proj_a")
        self.out_proj = FP8Linear.from_checkpoint(weights, prefix + ".out_proj")
        weight = weights[prefix + ".conv1d.weight"]
        channels, _, kernel_size = weight.shape
        self.conv1d = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=config["linear_conv_kernel_dim"] - 1,
            groups=channels,
            bias=False,
            device="meta",
        )
        self.conv1d.weight = nn.Parameter(weight, requires_grad=False)
        self.A_log = nn.Parameter(weights[prefix + ".A_log"], requires_grad=False)
        self.dt_bias = nn.Parameter(weights[prefix + ".dt_bias"], requires_grad=False)
        self.norm = QwenGatedRMSNorm(weights[prefix + ".norm.weight"], config["rms_norm_eps"])


class QwenMLP(nn.Module):
    def __init__(self, weights, prefix):
        super().__init__()
        self.gate_proj = FP8Linear.from_checkpoint(weights, prefix + ".gate_proj")
        self.up_proj = FP8Linear.from_checkpoint(weights, prefix + ".up_proj")
        self.down_proj = FP8Linear.from_checkpoint(weights, prefix + ".down_proj")


class QwenDecoderLayer(nn.Module):
    def __init__(self, config, weights, index):
        super().__init__()
        prefix = f"layers.{index}"
        self.input_layernorm = QwenRMSNorm(
            weights[prefix + ".input_layernorm.weight"], config["rms_norm_eps"]
        )
        self.post_attention_layernorm = QwenRMSNorm(
            weights[prefix + ".post_attention_layernorm.weight"], config["rms_norm_eps"]
        )
        if config["layer_types"][index] == "linear_attention":
            self.linear_attn = QwenGatedDeltaNet(config, weights, prefix + ".linear_attn")
            self.self_attn = None
        else:
            self.linear_attn = None
            self.self_attn = QwenAttention(config, weights, prefix + ".self_attn")
        self.mlp = QwenMLP(weights, prefix + ".mlp")


class QwenStage(nn.Module):
    def __init__(self, checkpoint, start, end, *, device, first=False, last=False):
        super().__init__()
        self.config = checkpoint.config["text_config"]
        if checkpoint.config["model_type"] != "qwen3_5":
            raise ValueError("Only the dense Qwen3.5 text backbone is implemented")
        quant = checkpoint.config.get("quantization_config", {})
        if (
            quant.get("weight_block_size") != [128, 128]
            or quant.get("activation_scheme") != "dynamic"
        ):
            raise ValueError("Expected dynamic block-FP8 Qwen checkpoint")
        if self.config["rope_parameters"]["rope_type"] != "default":
            raise ValueError("Only default text RoPE is supported")
        self.start, self.end = start, end
        prefix = "model.language_model."
        keys = [
            k
            for k in checkpoint.index
            if any(k.startswith(f"{prefix}layers.{i}.") for i in range(start, end))
        ]
        if first:
            keys.append(prefix + "embed_tokens.weight")
        if last:
            keys.extend((prefix + "norm.weight", "lm_head.weight"))
        loaded = checkpoint.load(keys, device)
        weights = {k.removeprefix(prefix): v for k, v in loaded.items()}
        self.layers = nn.ModuleList(
            [QwenDecoderLayer(self.config, weights, i) for i in range(start, end)]
        )
        self.embed_tokens = (
            nn.Embedding.from_pretrained(weights["embed_tokens.weight"], freeze=True)
            if first
            else None
        )
        self.norm = (
            QwenRMSNorm(weights["norm.weight"], self.config["rms_norm_eps"]) if last else None
        )
        self.lm_head = None
        if last:
            self.lm_head = nn.Linear(
                self.config["hidden_size"], self.config["vocab_size"], bias=False, device="meta"
            )
            self.lm_head.weight = nn.Parameter(weights["lm_head.weight"], requires_grad=False)
        self.eval()
