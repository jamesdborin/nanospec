"""DSPark weight containers and named activation buffers. Execution lives in train.py."""

from dataclasses import dataclass

import torch
from torch import nn

from nanospec import ops


@dataclass
class SelfAttentionActivations:
    input: torch.Tensor
    query_norm: ops.RMSActivations
    key_norm: ops.RMSActivations
    attention: ops.AttentionActivations
    output_input: torch.Tensor


class DSparkAttention(nn.Module):
    def __init__(self, config, *, device="meta", dtype=None):
        super().__init__()
        self.config = config
        h, d = config.hidden_size, config.head_dim
        q, k = config.num_attention_heads, config.num_key_value_heads
        factory = {"device": device, "dtype": dtype}
        self.q_proj = nn.Linear(h, q * d, bias=False, **factory)
        self.k_proj = nn.Linear(h, k * d, bias=False, **factory)
        self.v_proj = nn.Linear(h, k * d, bias=False, **factory)
        self.o_proj = nn.Linear(q * d, h, bias=False, **factory)
        self.q_norm = nn.RMSNorm(d, eps=config.rms_norm_eps, **factory)
        self.k_norm = nn.RMSNorm(d, eps=config.rms_norm_eps, **factory)


@dataclass
class MLPActivations:
    input: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    activated_gate: torch.Tensor
    product: torch.Tensor


class DSparkMLP(nn.Module):
    def __init__(self, config, *, device="meta", dtype=None):
        super().__init__()
        h, m = config.hidden_size, config.intermediate_size
        factory = {"device": device, "dtype": dtype}
        self.gate_proj = nn.Linear(h, m, bias=False, **factory)
        self.up_proj = nn.Linear(h, m, bias=False, **factory)
        self.down_proj = nn.Linear(m, h, bias=False, **factory)


@dataclass
class DecoderActivations:
    input_norm: ops.RMSActivations
    attention: SelfAttentionActivations
    post_attention_norm: ops.RMSActivations
    mlp: MLPActivations


class DSparkDecoderLayer(nn.Module):
    def __init__(self, config, *, device="meta", dtype=None):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.self_attn = DSparkAttention(config, **factory)
        self.mlp = DSparkMLP(config, **factory)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, **factory)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **factory
        )


class MarkovHead(nn.Module):
    def __init__(self, vocab_size, rank, *, device="meta", dtype=None):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank, device=device, dtype=dtype)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False, device=device, dtype=dtype)
