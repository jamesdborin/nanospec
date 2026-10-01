# Architecture/anchor conventions adapted from TorchSpec, Copyright (c) 2026
# LightSeek Foundation. MIT license: licenses/TorchSpec-MIT.txt.
"""DSPark weight storage, checkpoint layout, and anchor preparation.

Architecture and alignment follow TorchSpec at config.TORCHSPEC_REVISION.
Layers are ordinary nn.Modules with registered, requires_grad=False parameters.
Manual shards own persistent storage; named activation records own reverse-pass buffers.
"""

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn

from nanospec.layers import DSparkDecoderLayer, MarkovHead
from nanospec.shard import Shard, ShardingPlan


def parameter_shapes(c):
    h, d, q, k, m = (
        c.hidden_size,
        c.head_dim,
        c.num_attention_heads,
        c.num_key_value_heads,
        c.intermediate_size,
    )
    shapes = {
        "context_proj.weight": (h, c.num_target_layers * c.target_hidden_size),
        "context_norm.weight": (h,),
    }
    if c.fc_norm:
        shapes.update(
            {f"fc_norm.{i}.weight": (c.target_hidden_size,) for i in range(c.num_target_layers)}
        )
    for i in range(c.num_hidden_layers):
        for key, shape in {
            "self_attn.q_proj.weight": (q * d, h),
            "self_attn.k_proj.weight": (k * d, h),
            "self_attn.v_proj.weight": (k * d, h),
            "self_attn.o_proj.weight": (h, q * d),
            "self_attn.q_norm.weight": (d,),
            "self_attn.k_norm.weight": (d,),
            "mlp.gate_proj.weight": (m, h),
            "mlp.up_proj.weight": (m, h),
            "mlp.down_proj.weight": (h, m),
            "input_layernorm.weight": (h,),
            "post_attention_layernorm.weight": (h,),
        }.items():
            shapes[f"layers.{i}.{key}"] = shape
    shapes["final_norm.weight"] = (h,)
    if c.markov_rank:
        shapes["markov_head.markov_w1.weight"] = (c.vocab_size, c.markov_rank)
        shapes["markov_head.markov_w2.weight"] = (c.vocab_size, c.markov_rank)
    return shapes


@torch.no_grad()
def initialize(c, seed=42, dtype=torch.bfloat16):
    """Independent, reproducible checkpoint; both implementations load this state."""
    generator = torch.Generator().manual_seed(seed)
    state = {}
    for name, shape in parameter_shapes(c).items():
        if len(shape) == 1:
            state[name] = torch.ones(shape, dtype=dtype)
        else:
            state[name] = torch.empty(shape, dtype=dtype).normal_(0, 0.02, generator=generator)
    return state


@torch.no_grad()
def sample_anchors(ids, loss_mask, o):
    b, s = ids.shape
    device = ids.device
    if s <= o.block_size:
        return (
            torch.zeros(b, o.num_anchors, dtype=torch.long, device=device),
            torch.zeros(b, o.num_anchors, dtype=torch.bool, device=device),
        )
    n = min(s - o.block_size + 1, s - 1)
    valid = (loss_mask[:, :n] > 0.5) & (loss_mask[:, 1 : n + 1] > 0.5)
    counts = valid.sum(1)
    indices = torch.arange(n, device=device).expand(b, -1)
    ordered = torch.where(valid, indices, s + 1).sort(1).values
    stride = (counts // o.num_anchors).clamp(1, o.block_size + 1).unsqueeze(1)
    safe_counts = counts.unsqueeze(1).clamp_min(1)
    offset = (torch.rand(b, 1, device=device) * safe_counts).long()
    slot = torch.arange(o.num_anchors, device=device).unsqueeze(0)
    rank = (offset + slot * stride) % safe_counts
    keep = slot < counts.unsqueeze(1)
    anchors = ordered.gather(1, rank.clamp(max=n - 1))
    return torch.where(keep, anchors, 0), keep


@dataclass
class PreparedBatch:
    anchors: torch.Tensor
    keep: torch.Tensor
    positions: torch.Tensor
    context_positions: torch.Tensor
    target_ids: torch.Tensor
    previous: torch.Tensor
    weights: torch.Tensor
    mask: torch.Tensor
    teacher_indices: torch.Tensor


@torch.no_grad()
def prepare(ids, loss_mask, o, anchors=None):
    b, s = ids.shape
    if s < 1:
        raise ValueError("Empty input sequence")
    a, keep = sample_anchors(ids, loss_mask, o) if anchors is None else anchors
    nb = a.shape[1]
    slot = torch.arange(o.block_size, device=ids.device).view(1, 1, -1)
    draft_positions = (a.unsqueeze(-1) + slot).reshape(b, -1)
    context_positions = torch.arange(s, device=ids.device).expand(b, -1)
    labels = a.unsqueeze(-1) + slot + 1
    safe = torch.where(keep.unsqueeze(-1), labels.clamp(max=s - 1), 0)
    target_ids = ids.unsqueeze(1).expand(-1, nb, -1).gather(2, safe)
    supervision = loss_mask.unsqueeze(1).expand(-1, nb, -1).gather(2, safe)
    valid = (keep.unsqueeze(-1) & (labels < s) & (supervision > 0.5)).int().cumprod(-1).bool()
    weights = valid.float()
    if o.gamma > 0:
        weights = weights * torch.exp(-slot.float() / o.gamma)
    previous = torch.cat((ids.gather(1, a).unsqueeze(-1), target_ids[:, :, :-1]), -1)
    q_block = torch.arange(nb * o.block_size, device=ids.device) // o.block_size
    kv = torch.arange(s + nb * o.block_size, device=ids.device)
    context_mask = (kv < s) & (kv.view(1, 1, -1) < a[:, q_block, None])
    own_block = (kv >= s) & (q_block[:, None] == (kv - s)[None, :] // o.block_size)
    mask = ((context_mask | own_block) & keep[:, q_block, None]).unsqueeze(1)
    return PreparedBatch(
        a,
        keep,
        draft_positions,
        context_positions,
        target_ids,
        previous,
        weights,
        mask,
        (safe - 1).clamp_min(0).reshape(b, -1),
    )


class DSpark(nn.Module):
    def __init__(
        self,
        config,
        state,
        embedding,
        lm_head,
        *,
        device,
        dtype=torch.bfloat16,
        group=None,
        world_size=1,
        rank=0,
    ):
        super().__init__()
        self.config = config
        expected = parameter_shapes(config)
        if set(state) != set(expected) or any(
            tuple(state[k].shape) != expected[k] for k in expected
        ):
            raise ValueError("Draft checkpoint keys/shapes do not match config")
        if (
            embedding.shape != (config.vocab_size, config.hidden_size)
            or lm_head.shape != embedding.shape
        ):
            raise ValueError("Target embedding/head shape mismatch")
        c = config
        factory = {"device": "meta", "dtype": dtype}
        self.context_proj = nn.Linear(
            c.num_target_layers * c.target_hidden_size, c.hidden_size, bias=False, **factory
        )
        self.context_norm = nn.RMSNorm(c.hidden_size, c.rms_norm_eps, **factory)
        self.fc_norm = nn.ModuleList(
            [
                nn.RMSNorm(c.target_hidden_size, c.rms_norm_eps, **factory)
                for _ in range(c.num_target_layers)
            ]
            if c.fc_norm
            else []
        )
        self.layers = nn.ModuleList(
            [DSparkDecoderLayer(c, **factory) for _ in range(c.num_hidden_layers)]
        )
        self.final_norm = nn.RMSNorm(c.hidden_size, c.rms_norm_eps, **factory)
        self.markov_head = (
            MarkovHead(c.vocab_size, c.markov_rank, **factory) if c.markov_rank else None
        )
        # Standard loading establishes the registered parameter tree before sharding.
        nn.Module.load_state_dict(self, state, assign=True)
        self.requires_grad_(False)
        self.embed_tokens = nn.Embedding.from_pretrained(
            embedding.detach().to(device=device, dtype=dtype), freeze=True
        )
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False, **factory)
        self.lm_head.weight = nn.Parameter(
            lm_head.detach().to(device=device, dtype=dtype), requires_grad=False
        )

        # Define the flat shard layout independently of module traversal order.
        placement = {
            "device": device,
            "dtype": dtype,
            "group": group,
            "world_size": world_size,
            "rank": rank,
        }
        context_names = [name for name in expected if name.startswith(("context_", "fc_norm."))]
        head_names = [name for name in expected if name.startswith(("final_norm.", "markov_head."))]
        self.sharding = ShardingPlan(
            Shard(((name, self.get_parameter(name)) for name in context_names), **placement),
            tuple(Shard(layer.named_parameters(), **placement) for layer in self.layers),
            Shard(((name, self.get_parameter(name)) for name in head_names), **placement),
        )

    def _apply(self, fn, recurse=True):
        if not hasattr(self, "sharding"):
            return super()._apply(fn, recurse=recurse)
        # All layer parameters are views of these flat buffers. Move their owner
        # first, then rebind; moving the views independently would break sharding.
        if any(s.full is not None or s.full_grad is not None for s in self.sharding):
            raise RuntimeError("Release gathered buffers before moving the model")
        self.sharding.reset_pipeline()
        for shard in self.sharding:
            shard.apply_placement(fn)
        self.embed_tokens._apply(fn, recurse=recurse)
        self.lm_head._apply(fn, recurse=recurse)
        return self

    @torch.no_grad()
    def draft_state_dict(self):
        """Collective on training ranks; TorchSpec-compatible trainable CPU weights."""
        result = OrderedDict()
        for name, shard in self.sharding.named_shards():
            weights = shard.gather()
            prefix = name + "." if name.startswith("layers.") else ""
            result.update((prefix + k, v.detach().cpu().clone()) for k, v in weights.items())
            shard.release()
        return result

    @torch.no_grad()
    def state_dict(self, *, destination=None, prefix="", keep_vars=False):
        """Full CPU module state; all training ranks must participate when sharded."""
        result = OrderedDict() if destination is None else destination
        result.update((prefix + name, value) for name, value in self.draft_state_dict().items())
        for name in ("embed_tokens.weight", "lm_head.weight"):
            value = self.get_parameter(name)
            if not keep_vars:
                value = value.detach()
            result[prefix + name] = value.cpu().clone()
        return result

    @torch.no_grad()
    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not hasattr(self, "sharding"):
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        if assign:
            raise ValueError("assign=True would replace parameters bound to manual shards")
        expected = parameter_shapes(self.config)
        expected.update(
            {
                name: tuple(self.get_parameter(name).shape)
                for name in ("embed_tokens.weight", "lm_head.weight")
            }
        )
        missing, unexpected = (
            sorted(expected.keys() - state_dict.keys()),
            sorted(state_dict.keys() - expected.keys()),
        )
        if strict and (missing or unexpected):
            raise RuntimeError(f"Missing keys: {missing}; unexpected keys: {unexpected}")
        for name in expected.keys() & state_dict.keys():
            if tuple(state_dict[name].shape) != expected[name]:
                raise RuntimeError(f"Shape mismatch for {name}: expected {expected[name]}")
        # Loading weights updates the FP32 masters. Optimizer moments are restored
        # separately by checkpoint.restore(), just as with ordinary PyTorch optimizers.
        for name, shard in self.sharding.named_shards():
            prefix = name + "." if name.startswith("layers.") else ""
            start = shard.rank * shard.shard_size
            for key, (lo, hi) in shard.offsets.items():
                full_name = prefix + key
                a, b = max(start, lo), min(start + shard.shard_size, hi)
                if full_name in state_dict and a < b:
                    shard.param[a - start : b - start].copy_(
                        state_dict[full_name].reshape(-1)[a - lo : b - lo]
                    )
            shard.master.copy_(shard.param)
        for name in ("embed_tokens.weight", "lm_head.weight"):
            if name in state_dict:
                self.get_parameter(name).copy_(state_dict[name])
        return torch.nn.modules.module._IncompatibleKeys(missing, unexpected)
