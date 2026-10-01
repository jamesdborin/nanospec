"""Handwritten local derivatives. Saved tuples contain activations, never weights.

These are fixed forward/backward pairs, not a dynamic differentiation engine.
All calls work with grad mode disabled. Casting boundaries follow DFlashRMSNorm.
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F

from nanospec.objective import ce_kl_backward as ce_kl_backward, ce_kl_loss as ce_kl_loss
from nanospec.shard import adamw_step as adamw_step


class RMSActivations(NamedTuple):
    input: torch.Tensor
    inverse_rms: torch.Tensor
    normalized: torch.Tensor


class AttentionActivations(NamedTuple):
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    probabilities: torch.Tensor


class RotaryFactors(NamedTuple):
    cos: torch.Tensor
    sin: torch.Tensor


def linear(x, weight, bias=None):
    return F.linear(x, weight, bias)


def linear_backward(dy, x, weight, weight_grad, *, accumulate=False):
    dx = dy @ weight
    dw = dy.reshape(-1, dy.shape[-1]).T @ x.reshape(-1, x.shape[-1])
    if accumulate:
        weight_grad.add_(dw)
    else:
        weight_grad.copy_(dw)
    return dx


def rms_norm(x, weight, eps):
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    normalized = (xf * inv).to(x.dtype)
    return weight * normalized, RMSActivations(x, inv, normalized)


def rms_norm_backward(dy, saved, weight, weight_grad):
    x, inv, normalized = saved
    dw = (dy * normalized).sum(tuple(range(dy.ndim - 1)))
    # The multiply by weight is in the input dtype; the normalization is FP32.
    dz = (dy * weight).float()
    xf = x.float()
    # Keep PyTorch's pow -> mean -> rsqrt -> multiply derivative ordering.
    dinv = (dz * xf).sum(-1, keepdim=True)
    dvar = dinv * (-0.5 * inv.pow(3))
    dx = dz * inv + (dvar.expand_as(xf) / x.shape[-1]) * (2.0 * xf)
    weight_grad.copy_(dw)
    return dx.to(x.dtype)


def embedding(ids, weight):
    return F.embedding(ids, weight)


def embedding_backward(dy, ids, weight_grad):
    weight_grad.index_add_(0, ids.flatten(), dy.reshape(-1, weight_grad.shape[-1]))


def silu(x):
    return F.silu(x)


def softplus(x):
    return F.softplus(x)


def conv1d(x, weight, *, padding, groups):
    return F.conv1d(x, weight, padding=padding, groups=groups)


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def rotary_factors(position_ids, dim, theta, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=position_ids.device).float() / dim))
    angles = position_ids.float().unsqueeze(-1) * inv
    angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
    return RotaryFactors(angles.cos().to(dtype), angles.sin().to(dtype))


def rope(x, factors):
    cos, sin = factors
    return x * cos + rotate_half(x) * sin


def rope_backward(dy, factors):
    cos, sin = factors
    return dy * cos - rotate_half(dy * sin)


def partial_rope(x, factors, rotary_dim):
    return torch.cat((rope(x[..., :rotary_dim], factors), x[..., rotary_dim:]), -1)


def causal_attention(q, k, v):
    """Qwen's BF16 eager attention ordering, distinct from draft attention."""
    groups = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    scores = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    s = q.shape[-2]
    mask = torch.ones(s, s, device=q.device, dtype=torch.bool).triu(1)
    scores = scores + torch.zeros(s, s, device=q.device, dtype=q.dtype).masked_fill(
        mask, torch.finfo(q.dtype).min
    )
    probs = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
    return probs @ v


def attention(q, k, v, mask):
    """Explicit GQA and masked softmax, including fully masked anchor rows.

    FlexAttention uses fused FP32 score/probability accumulation. Here those
    buffers are visible; changing fusion necessarily changes rounding slightly.
    """
    groups = q.shape[1] // k.shape[1]
    kr = k.repeat_interleave(groups, dim=1)
    vr = v.repeat_interleave(groups, dim=1)
    scores = (q.float() @ kr.float().transpose(-1, -2)) * q.shape[-1] ** -0.5
    valid = mask.any(-1, keepdim=True)
    scores.masked_fill_(~mask, -torch.inf)
    scores = torch.where(valid, scores, 0.0)
    prob = scores.softmax(-1) * valid
    out = (prob @ vr.float()).to(q.dtype)
    return out, AttentionActivations(q, k, v, prob)


def attention_backward(dout, saved):
    q, k, v, prob = saved
    groups = q.shape[1] // k.shape[1]
    kr = k.repeat_interleave(groups, dim=1).float()
    vr = v.repeat_interleave(groups, dim=1).float()
    do = dout.float()
    dp = do @ vr.transpose(-1, -2)
    ds = prob * (dp - (dp * prob).sum(-1, keepdim=True))
    ds *= q.shape[-1] ** -0.5
    dq = ds @ kr
    dk = ds.transpose(-1, -2) @ q.float()
    dv = prob.transpose(-1, -2) @ do
    b, hk, s, d = k.shape
    dk = dk.view(b, hk, groups, s, d).sum(2)
    dv = dv.view(b, hk, groups, s, d).sum(2)
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


def silu_backward(dy, x):
    sig = x.sigmoid()
    return dy * (sig * (1 + x * (1 - sig)))


def qwen_norm(x, weight, eps):
    y = x.float()
    return ((y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)) * (1 + weight.float())).to(
        x.dtype
    )


def qwen_gated_norm(x, gate, weight, eps):
    xf = x.float()
    normalized = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    return (weight * normalized * F.silu(gate.float())).to(x.dtype)


def fp8_linear(x, weight, scale):
    """Reference quantization, expanded scales, FP32 accumulation, BF16 output."""
    if weight.dtype != torch.float8_e4m3fn:
        return F.linear(x, weight)
    if weight.shape[0] % 128 or weight.shape[1] % 128:
        raise ValueError("Expected checkpoint's 128x128 FP8 weight blocks")
    xb = x.float().reshape(*x.shape[:-1], -1, 128)
    xs = xb.abs().amax(-1, keepdim=True) / 448.0
    # A zero block represents exact zero (avoid zero/zero during quantization).
    qx = (
        (xb / xs.clamp_min(torch.finfo(torch.float32).tiny))
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    dx = (qx.float() * xs).reshape_as(x)
    n, k = weight.shape
    dw = (weight.float().view(n // 128, 128, k // 128, 128) * scale[:, None, :, None]).reshape(n, k)
    return F.linear(dx, dw).to(x.dtype)


def chunk_delta_rule(q, k, v, g, beta, chunk_size=64):
    """Native PyTorch chunked gated delta rule, matching HF's fallback ordering."""
    dtype = q.dtype
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g)]
    b, h, s, d = k.shape
    dv = v.shape[-1]
    pad = (-s) % chunk_size
    q, k, v = [F.pad(x, (0, 0, 0, pad)) for x in (q, k, v)]
    beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    q = q * d**-0.5
    vb, kb = v * beta.unsqueeze(-1), k * beta.unsqueeze(-1)
    q, k, v, kb, vb = [x.reshape(b, h, -1, chunk_size, x.shape[-1]) for x in (q, k, v, kb, vb)]
    g = g.reshape(b, h, -1, chunk_size).cumsum(-1)
    upper = torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool).triu()
    decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()
    a = -((kb @ k.transpose(-1, -2)) * decay).masked_fill(upper, 0)
    for i in range(1, chunk_size):
        row, sub = a[..., i, :i].clone(), a[..., :i, :i].clone()
        a[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    a = a + torch.eye(chunk_size, device=q.device)
    v = a @ vb
    kc = a @ (kb * g.exp().unsqueeze(-1))
    state = torch.zeros(b, h, d, dv, device=q.device)
    out = torch.zeros_like(v)
    upper = upper.triu(1)
    for i in range((s + pad) // chunk_size):
        qi, ki, vi, gi = q[:, :, i], k[:, :, i], v[:, :, i], g[:, :, i]
        ai = (qi @ ki.transpose(-1, -2) * decay[:, :, i]).masked_fill(upper, 0)
        vn = vi - kc[:, :, i] @ state
        out[:, :, i] = (qi * gi[..., None].exp()) @ state + ai @ vn
        state = (
            state * gi[..., -1, None, None].exp()
            + (ki * (gi[..., -1, None] - gi).exp()[..., None]).transpose(-1, -2) @ vn
        )
    return out.reshape(b, h, -1, dv)[:, :, :s].transpose(1, 2).contiguous().to(dtype)
