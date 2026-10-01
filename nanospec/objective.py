"""CE + KL tensor operations and their handwritten derivative."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nanospec.config import Objective


@dataclass
class LossMetrics:
    ce: torch.Tensor
    kl: torch.Tensor
    numerator: torch.Tensor
    denominator: torch.Tensor

    @property
    def loss(self):
        return self.numerator / self.denominator.clamp_min(1e-6)


@dataclass
class LossActivations:
    objective: Objective
    logits: torch.Tensor
    probabilities: torch.Tensor
    target_probabilities: torch.Tensor
    weights: torch.Tensor
    target_ids: torch.Tensor


@torch.no_grad()
def ce_kl_loss(logits, target_logits, prepared, objective):
    o = objective
    weights, labels = prepared.weights, prepared.target_ids
    denominator = weights.sum()
    ce = F.cross_entropy(logits.flatten(0, 2), labels.flatten(), reduction="none").view_as(weights)
    probs = logits.float().softmax(-1)
    target = target_logits.float().softmax(-1)
    logp, logq = logits.float().log_softmax(-1), target_logits.float().log_softmax(-1)
    kl = (target * (logq - logp)).sum(-1)
    ce, kl = [(x * weights).sum() for x in (ce, kl)]
    numerator = o.ce * ce + o.kl * kl
    den = denominator.clamp_min(1e-6)
    metrics = LossMetrics(ce / den, kl / den, numerator, denominator)
    saved = LossActivations(o, logits, probs, target, weights, labels)
    return metrics, saved


@torch.no_grad()
def ce_kl_backward(saved, accumulation_steps=1):
    o = saved.objective
    logits, probs, target = saved.logits, saved.probabilities, saved.target_probabilities
    scale = saved.weights / accumulation_steps
    # Match upstream CE's BF16 boundary; distribution matching accumulates in FP32.
    ce_scale = (scale * o.ce).to(logits.dtype).unsqueeze(-1)
    dl = logits.log_softmax(-1).exp() * ce_scale
    dl.scatter_add_(-1, saved.target_ids.unsqueeze(-1), -ce_scale)
    kl_grad = (probs - target) * (scale * o.kl).unsqueeze(-1)
    return dl + kl_grad.to(logits.dtype)
