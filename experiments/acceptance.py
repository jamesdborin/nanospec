"""Focused acceptance-metric checks without a teacher checkpoint or distributed node."""

import argparse

import torch
import torch.nn.functional as F

from nanospec.config import Objective
from nanospec.dspark import prepare
from nanospec.objective import acceptance_length_counts, ce_kl_backward, ce_kl_loss


def check_counts(device):
    target = torch.tensor([[[[1.0, 0.0]] * 3]], device=device)
    draft = torch.tensor([[[[0.8, 0.2], [0.5, 0.5], [0.9, 0.1]]]], device=device)
    valid = torch.ones(1, 1, 3, dtype=torch.bool, device=device)
    total, count = acceptance_length_counts(draft, target, valid)
    torch.testing.assert_close(total, torch.tensor(2.56, device=device))
    assert count.item() == 1
    perfect, _ = acceptance_length_counts(target, target, valid)
    assert perfect.item() == 4
    wrong = target.flip(-1)
    rejected, _ = acceptance_length_counts(wrong, target, valid)
    assert rejected.item() == 1

    # A gap ends the prefix even if a later slot has supervision.
    gap = torch.tensor([[[True, False, True]]], device=device)
    truncated, _ = acceptance_length_counts(draft, target, gap)
    torch.testing.assert_close(truncated, torch.tensor(1.8, device=device))
    empty_sum, empty_count = acceptance_length_counts(draft, target, ~valid)
    assert empty_sum.item() == empty_count.item() == 0

    # Pool counts, not rank-local means: one valid block vs. two valid blocks.
    other_sum, other_count = acceptance_length_counts(
        target.expand(2, -1, -1, -1),
        target.expand(2, -1, -1, -1),
        valid.expand(2, -1, -1),
    )
    pooled = (total + other_sum + empty_sum) / (count + other_count + empty_count)
    torch.testing.assert_close(pooled, torch.tensor(10.56 / 3, device=device))


def check_objective(device, dtype):
    ids = torch.tensor([[0, 1, 2, 3, 0]], device=device)
    supervision = torch.tensor([[1, 1, 1, 0, 1]], device=device)
    anchors = (
        torch.tensor([[0, 1, 0]], device=device),
        torch.tensor([[True, True, False]], device=device),
    )
    objective = Objective(block_size=3, num_anchors=3)
    prepared = prepare(ids, supervision, objective, anchors=anchors)
    expected_valid = torch.tensor(
        [[[True, True, False], [True, False, False], [False, False, False]]],
        device=device,
    )
    assert torch.equal(prepared.valid, expected_valid)
    logits = torch.randn(1, 3, 3, 4, device=device, dtype=dtype)
    target_logits = torch.randn_like(logits)
    metrics, saved = ce_kl_loss(logits, target_logits, prepared, objective)
    assert metrics.accept_len_total.item() == 2
    assert 1 <= metrics.accept_len.item() <= 3
    assert not metrics.accept_len.requires_grad

    # Metric validity is independent of decay, even when weights underflow.
    decayed = prepare(
        ids,
        supervision,
        Objective(block_size=3, num_anchors=3, gamma=1e-8),
        anchors=anchors,
    )
    decayed_metrics, _ = ce_kl_loss(logits, target_logits, decayed, objective)
    torch.testing.assert_close(metrics.accept_len, decayed_metrics.accept_len)
    empty = prepare(ids, torch.zeros_like(supervision), objective, anchors=anchors)
    empty_metrics, _ = ce_kl_loss(logits, target_logits, empty, objective)
    assert empty_metrics.accept_len.item() == 0
    assert empty_metrics.accept_len_total.item() == 0

    # The metric must not change the existing CE + KL objective.
    ce = F.cross_entropy(
        logits.flatten(0, 2), prepared.target_ids.flatten(), reduction="none"
    ).view_as(prepared.weights)
    logp = logits.float().log_softmax(-1)
    logq = target_logits.float().log_softmax(-1)
    kl = (target_logits.float().softmax(-1) * (logq - logp)).sum(-1)
    expected_loss = (
        objective.ce * (ce * prepared.weights).sum() + objective.kl * (kl * prepared.weights).sum()
    ) / prepared.weights.sum()
    torch.testing.assert_close(metrics.loss, expected_loss)
    assert torch.isfinite(ce_kl_backward(saved)).all()

    if dtype == torch.float32:
        # Independent autograd oracle for the unnormalized manual derivative.
        reference_logits = logits.clone().requires_grad_()
        reference_ce = F.cross_entropy(
            reference_logits.flatten(0, 2),
            prepared.target_ids.flatten(),
            reduction="none",
        ).view_as(prepared.weights)
        reference_kl = (logq.exp() * (logq - reference_logits.log_softmax(-1))).sum(-1)
        reference = (
            (objective.ce * reference_ce + objective.kl * reference_kl) * prepared.weights
        ).sum()
        reference.backward()
        torch.testing.assert_close(ce_kl_backward(saved), reference_logits.grad)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    torch.manual_seed(42)
    check_counts(args.device)
    for dtype in (torch.float32, torch.bfloat16):
        check_objective(args.device, dtype)
    print(f"Acceptance counts, masks, pooling, loss and gradient checks passed on {args.device}.")


if __name__ == "__main__":
    main()
