"""TorchSpec backbone and Markov head with an independent autograd CE + KL loss.

The pinned DSPark draft consists of DFlashDraftModel plus output heads. Build the
retained components directly from TorchSpec; use its backbone, anchor sampling,
masking, and FlexAttention unchanged. Only the loss tail is defined here.
"""

import torch
import torch.nn.functional as F
from torchspec.models.dflash import DFlashModel
from torchspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
from torchspec.models.draft.dspark import VanillaMarkov


def build_reference_draft(config):
    draft = DFlashDraftModel(DFlashConfig(**config.to_dict()))
    draft.markov_head = (
        VanillaMarkov(vocab_size=config.vocab_size, markov_rank=config.markov_rank)
        if config.markov_rank
        else None
    )
    return draft


class TorchSpecReference(DFlashModel):
    def __init__(self, draft, objective):
        super().__init__(
            draft,
            block_size=objective.block_size,
            num_anchors=objective.num_anchors,
            loss_decay_gamma=objective.gamma,
        )
        self.objective = objective

    def forward(self, input_ids, hidden_states, loss_mask, lm_head_weight, last_hidden_states):
        o = self.objective
        batch, length = input_ids.shape
        hidden, anchors, keep, blocks = self._draft_backbone(input_ids, hidden_states, loss_mask)
        logits = F.linear(hidden, lm_head_weight).view(batch, blocks, o.block_size, -1)

        # DSPark supervises every slot, predicting anchor + slot + 1.
        slot = torch.arange(o.block_size, device=input_ids.device).view(1, 1, -1)
        positions = anchors.unsqueeze(-1) + slot + 1
        indices = torch.where(keep.unsqueeze(-1), positions.clamp(max=length - 1), 0)
        labels = input_ids.unsqueeze(1).expand(-1, blocks, -1).gather(2, indices)
        supervised = loss_mask.unsqueeze(1).expand(-1, blocks, -1).gather(2, indices)
        valid = keep.unsqueeze(-1) & (positions < length) & (supervised > 0.5)
        weights = valid.int().cumprod(-1).float()
        if o.gamma > 0:
            weights = weights * torch.exp(-slot.float() / o.gamma)

        if self.draft_model.markov_head is not None:
            previous = torch.cat(
                (input_ids.gather(1, anchors).unsqueeze(-1), labels[:, :, :-1]), dim=-1
            )
            logits = self.draft_model.markov_head.apply_block_logits(logits, token_ids=previous)

        with torch.no_grad():
            target_indices = (indices - 1).clamp_min(0).reshape(batch, -1, 1)
            aligned = last_hidden_states.gather(
                1, target_indices.expand(-1, -1, last_hidden_states.shape[-1])
            )
            target_logits = F.linear(aligned, lm_head_weight).view_as(logits)
            target_probs = target_logits.float().softmax(-1)

        ce = F.cross_entropy(logits.flatten(0, 2), labels.flatten(), reduction="none")
        kl = F.kl_div(logits.float().log_softmax(-1), target_probs, reduction="none").sum(-1)
        ce_numerator = (ce.view_as(weights) * weights).sum()
        kl_numerator = (kl * weights).sum()
        return o.ce * ce_numerator + o.kl * kl_numerator, weights.sum()
