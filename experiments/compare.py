"""Run a few real TorchSpec and nanospec steps; save measurements, not a test suite.

Autograd, automatic FSDP, and torch.optim are confined to this reference oracle.
The nanospec path is always inside torch.no_grad().
"""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from nanospec import ops
from nanospec.config import (
    TARGET_MODEL,
    TARGET_REVISION,
    TORCHSPEC_REVISION,
    DraftConfig,
    Objective,
)
from nanospec.distributed import Node, TeacherBatch
from nanospec.shard import AdamW
from nanospec.train import arguments, build_draft, config_path, dummy_ids, setup_target, train


def reference(node, model, c, o, dtype, lr=3e-4):
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torchspec.training.fsdp import apply_fsdp2
    from torchspec.training.optimizer import BF16Optimizer

    from experiments.reference import TorchSpecReference, build_reference_draft

    # Meta construction avoids allocating a second random target-sized embedding.
    with torch.device("meta"):
        draft = build_reference_draft(c)
    state = model.draft_state_dict()
    state["embed_tokens.weight"] = model.embed_tokens.weight.detach()
    draft.load_state_dict(state, assign=True)
    # Rotary buffers were nonpersistent on meta and must be rebuilt explicitly.
    from torchspec.models.draft.dflash import DFlashRotaryEmbedding

    for layer in draft.layers:
        layer.self_attn.rotary_emb = DFlashRotaryEmbedding(
            c.head_dim, min(c.max_position_embeddings, 4096), c.rope_theta
        ).to(node.device)
    draft = draft.to(device=node.device, dtype=dtype)
    draft.freeze_embedding()
    ref = TorchSpecReference(draft, o)
    if node.train_gpus > 1:
        mesh = DeviceMesh.from_group(node.train_group, "cuda", mesh=list(range(node.train_gpus)))
        if dtype == torch.bfloat16:
            ref = apply_fsdp2(
                ref,
                mesh=mesh,
                args=SimpleNamespace(fsdp_strategy="FULL_SHARD", fsdp_reduce_dtype="bfloat16"),
                modules_to_shard=list(draft.layers),
            )
        else:
            # TorchSpec hardcodes BF16. The FP32 diagnostic only changes precision.
            policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)
            for layer in draft.layers:
                fully_shard(layer, mesh=mesh, mp_policy=policy)
            fully_shard(ref, mesh=mesh, mp_policy=policy)
    optimizer = BF16Optimizer(
        draft,
        lr=lr,
        weight_decay=0.01,
        max_grad_norm=1.0,
        total_steps=100,
        warmup_ratio=0,
        decay_style="constant",
    )
    return ref, optimizer


@torch.no_grad()
def similarity(a, b):
    a, b = a.float(), b.float()
    cosine = F.cosine_similarity(a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]), dim=-1)
    return {
        "cosine_mean": cosine.mean().item(),
        "cosine_min": cosine.min().item(),
        "relative_l2": ((a - b).norm() / b.norm().clamp_min(1e-12)).item(),
        "max_abs": (a - b).abs().max().item(),
    }


@torch.no_grad()
def compare_parameters(model, ref, gradients=False):
    ref_params = dict(ref.draft_model.named_parameters())
    dot = torch.zeros((), device=model.embed_tokens.weight.device, dtype=torch.float64)
    aa, bb, error, maximum = [dot.clone() for _ in range(4)]
    worst, worst_error = "", -1
    for unit_name, unit in model.sharding.named_shards():
        if gradients:
            if unit.world_size == 1:
                flat = unit.grad
            else:
                flat = torch.empty(
                    unit.shard_size * unit.world_size,
                    device=unit.grad.device,
                    dtype=unit.grad.dtype,
                )
                dist.all_gather_into_tensor(flat, unit.grad.contiguous(), group=unit.group)
            values = {
                name: flat[a:b].view(unit.shapes[name]) for name, (a, b) in unit.offsets.items()
            }
        else:
            values = unit.gather()
        for name, actual in values.items():
            key = f"{unit_name}.{name}" if unit_name.startswith("layers.") else name
            expected = ref_params[key].grad if gradients else ref_params[key].detach()
            if hasattr(expected, "full_tensor"):
                expected = expected.full_tensor()
            a, b = actual.float().flatten(), expected.float().flatten()
            dot += (a * b).sum(dtype=torch.float64)
            aa += a.square().sum(dtype=torch.float64)
            bb += b.square().sum(dtype=torch.float64)
            error += (a - b).square().sum(dtype=torch.float64)
            delta = (a - b).abs().max().item()
            if delta > worst_error:
                worst, worst_error = key, delta
            maximum = torch.maximum(maximum, (a - b).abs().max())
        if not gradients:
            unit.release()
    return {
        "cosine": (dot / (aa * bb).sqrt().clamp_min(1e-30)).clamp(-1, 1).item(),
        "relative_l2": (error / bb.clamp_min(1e-30)).sqrt().item(),
        "max_abs": maximum.item(),
        "worst_parameter": worst,
    }


class HFOracle:
    def __init__(self, node, path, c):
        self.node, self.c = node, c
        self.model = None
        if node.rank == 7:
            from transformers import Qwen3_5ForConditionalGeneration

            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                path,
                dtype=torch.bfloat16,
                device_map={"": node.device},
                attn_implementation="eager",
            )
            self.model.eval()
            # Use the library's actual FP8 linears and PyTorch delta-rule fallback.
            # No shared nanospec inference functions or replacement projections.

    @torch.no_grad()
    def forward(self, ids):
        node, c = self.node, self.c
        result = None
        if self.model is not None:
            result = self.model.model.language_model(
                input_ids=ids, use_cache=False, output_hidden_states=True
            )
        b, s = ids.shape
        layers = []
        for i in c.target_layer_ids:
            # HF hidden_states[0] is the embedding; decoder i's output is i+1.
            tensor = result.hidden_states[i + 1] if result is not None else None
            tensor = node.broadcast(tensor, (b, s, c.hidden_size), 7)
            if node.training:
                layers.append(tensor[node.rank : node.rank + 1].contiguous())
        last = node.broadcast(
            result.last_hidden_state if result is not None else None, (b, s, c.hidden_size), 7
        )
        logits = self.model.lm_head(result.last_hidden_state) if result is not None else None
        logits = node.broadcast(logits, (b, s, c.vocab_size), 7)
        if node.training:
            return (
                layers,
                last[node.rank : node.rank + 1].contiguous(),
                logits[node.rank : node.rank + 1].contiguous(),
            )


def main():
    args = arguments()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    node = Node(args.train_gpus)
    torch.set_float32_matmul_precision("highest")
    c = DraftConfig.load(config_path(args.config))
    dtype = getattr(torch, args.dtype)
    o = Objective(num_anchors=args.anchors)
    teacher = oracle = None
    if args.target:
        path, teacher = setup_target(node, c)
    model = build_draft(node, c, dtype, args.seed, teacher)
    if node.training:
        learning_rate = (
            args.learning_rate
            if args.learning_rate is not None
            else (1e-5 if args.target else 3e-4)
        )
        if learning_rate <= 0:
            raise ValueError("learning-rate must be positive")
        ref, ref_opt = reference(node, model, c, o, dtype, lr=learning_rate)
        opt = AdamW(model.sharding, lr=learning_rate, weight_decay=0.01)
        # Persist the exact common starting weights used by both implementations.
        initial = model.draft_state_dict()
        if node.rank == 0:
            initial_path = str(Path(args.output).with_suffix("")) + "-initial.pt"
            torch.save({"config": c.to_dict(), "state": initial, "seed": args.seed}, initial_path)
        del initial
        ref_capture = {}
        hook = ref.draft_model.final_norm.register_forward_hook(
            lambda _module, _args, output: ref_capture.update(hidden=output.detach().clone())
        )
    if args.target:
        oracle = HFOracle(node, path, c)
    rows, target_comparisons = [], []
    # Fix data for the whole run. Different ranks see different examples/masks.
    ids = dummy_ids(node, c, args.seq_len, args.seed + 100)
    native_teacher = hf_teacher = None
    if teacher:
        native_teacher = train(ids, verifier=teacher).teacher
        hf_teacher = oracle.forward(ids)
        if node.training:
            for i, (a, b) in enumerate(zip(native_teacher.hidden_states, hf_teacher[0])):
                target_comparisons.append({"layer": c.target_layer_ids[i], **similarity(a, b)})
            target_comparisons.append(
                {"layer": "final_norm", **similarity(native_teacher.last_hidden, hf_teacher[1])}
            )
            target_comparisons.append(
                {"layer": "logits", **similarity(native_teacher.logits, hf_teacher[2])}
            )
            print(f"rank {node.rank} target: {json.dumps(target_comparisons)}", flush=True)
    dist.barrier()
    for step in range(args.steps):
        if node.training:
            local_ids = ids[node.rank : node.rank + 1]
            mask = torch.ones_like(local_ids, dtype=torch.float32)
            # Unequal valid-token counts exercise pooled normalization across shards.
            if node.rank % 2:
                mask[:, args.seq_len // 2 : args.seq_len // 2 + 2] = 0
            if teacher:
                hs, last, logits = (
                    native_teacher.hidden_states,
                    native_teacher.last_hidden,
                    native_teacher.logits,
                )
                rhs, rlast, _ = hf_teacher
                hs, rhs = [x.to(dtype) for x in hs], [x.to(dtype) for x in rhs]
                last, rlast = last.to(dtype), rlast.to(dtype)
            else:
                gen = torch.Generator(device=node.device).manual_seed(args.seed + 1000 + node.rank)
                hs = [
                    torch.randn(
                        1,
                        args.seq_len,
                        c.hidden_size,
                        generator=gen,
                        device=node.device,
                        dtype=dtype,
                    )
                    for _ in c.target_layer_ids
                ]
                last = torch.randn(
                    1, args.seq_len, c.hidden_size, generator=gen, device=node.device, dtype=dtype
                )
                rhs, rlast = hs, last
            seed = args.seed + node.rank  # Keep anchors fixed to make loss trajectories comparable.
            torch.manual_seed(seed)
            with torch.no_grad():
                result = train(
                    local_ids,
                    model,
                    o,
                    loss_mask=mask,
                    teacher_batch=TeacherBatch(hs, last, logits.to(dtype) if teacher else None),
                )
                metrics, native_hidden = result.metrics, result.hidden
            torch.manual_seed(seed)
            ref_numerator, ref_denominator = ref(
                local_ids, rhs, mask, model.lm_head.weight, last_hidden_states=rlast
            )
            ref_numerator.backward()  # Reference ONLY. No autograd in nanospec.
            grads = compare_parameters(model, ref, gradients=True)
            hidden_comparison = similarity(native_hidden, ref_capture["hidden"])
            pooled = torch.stack(
                (
                    metrics.numerator,
                    metrics.denominator,
                    ref_numerator.detach(),
                    ref_denominator,
                )
            )
            dist.all_reduce(pooled, group=node.train_group)
            global_den = pooled[3]
            with torch.no_grad():
                for p in ref.draft_model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(node.train_gpus / global_den.clamp_min(1e-6))
                update = ops.adamw_step(opt, metrics.denominator)
            # BF16Optimizer's finite check uses the default process group. The
            # inference ranks participate below, with a zero (finite) contribution.
            reference_norm = ref_opt.step()
            if hasattr(reference_norm, "full_tensor"):
                reference_norm = reference_norm.full_tensor()
            params = compare_parameters(model, ref)
            row = {
                "step": step + 1,
                "nanospec_loss": (pooled[0] / pooled[1]).item(),
                "torchspec_loss": (pooled[2] / pooled[3]).item(),
                "draft_hidden": hidden_comparison,
                "gradients": grads,
                "parameters": params,
                "nanospec_grad_norm": update["grad_norm"],
                "torchspec_grad_norm": reference_norm.item(),
            }
            rows.append(row)
            if node.rank == 0:
                print(json.dumps(row), flush=True)
        else:
            dist.all_reduce(torch.zeros((), device=node.device))
        dist.barrier()
    if node.training:
        hook.remove()
        # Aggregate acceptance of measured tolerances over every training rank.
        fp32 = dtype == torch.float32 and not args.target
        tolerances = {
            "loss_relative": 2e-5 if fp32 else 0.02,
            "gradient_cosine": 0.99999 if fp32 else (0.98 if args.target else 0.995),
            "gradient_relative_l2": 2e-4 if fp32 else (0.2 if args.target else 0.06),
            "draft_hidden_cosine": 0.99999 if fp32 else 0.99,
            "teacher_hidden_cosine": 0.99,
            "parameter_relative_l2": 1e-5 if fp32 else 0.003,
            "parameter_max_abs": 2e-5 if fp32 else 0.01,
        }
        passed = all(
            abs(r["nanospec_loss"] - r["torchspec_loss"]) / max(abs(r["torchspec_loss"]), 1e-8)
            < tolerances["loss_relative"]
            and r["gradients"]["cosine"] > tolerances["gradient_cosine"]
            and r["gradients"]["relative_l2"] < tolerances["gradient_relative_l2"]
            and r["draft_hidden"]["cosine_mean"] > tolerances["draft_hidden_cosine"]
            and r["parameters"]["relative_l2"] < tolerances["parameter_relative_l2"]
            and r["parameters"]["max_abs"] < tolerances["parameter_max_abs"]
            for r in rows
        )
        passed = passed and all(
            r["cosine_mean"] > tolerances["teacher_hidden_cosine"] for r in target_comparisons
        )
        all_reports = [None] * node.train_gpus
        dist.all_gather_object(
            all_reports,
            {"rank": node.rank, "passed": passed, "steps": rows, "teacher": target_comparisons},
            group=node.train_group,
        )
        if node.rank == 0:
            report = {
                "passed": all(r["passed"] for r in all_reports),
                "tolerances": tolerances,
                "torchspec_revision": TORCHSPEC_REVISION,
                "reference_objective": "independent autograd CE + KL on TorchSpec components",
                "torch": torch.__version__,
                "transformers": "5.3.0",
                "config": c.to_dict(),
                "objective": asdict(o),
                "target": TARGET_MODEL if teacher else "fixed synthetic hidden states",
                "target_revision": TARGET_REVISION if teacher else None,
                "dtype": args.dtype,
                "learning_rate": learning_rate,
                "train_gpus": node.train_gpus,
                "seed": args.seed,
                "sequence_length": args.seq_len,
                "data": "fixed random token IDs; unequal rank supervision masks",
                "initial_checkpoint": initial_path,
                "ranks": all_reports,
            }
            Path(args.output).write_text(json.dumps(report, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
