"""Synchronous training entry point: teacher -> transfer -> draft -> explicit update."""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download

from nanospec import checkpoint, ops
from nanospec.config import TARGET_MODEL, TARGET_REVISION, DraftConfig, Objective
from nanospec.distributed import Node, Teacher, TeacherBatch
from nanospec.dspark import DSpark, initialize, prepare
from nanospec.layers import DecoderActivations, MLPActivations, SelfAttentionActivations
from nanospec.objective import LossMetrics
from nanospec.shard import AdamW


@dataclass
class TrainResult:
    teacher: TeacherBatch | None = None
    metrics: LossMetrics | None = None
    hidden: torch.Tensor | None = None
    update: dict | None = None


@torch.no_grad()
def train(
    ids,
    drafter=None,
    objective=None,
    *,
    verifier=None,
    teacher_batch=None,
    loss_mask=None,
    optimizer=None,
    anchors=None,
    accumulation_steps=1,
    overlap=True,
):
    """One synchronous step: verifier -> NVLink -> draft -> loss -> backward -> AdamW.

    Modules hold weights; this function owns execution and activation lifetimes.
    Comparisons can supply fixed teacher tensors and omit the optimizer to inspect
    raw gradients. Passing only a verifier runs the same inference/transfer path.
    """
    if verifier is not None and teacher_batch is not None:
        raise ValueError("Supply a verifier or fixed teacher tensors, not both")

    # Verifier: each inference GPU executes its contiguous range of Qwen layers.
    if verifier is not None:
        node, stage = verifier.node, verifier.stage
        ranks = node.inference_ranks
        batch, sequence = ids.shape
        width = verifier.config["hidden_size"]
        captures, output = {}, None
        if stage is not None:
            index = ranks.index(node.rank)
            if stage.embed_tokens is not None:
                x = ops.embedding(ids, stage.embed_tokens.weight)
            else:
                x = torch.empty(batch, sequence, width, device=node.device, dtype=torch.bfloat16)
                dist.recv(x, src=ranks[index - 1])

            for layer_index, layer in enumerate(stage.layers, start=stage.start):
                n = ops.qwen_norm(x, layer.input_layernorm.weight, layer.input_layernorm.eps)
                if layer.linear_attn is not None:
                    # Gated delta attention: projections, causal convolution, recurrence, gate.
                    attn = layer.linear_attn
                    nk, nv = attn.num_key_heads, attn.num_value_heads
                    dk, dv = attn.key_head_dim, attn.value_head_dim
                    mixed = ops.fp8_linear(
                        n, attn.in_proj_qkv.weight, attn.in_proj_qkv.weight_scale_inv
                    ).transpose(1, 2)
                    z = ops.fp8_linear(
                        n, attn.in_proj_z.weight, attn.in_proj_z.weight_scale_inv
                    ).reshape(batch, sequence, nv, dv)
                    beta = ops.fp8_linear(
                        n, attn.in_proj_b.weight, attn.in_proj_b.weight_scale_inv
                    ).sigmoid()
                    a = ops.fp8_linear(n, attn.in_proj_a.weight, attn.in_proj_a.weight_scale_inv)
                    mixed = ops.conv1d(
                        mixed,
                        attn.conv1d.weight,
                        padding=attn.conv1d.padding,
                        groups=attn.conv1d.groups,
                    )[:, :, :sequence]
                    mixed = ops.silu(mixed).transpose(1, 2)
                    q, k, v = mixed.split((nk * dk, nk * dk, nv * dv), -1)
                    q, k = [
                        y.reshape(batch, sequence, nk, dk).repeat_interleave(nv // nk, 2)
                        for y in (q, k)
                    ]
                    v = v.reshape(batch, sequence, nv, dv)
                    g = -attn.A_log.float().exp() * ops.softplus(a.float() + attn.dt_bias)
                    out = ops.chunk_delta_rule(q, k, v, g, beta).reshape(-1, dv)
                    out = ops.qwen_gated_norm(
                        out, z.reshape(-1, dv), attn.norm.weight, attn.norm.eps
                    )
                    out = ops.fp8_linear(
                        out.reshape(batch, sequence, nv * dv),
                        attn.out_proj.weight,
                        attn.out_proj.weight_scale_inv,
                    )
                else:
                    # Full attention: Q/K normalization, partial RoPE, causal GQA, output gate.
                    attn = layer.self_attn
                    d, heads, kv_heads = attn.head_dim, attn.num_heads, attn.num_key_value_heads
                    q, gate = (
                        ops.fp8_linear(n, attn.q_proj.weight, attn.q_proj.weight_scale_inv)
                        .view(batch, sequence, heads, 2 * d)
                        .chunk(2, -1)
                    )
                    gate = gate.reshape(batch, sequence, heads * d)
                    q = ops.qwen_norm(q, attn.q_norm.weight, attn.q_norm.eps).transpose(1, 2)
                    k = ops.fp8_linear(n, attn.k_proj.weight, attn.k_proj.weight_scale_inv)
                    k = ops.qwen_norm(
                        k.view(batch, sequence, kv_heads, d), attn.k_norm.weight, attn.k_norm.eps
                    ).transpose(1, 2)
                    v = (
                        ops.fp8_linear(n, attn.v_proj.weight, attn.v_proj.weight_scale_inv)
                        .view(batch, sequence, kv_heads, d)
                        .transpose(1, 2)
                    )
                    positions = torch.arange(sequence, device=x.device).expand(batch, -1)
                    rope = ops.rotary_factors(positions, attn.rotary_dim, attn.rope_theta, x.dtype)
                    q = ops.partial_rope(q, rope, attn.rotary_dim)
                    k = ops.partial_rope(k, rope, attn.rotary_dim)
                    out = ops.causal_attention(q, k, v)
                    out = out.transpose(1, 2).contiguous().reshape(batch, sequence, heads * d)
                    out = ops.fp8_linear(
                        out * gate.sigmoid(), attn.o_proj.weight, attn.o_proj.weight_scale_inv
                    )
                x = x + out

                # Qwen's dense SwiGLU MLP and residual.
                n = ops.qwen_norm(
                    x, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps
                )
                mlp = layer.mlp
                gate = ops.fp8_linear(n, mlp.gate_proj.weight, mlp.gate_proj.weight_scale_inv)
                up = ops.fp8_linear(n, mlp.up_proj.weight, mlp.up_proj.weight_scale_inv)
                out = ops.fp8_linear(
                    ops.silu(gate) * up, mlp.down_proj.weight, mlp.down_proj.weight_scale_inv
                )
                x = x + out
                if layer_index in verifier.capture:
                    captures[layer_index] = x
            if stage.norm is not None:
                x = ops.qwen_norm(x, stage.norm.weight, stage.norm.eps)
            output = x
            if index < len(ranks) - 1:
                dist.send(output.contiguous(), dst=ranks[index + 1])

        torch.cuda.synchronize()
        dist.barrier()
        # Transfer captured states and logits using NCCL CUDA buffers over NVLink.
        features = []
        for layer_index in verifier.capture:
            source = next(
                r for r, (start, end) in verifier.partitions.items() if start <= layer_index < end
            )
            feature = node.broadcast(captures.get(layer_index), (batch, sequence, width), source)
            if node.training:
                features.append(feature[node.rank : node.rank + 1].contiguous())
        final = node.broadcast(
            output if node.rank == ranks[-1] else None, (batch, sequence, width), ranks[-1]
        )
        logits = ops.linear(output, stage.lm_head.weight) if node.rank == ranks[-1] else None
        logits = node.broadcast(logits, (batch, sequence, verifier.config["vocab_size"]), ranks[-1])
        dist.barrier()
        if node.training:
            teacher_batch = TeacherBatch(
                features,
                final[node.rank : node.rank + 1].contiguous(),
                logits[node.rank : node.rank + 1].contiguous(),
            )
            ids = ids[node.rank : node.rank + 1]

    if drafter is None:
        return TrainResult(teacher=teacher_batch)
    if teacher_batch is None or objective is None:
        raise ValueError("Draft training requires teacher tensors and an objective")
    c, o = drafter.config, objective
    dtype = drafter.embed_tokens.weight.dtype
    hidden_states = [h.to(dtype) for h in teacher_batch.hidden_states]
    loss_mask = torch.ones_like(ids, dtype=torch.float32) if loss_mask is None else loss_mask

    # Draft context projection. Save only the activations needed by the reverse pass.
    forward = drafter.sharding.schedule(overlap=overlap)
    forward.prefetch(0)
    forward.acquire(0)
    forward.prefetch(1)
    feature_norms, inputs = [], hidden_states
    if drafter.fc_norm:
        inputs = []
        for norm, feature in zip(drafter.fc_norm, hidden_states):
            normalized, saved = ops.rms_norm(feature, norm.weight, norm.eps)
            inputs.append(normalized)
            feature_norms.append(saved)
    context_input = torch.cat(inputs, -1).to(dtype)
    projected = ops.linear(context_input, drafter.context_proj.weight)
    context, context_norm = ops.rms_norm(
        projected, drafter.context_norm.weight, drafter.context_norm.eps
    )
    forward.release(0)
    prepared = prepare(ids, loss_mask, o, anchors)
    batch, blocks = prepared.anchors.shape
    noise = torch.full(
        (batch, blocks, o.block_size), c.mask_token_id, dtype=torch.long, device=ids.device
    )
    noise[:, :, 0] = torch.where(prepared.keep, ids.gather(1, prepared.anchors), c.mask_token_id)
    x = ops.embedding(noise.flatten(1), drafter.embed_tokens.weight)
    query_rope = ops.rotary_factors(prepared.positions, c.head_dim, c.rope_theta, dtype)
    key_rope = ops.rotary_factors(
        torch.cat((prepared.context_positions, prepared.positions), 1),
        c.head_dim,
        c.rope_theta,
        dtype,
    )
    sequence, ctx = x.shape[1], context.shape[1]
    heads, kv_heads, d = c.num_attention_heads, c.num_key_value_heads, c.head_dim
    activations = []

    # Draft forward: gather -> norm -> attention -> residual -> norm -> MLP -> residual.
    for i, (layer, shard) in enumerate(zip(drafter.layers, drafter.sharding.layers), start=1):
        forward.acquire(i)
        # Gather unit i+1 in the other slot while compute reads unit i's weights.
        forward.prefetch(i + 1)
        n, input_norm = ops.rms_norm(x, layer.input_layernorm.weight, layer.input_layernorm.eps)
        attn = layer.self_attn
        q = ops.linear(n, attn.q_proj.weight).view(batch, sequence, heads, d)
        q, query_norm = ops.rms_norm(q, attn.q_norm.weight, attn.q_norm.eps)
        kc, vc = ops.linear(context, attn.k_proj.weight), ops.linear(context, attn.v_proj.weight)
        kd, vd = ops.linear(n, attn.k_proj.weight), ops.linear(n, attn.v_proj.weight)
        k = torch.cat((kc, kd), 1).view(batch, ctx + sequence, kv_heads, d)
        k, key_norm = ops.rms_norm(k, attn.k_norm.weight, attn.k_norm.eps)
        v = torch.cat((vc, vd), 1).view(batch, ctx + sequence, kv_heads, d).transpose(1, 2)
        q = ops.rope(q.transpose(1, 2), query_rope)
        k = ops.rope(k.transpose(1, 2), key_rope)
        out, attention = ops.attention(q, k, v, prepared.mask)
        output_input = out.transpose(1, 2).contiguous().reshape(batch, sequence, heads * d)
        out = ops.linear(output_input, attn.o_proj.weight)
        attention_saved = SelfAttentionActivations(n, query_norm, key_norm, attention, output_input)
        residual = x + out
        n, post_norm = ops.rms_norm(
            residual, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps
        )
        mlp = layer.mlp
        gate, up = ops.linear(n, mlp.gate_proj.weight), ops.linear(n, mlp.up_proj.weight)
        activated = ops.silu(gate)
        product = activated * up
        out = ops.linear(product, mlp.down_proj.weight)
        x = residual + out
        activations.append(
            DecoderActivations(
                input_norm,
                attention_saved,
                post_norm,
                MLPActivations(n, gate, up, activated, product),
            )
        )
        forward.release(i)

    # Frozen LM projection plus trainable Markov logits, then CE + KL.
    head_index = len(drafter.layers) + 1
    forward.acquire(head_index)
    hidden, final_norm = ops.rms_norm(x, drafter.final_norm.weight, drafter.final_norm.eps)
    logits = ops.linear(hidden, drafter.lm_head.weight).view(
        batch, blocks, o.block_size, c.vocab_size
    )
    previous = None
    if drafter.markov_head is not None:
        markov = drafter.markov_head
        previous = ops.embedding(prepared.previous, markov.markov_w1.weight)
        logits = logits + ops.linear(previous, markov.markov_w2.weight)
    if teacher_batch.logits is None:
        aligned = teacher_batch.last_hidden.to(dtype).gather(
            1, prepared.teacher_indices.unsqueeze(-1).expand(-1, -1, c.hidden_size)
        )
        target_logits = ops.linear(aligned, drafter.lm_head.weight).view_as(logits)
    else:
        target_logits = (
            teacher_batch.logits.to(dtype)
            .gather(1, prepared.teacher_indices.unsqueeze(-1).expand(-1, -1, c.vocab_size))
            .view_as(logits)
        )
    metrics, loss_saved = ops.ce_kl_loss(logits, target_logits, prepared, o)
    forward.release(head_index)
    forward.finish()

    # Heads backward: bind explicit gradient buffers, then reverse each operation.
    backward = drafter.sharding.schedule(backward=True, overlap=overlap)
    backward.prefetch(0)
    backward.acquire(0)
    backward.prefetch(1)
    drafter.sharding.heads.gradient_buffers(backward.gradient_buffer(0))
    dl = ops.ce_kl_backward(loss_saved, accumulation_steps)
    dh = dl @ drafter.lm_head.weight
    if drafter.markov_head is not None:
        dprevious = ops.linear_backward(
            dl, previous, markov.markov_w2.weight, markov.markov_w2.weight.grad
        )
        ops.embedding_backward(dprevious, prepared.previous, markov.markov_w1.weight.grad)
    dx = ops.rms_norm_backward(
        dh.reshape_as(hidden), final_norm, drafter.final_norm.weight, drafter.final_norm.weight.grad
    )
    backward.reduce(0)
    del dl, dh, loss_saved
    dc = torch.zeros_like(context)

    # Draft backward: MLP -> residual -> attention -> residual, in reverse layer order.
    for i, (layer, shard) in enumerate(
        zip(reversed(drafter.layers), reversed(drafter.sharding.layers)), start=1
    ):
        backward.acquire(i)
        backward.prefetch(i + 1)
        shard.gradient_buffers(backward.gradient_buffer(i))
        saved = activations.pop()
        mlp = layer.mlp
        dp = ops.linear_backward(
            dx, saved.mlp.product, mlp.down_proj.weight, mlp.down_proj.weight.grad
        )
        dgate = ops.silu_backward(dp * saved.mlp.up, saved.mlp.gate)
        dup = dp * saved.mlp.activated_gate
        dx_up = ops.linear_backward(
            dup, saved.mlp.input, mlp.up_proj.weight, mlp.up_proj.weight.grad
        )
        dx_gate = ops.linear_backward(
            dgate, saved.mlp.input, mlp.gate_proj.weight, mlp.gate_proj.weight.grad
        )
        dr = ops.rms_norm_backward(
            dx_up + dx_gate,
            saved.post_attention_norm,
            layer.post_attention_layernorm.weight,
            layer.post_attention_layernorm.weight.grad,
        )
        dr = dr + dx

        attn, a = layer.self_attn, saved.attention
        da = ops.linear_backward(dr, a.output_input, attn.o_proj.weight, attn.o_proj.weight.grad)
        da = da.view(batch, sequence, heads, d).transpose(1, 2)
        dq, dk, dv = ops.attention_backward(da, a.attention)
        dq = ops.rope_backward(dq, query_rope).transpose(1, 2)
        dk = ops.rope_backward(dk, key_rope).transpose(1, 2)
        dk = ops.rms_norm_backward(dk, a.key_norm, attn.k_norm.weight, attn.k_norm.weight.grad)
        dq = ops.rms_norm_backward(dq, a.query_norm, attn.q_norm.weight, attn.q_norm.weight.grad)
        dk = dk.reshape(batch, -1, kv_heads * d)
        dv = dv.transpose(1, 2).reshape_as(dk)
        # Shared K/V matrices accumulate draft and context gradients in this order.
        dx_value = ops.linear_backward(
            dv[:, ctx:], a.input, attn.v_proj.weight, attn.v_proj.weight.grad
        )
        dc_value = ops.linear_backward(
            dv[:, :ctx], context, attn.v_proj.weight, attn.v_proj.weight.grad, accumulate=True
        )
        dx_key = ops.linear_backward(
            dk[:, ctx:], a.input, attn.k_proj.weight, attn.k_proj.weight.grad
        )
        dc_key = ops.linear_backward(
            dk[:, :ctx], context, attn.k_proj.weight, attn.k_proj.weight.grad, accumulate=True
        )
        dn = dx_value + dx_key
        dc_layer = dc_value + dc_key
        dx_query = ops.linear_backward(
            dq.reshape(batch, sequence, -1), a.input, attn.q_proj.weight, attn.q_proj.weight.grad
        )
        dn = dn + dx_query
        dx = ops.rms_norm_backward(
            dn, saved.input_norm, layer.input_layernorm.weight, layer.input_layernorm.weight.grad
        )
        dx = dr + dx
        dc.add_(dc_layer)
        backward.reduce(i)
        del saved, a

    # Context projection backward, reduction, then explicit AdamW.
    backward.acquire(head_index)
    drafter.sharding.context.gradient_buffers(backward.gradient_buffer(head_index))
    dp = ops.rms_norm_backward(
        dc, context_norm, drafter.context_norm.weight, drafter.context_norm.weight.grad
    )
    dcat = ops.linear_backward(
        dp, context_input, drafter.context_proj.weight, drafter.context_proj.weight.grad
    )
    width = c.target_hidden_size
    for i, (norm, saved) in enumerate(zip(drafter.fc_norm, feature_norms)):
        ops.rms_norm_backward(
            dcat[..., i * width : (i + 1) * width], saved, norm.weight, norm.weight.grad
        )
    backward.reduce(head_index)
    backward.finish()
    update = (
        ops.adamw_step(optimizer, metrics.denominator, accumulation_steps)
        if optimizer is not None
        else None
    )
    return TrainResult(teacher_batch, metrics, hidden, update)


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--train-gpus", type=int, default=2, choices=(1, 2, 4))
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--config", default="qwen")
    p.add_argument("--output", default="outputs/train.json")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--anchors", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument(
        "--tokens",
        default="",
        help="Optional .pt tensor [samples, seq_len] of already tokenized fixed data",
    )
    p.add_argument("--target", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def config_path(name):
    path = Path(name)
    return (
        path if path.exists() else Path(__file__).resolve().parents[1] / "configs" / f"{name}.json"
    )


def setup_target(node, c):
    path = snapshot_download(
        TARGET_MODEL,
        revision=TARGET_REVISION,
        local_files_only=True,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"],
    )
    teacher = Teacher(node, path, c.target_layer_ids)
    tc = teacher.config
    if (c.target_hidden_size, c.vocab_size, c.target_num_hidden_layers) != (
        tc["hidden_size"],
        tc["vocab_size"],
        tc["num_hidden_layers"],
    ):
        raise ValueError("Draft configuration does not match the pinned teacher checkpoint")
    return path, teacher


def build_draft(node, c, dtype, seed, teacher=None):
    if not node.training:
        return None
    if teacher:
        frozen = teacher.checkpoint.frozen()
        embedding, head = (
            frozen["model.language_model.embed_tokens.weight"],
            frozen["lm_head.weight"],
        )
    else:
        gen = torch.Generator().manual_seed(seed + 1)
        embedding = torch.randn(c.vocab_size, c.hidden_size, generator=gen, dtype=dtype) * 0.1
        head = torch.randn(c.vocab_size, c.hidden_size, generator=gen, dtype=dtype) * 0.1
    state = initialize(c, seed, dtype)
    return DSpark(
        c,
        state,
        embedding,
        head,
        device=node.device,
        dtype=dtype,
        group=node.train_group,
        world_size=node.train_gpus,
        rank=node.rank,
    )


def dummy_ids(node, c, seq_len, seed):
    gen = torch.Generator(device=node.device).manual_seed(seed)
    return torch.randint(
        0, c.vocab_size, (node.train_gpus, seq_len), device=node.device, generator=gen
    )


@torch.no_grad()
def main():
    args = arguments()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    node = Node(args.train_gpus)
    torch.set_float32_matmul_precision("highest")
    c = DraftConfig.load(config_path(args.config))
    o = Objective(num_anchors=args.anchors)
    _path, teacher = setup_target(node, c)
    model = build_draft(node, c, getattr(torch, args.dtype), args.seed, teacher)
    learning_rate = args.learning_rate if args.learning_rate is not None else 1e-5
    if learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    optimizer = AdamW(model.sharding, lr=learning_rate) if model else None
    start = (
        checkpoint.restore(args.checkpoint, model, optimizer, o) if model and args.checkpoint else 0
    )
    resume_step = torch.tensor(start, device=node.device, dtype=torch.int64)
    dist.broadcast(resume_step, src=0)
    start = resume_step.item()
    data = torch.load(args.tokens, weights_only=True) if args.tokens else None
    if data is not None and (
        data.ndim != 2 or data.shape[1] != args.seq_len or len(data) < node.train_gpus
    ):
        raise ValueError("--tokens needs at least train_gpus rows with --seq-len columns")
    rows = []
    for offset in range(args.steps):
        step = start + offset
        ids = (
            dummy_ids(node, c, args.seq_len, args.seed + step)
            if data is None
            else data[(torch.arange(node.train_gpus) + step * node.train_gpus) % len(data)]
            .long()
            .to(node.device)
        )
        t0 = time.perf_counter()
        torch.manual_seed(args.seed + step * node.train_gpus + node.rank)
        result = train(ids, model, o, verifier=teacher, optimizer=optimizer)
        if node.training:
            metrics, update = result.metrics, result.update
            pooled = torch.stack((metrics.numerator, metrics.denominator))
            dist.all_reduce(pooled, group=node.train_group)
            row = {"step": step + 1, "loss": float(pooled[0] / pooled[1].clamp_min(1e-6)), **update}
            torch.cuda.synchronize()
            row["seconds"] = time.perf_counter() - t0
            rows.append(row)
            if node.rank == 0:
                print(json.dumps(row), flush=True)
        dist.barrier()
    if node.training:
        ckpt_path = str(Path(args.output).with_suffix("")) + "-checkpoint"
        ckpt_path = checkpoint.save(ckpt_path, model, optimizer, o, start + args.steps)
        if node.rank == 0:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {
                        "config": c.to_dict(),
                        "objective": asdict(o),
                        "target": TARGET_MODEL,
                        "target_revision": TARGET_REVISION,
                        "train_gpus": args.train_gpus,
                        "dtype": args.dtype,
                        "learning_rate": optimizer.lr,
                        "steps": rows,
                        "checkpoint": ckpt_path,
                    },
                    indent=2,
                )
            )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
