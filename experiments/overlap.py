"""Paired overlap/serial measurements on fixed teacher tensors, not a test suite."""

import hashlib
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from nanospec import ops
from nanospec.config import DraftConfig, Objective
from nanospec.distributed import Node, TeacherBatch
from nanospec.shard import AdamW
from nanospec.train import arguments, build_draft, config_path, train


def digest(tensors):
    result = hashlib.sha256()
    for tensor in tensors:
        data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
        result.update(memoryview(data))
    return result.hexdigest()


@torch.no_grad()
def measure(node, args):
    c = DraftConfig.load(config_path(args.config))
    dtype = getattr(torch, args.dtype)
    o = Objective(num_anchors=args.anchors)
    model = build_draft(node, c, dtype, args.seed)
    opt = AdamW(model.sharding, lr=1e-5, weight_decay=0.01)
    initial = [s.param.cpu().clone() for s in model.sharding]
    gen = torch.Generator(device=node.device).manual_seed(args.seed + 1000 + node.rank)
    ids = torch.randint(c.vocab_size, (1, args.seq_len), device=node.device, generator=gen)
    states = [
        torch.randn(1, args.seq_len, c.hidden_size, device=node.device, dtype=dtype, generator=gen)
        for _ in c.target_layer_ids
    ]
    last = torch.randn(
        1, args.seq_len, c.hidden_size, device=node.device, dtype=dtype, generator=gen
    )
    batch = TeacherBatch(states, last)
    mask = torch.ones_like(ids, dtype=torch.float32)
    if node.rank % 2:
        mask[:, args.seq_len // 2 : args.seq_len // 2 + 2] = 0

    def reset():
        torch.cuda.synchronize()
        for shard, value in zip(model.sharding, initial):
            shard.param.copy_(value)
            shard.master.copy_(shard.param)
            shard.grad.zero_()
            shard.m.zero_()
            shard.v.zero_()
            shard.optimizer_grad.zero_()
        opt.steps = 0
        torch.cuda.synchronize()

    def step(overlap, update=True):
        torch.manual_seed(args.seed + node.rank)
        return train(
            ids,
            model,
            o,
            teacher_batch=batch,
            loss_mask=mask,
            optimizer=opt if update else None,
            overlap=overlap,
        )

    # Hash every element, including raw reduced gradients, master weights and moments.
    # The same initial checkpoint and anchors are used by both schedules.
    equality = []
    for overlap in (False, True):
        reset()
        rows = []
        for i in range(3):
            result = step(overlap, update=False)
            gradients = digest(s.grad for s in model.sharding)
            update = ops.adamw_step(opt, result.metrics.denominator)
            rows.append(
                {
                    "step": i + 1,
                    "loss": result.metrics.loss.item(),
                    "hidden": digest([result.hidden]),
                    "gradients": gradients,
                    "optimizer": update,
                    "state": digest(
                        getattr(s, field)
                        for s in model.sharding
                        for field in ("param", "master", "m", "v")
                    ),
                }
            )
        equality.append(rows)
    identical = equality[0] == equality[1]
    print(json.dumps({"rank": node.rank, "bitwise_identical": identical}), flush=True)

    # ABBA ordering, same starting state for each window, warmups outside measurements.
    windows = []
    for overlap in (False, True, True, False):
        reset()
        for _ in range(3):
            step(overlap)
        torch.cuda.synchronize()
        dist.barrier(group=node.train_group)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.steps):
            dist.barrier(group=node.train_group)
            torch.cuda.synchronize()
            start = time.perf_counter()
            step(overlap)
            torch.cuda.synchronize()
            elapsed = torch.tensor((time.perf_counter() - start) * 1000, device=node.device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=node.train_group)
            times.append(elapsed.item())
        windows.append(
            {
                "overlap": overlap,
                "step_ms": times,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        )
        if node.rank == 0:
            print(json.dumps({"window": len(windows), **windows[-1]}), flush=True)

    # One separate profiler step proves that CUDA kernels actually overlap.
    trace_path = str(Path(args.output).with_suffix("")) + "-trace.json"
    if node.rank == 0:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            step(True)
        profile.export_chrome_trace(trace_path)
    else:
        step(True)
    dist.barrier(group=node.train_group)
    report = {
        "rank": node.rank,
        "identical": identical,
        "serial": equality[0],
        "overlap": equality[1],
        "windows": windows,
    }
    ranks = [None] * node.train_gpus
    dist.all_gather_object(ranks, report, group=node.train_group)
    if node.rank == 0:
        serial = [t for w in windows if not w["overlap"] for t in w["step_ms"]]
        overlapped = [t for w in windows if w["overlap"] for t in w["step_ms"]]
        serial_ms, overlap_ms = statistics.median(serial), statistics.median(overlapped)
        trace = json.loads(Path(trace_path).read_text())
        kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel" and "dur" in e]
        compute = [e for e in kernels if "nccl" not in e.get("name", "").lower()]
        communication = [e for e in kernels if "nccl" in e.get("name", "").lower()]
        intersections = []
        for comm in communication:
            spans = sorted(
                (
                    max(comm["ts"], comp["ts"]),
                    min(comm["ts"] + comm["dur"], comp["ts"] + comp["dur"]),
                )
                for comp in compute
                if comp.get("args", {}).get("stream") != comm.get("args", {}).get("stream")
                and comp["ts"] < comm["ts"] + comm["dur"]
                and comm["ts"] < comp["ts"] + comp["dur"]
            )
            # Union intervals so simultaneous compute kernels are counted only once.
            total, end = 0, float("-inf")
            for lo, hi in spans:
                total += max(0, hi - max(lo, end))
                end = max(end, hi)
            intersections.append(
                {"kernel": comm["name"], "duration_us": comm["dur"], "compute_overlap_us": total}
            )
        result = {
            "passed": all(r["identical"] for r in ranks) and overlap_ms < serial_ms,
            "bitwise_identical": all(r["identical"] for r in ranks),
            "serial_median_ms": serial_ms,
            "overlap_median_ms": overlap_ms,
            "speedup": serial_ms / overlap_ms,
            "time_reduction_percent": 100 * (1 - overlap_ms / serial_ms),
            "config": c.to_dict(),
            "objective": asdict(o),
            "train_gpus": node.train_gpus,
            "dtype": args.dtype,
            "sequence_length": args.seq_len,
            "timed_steps_per_window": args.steps,
            "measurement": "Full draft forward, backward and AdamW; fixed teacher tensors; 3 warmups/window; ABBA order; synchronized wall time, maximum across ranks; inference excluded",
            "trace": trace_path,
            "communication_compute_intersections": intersections,
            "ranks": ranks,
        }
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k
                    not in ("ranks", "communication_compute_intersections", "config", "objective")
                }
            ),
            flush=True,
        )


def main():
    args = arguments()
    if args.steps < 3:
        raise ValueError("Use at least three measured steps per timing window")
    node = Node(args.train_gpus)
    torch.set_float32_matmul_precision("highest")
    if node.training:
        measure(node, args)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
