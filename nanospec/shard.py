"""Explicit FULL_SHARD with two reusable communication slots and CUDA dependencies."""

import math
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


@dataclass
class ShardingPlan:
    context: "Shard"
    layers: tuple["Shard", ...]
    heads: "Shard"
    _pipeline: "DoubleBuffer | None" = field(default=None, init=False, repr=False)

    def schedule(self, *, backward=False, overlap=True):
        units = tuple(self)
        if backward:
            units = tuple(reversed(units))
        first = self.context
        if not overlap or first.world_size == 1:
            return SerialSchedule(units)
        if self._pipeline is None:
            self._pipeline = DoubleBuffer(tuple(self))
        self._pipeline.begin(units)
        return self._pipeline

    def reset_pipeline(self):
        if self._pipeline is not None:
            self._pipeline.close()
            self._pipeline = None

    def named_shards(self):
        yield "context", self.context
        for i, shard in enumerate(self.layers):
            yield f"layers.{i}", shard
        yield "heads", self.heads

    def __iter__(self):
        for _, shard in self.named_shards():
            yield shard


class Shard:
    @torch.no_grad()
    def __init__(self, named_parameters, *, device, dtype, group=None, world_size=1, rank=0):
        if world_size not in (1, 2, 4):
            raise ValueError("Training supports 1, 2, or 4 GPUs")
        self.group, self.world_size, self.rank = group, world_size, rank
        # Registered nn.Parameters remain the public layer interface. Their .data
        # and .grad point into explicit gathered buffers only while the unit is live.
        tensors = dict(named_parameters)
        self.parameters = tuple(tensors.values())
        if any(p.requires_grad for p in self.parameters):
            raise ValueError("Manual sharding requires requires_grad=False parameters")
        self.names = list(tensors)
        self.shapes = {k: v.shape for k, v in tensors.items()}
        self.offsets, n = {}, 0
        for k, v in tensors.items():
            self.offsets[k] = (n, n + v.numel())
            n += v.numel()
        self.numel = n
        self.shard_size = math.ceil(n / world_size)
        start = rank * self.shard_size
        # Construct only this rank's persistent shard, without a full GPU copy.
        self.param = torch.zeros(self.shard_size, device=device, dtype=dtype)
        for k, tensor in tensors.items():
            lo, hi = self.offsets[k]
            a, b = max(start, lo), min(start + self.shard_size, hi)
            if a < b:
                self.param[a - start : b - start].copy_(tensor.reshape(-1)[a - lo : b - lo])
        self.master = self.param.float().clone()
        self.grad = torch.zeros_like(self.param)
        self.optimizer_grad = torch.zeros_like(self.master)
        self.m = torch.zeros_like(self.master)
        self.v = torch.zeros_like(self.master)
        self.full = None
        self.full_grad = None
        self.empty = torch.empty(0, device=device, dtype=dtype)
        if self.world_size == 1:
            self._bind_parameters(self.param)
        else:
            self.release()

    def _bind_parameters(self, flat):
        for name, parameter in zip(self.names, self.parameters):
            a, b = self.offsets[name]
            parameter.data = flat[a:b].view(self.shapes[name])

    @torch.no_grad()
    def apply_placement(self, fn):
        """Follow Module.to()/cuda()/cpu() while keeping optimizer state in FP32."""
        if self.full is not None or self.full_grad is not None:
            raise RuntimeError("Release gathered buffers before moving a manually sharded module")
        self.param, self.grad, self.empty = map(fn, (self.param, self.grad, self.empty))
        for name in ("master", "optimizer_grad", "m", "v"):
            setattr(self, name, getattr(self, name).to(device=self.param.device))
        if self.world_size == 1:
            self._bind_parameters(self.param)
        else:
            self.release()

    @torch.no_grad()
    def gather(self):
        if self.full is not None:
            raise RuntimeError("Gathered weights must be released before another gather")
        if self.world_size == 1:
            self.full = self.param
        else:
            self.full = torch.empty(
                self.shard_size * self.world_size, device=self.param.device, dtype=self.param.dtype
            )
            dist.all_gather_into_tensor(self.full, self.param, group=self.group)
        self._bind_parameters(self.full)
        return dict(zip(self.names, self.parameters))

    def release(self):
        for parameter in self.parameters:
            parameter.grad = None
            if self.world_size > 1:
                parameter.data = self.empty
        self.full = None

    def gradient_buffers(self, buffer=None):
        if self.full is None:
            raise RuntimeError("Gather weights before binding gradient buffers")
        self.full_grad = (
            torch.empty(
                self.shard_size * self.world_size, device=self.param.device, dtype=self.param.dtype
            )
            if buffer is None
            else buffer
        )
        self.full_grad.zero_()
        for name, parameter in zip(self.names, self.parameters):
            a, b = self.offsets[name]
            parameter.grad = self.full_grad[a:b].view(self.shapes[name])

    @torch.no_grad()
    def reduce(self):
        if self.world_size == 1:
            self.grad.add_(self.full_grad)
        else:
            out = torch.empty_like(self.param)
            # PyTorch 2.10 FSDP2 uses NCCL AVG for both FP32 and BF16.
            dist.reduce_scatter_tensor(out, self.full_grad, op=dist.ReduceOp.AVG, group=self.group)
            self.grad.add_(out)
        self.full_grad = None
        self.release()


class SerialSchedule:
    """The original schedule, retained for paired measurements and the one-GPU path."""

    def __init__(self, units):
        self.units = units

    def prefetch(self, index):
        pass

    def acquire(self, index):
        self.units[index].gather()

    def gradient_buffer(self, index):
        return None

    def release(self, index):
        self.units[index].release()

    def reduce(self, index):
        self.units[index].reduce()

    def finish(self):
        pass


class BufferSlot:
    def __init__(self, size, first):
        self.weights = torch.empty(size, device=first.param.device, dtype=first.param.dtype)
        self.gradients = torch.empty_like(self.weights)
        self.reduced = torch.empty(
            size // first.world_size, device=first.param.device, dtype=first.param.dtype
        )
        self.ready = torch.cuda.Event()
        self.free = torch.cuda.Event()
        self.reduced_done = torch.cuda.Event()
        self.has_free = self.has_reduction = False
        self.owner = None
        self.bound = False
        self.work = []


class DoubleBuffer:
    """One NCCL launch stream, two weight/gradient slots, explicit reuse dependencies.

    All ranks enqueue collectives in the same order. Work.wait() below inserts a
    CUDA dependency on the communication stream; it does not synchronize the CPU.
    Persistent buffers and retained Work objects keep asynchronous inputs alive.
    """

    def __init__(self, units):
        first = units[0]
        self.device = first.param.device
        size = max(s.shard_size * s.world_size for s in units)
        self.slots = [BufferSlot(size, first) for _ in range(2)]
        self.stream = torch.cuda.Stream(device=self.device)
        for slot in self.slots:
            for buffer in (slot.weights, slot.gradients, slot.reduced):
                buffer.record_stream(self.stream)
        for shard in units:
            shard.param.record_stream(self.stream)
            shard.grad.record_stream(self.stream)
        self.done = torch.cuda.Event()
        self.active = False

    def begin(self, units):
        if self.active:
            raise RuntimeError("Finish the communication schedule before starting another")
        self.units = units
        self.compute = torch.cuda.current_stream(self.device)
        # Optimizer writes from the previous step must precede reads of local shards.
        self.stream.wait_stream(self.compute)
        self.active = True

    def prefetch(self, index):
        if index >= len(self.units):
            return
        shard, slot = self.units[index], self.slots[index % 2]
        if slot.bound or slot.owner is not None:
            raise RuntimeError("Release a slot before reusing it")
        slot.owner = index
        slot.work.clear()
        with torch.cuda.stream(self.stream):
            if slot.has_free:
                self.stream.wait_event(slot.free)
            size = shard.shard_size * shard.world_size
            work = dist.all_gather_into_tensor(
                slot.weights[:size], shard.param, group=shard.group, async_op=True
            )
            work.wait()
            slot.work.append(work)
            slot.ready.record(self.stream)

    def acquire(self, index):
        shard, slot = self.units[index], self.slots[index % 2]
        if slot.owner != index or slot.bound or shard.full is not None:
            raise RuntimeError("Acquire exactly one prefetched unit")
        self.compute.wait_event(slot.ready)
        slot.weights.record_stream(self.compute)
        shard.full = slot.weights[: shard.shard_size * shard.world_size]
        shard._bind_parameters(shard.full)
        slot.bound = True

    def gradient_buffer(self, index):
        shard, slot = self.units[index], self.slots[index % 2]
        if slot.owner != index or not slot.bound:
            raise RuntimeError("Acquire weights before requesting their gradient buffer")
        if slot.has_reduction:
            self.compute.wait_event(slot.reduced_done)
        slot.gradients.record_stream(self.compute)
        return slot.gradients[: shard.shard_size * shard.world_size]

    def release(self, index):
        shard, slot = self.units[index], self.slots[index % 2]
        if slot.owner != index or not slot.bound:
            raise RuntimeError("Release exactly one acquired unit")
        # Future writes into this slot must wait for every compute-side weight read.
        slot.free.record(self.compute)
        slot.has_free = True
        shard.release()
        slot.owner, slot.bound = None, False

    def reduce(self, index):
        shard, slot = self.units[index], self.slots[index % 2]
        gradients = shard.full_grad
        self.release(index)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(slot.free)
            out = slot.reduced[: shard.shard_size]
            work = dist.reduce_scatter_tensor(
                out, gradients, op=dist.ReduceOp.AVG, group=shard.group, async_op=True
            )
            work.wait()
            slot.work.append(work)
            shard.grad.add_(out)
            slot.reduced_done.record(self.stream)
            slot.has_reduction = True
        shard.full_grad = None

    def finish(self):
        if any(s.bound or s.owner is not None for s in self.slots):
            raise RuntimeError("All scheduled units must be consumed before finishing")
        self.stream.wait_stream(self.compute)
        self.done.record(self.stream)
        # Raw-gradient readers, clipping and AdamW must see completed reductions.
        self.compute.wait_event(self.done)
        self.active = False

    def close(self):
        if self.active:
            raise RuntimeError("Cannot move a model with an active communication schedule")
        self.stream.synchronize()


class AdamW:
    """FP32 master parameters/moments, global clipping, no torch.optim calls."""

    def __init__(
        self, shards, lr=6e-4, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8, max_grad_norm=1.0
    ):
        self.shards = list(shards)
        self.lr, self.weight_decay, self.betas = lr, weight_decay, betas
        self.eps, self.max_grad_norm, self.steps = eps, max_grad_norm, 0


@torch.no_grad()
def adamw_step(optimizer, local_denominator, accumulation_steps=1):
    first = optimizer.shards[0]
    denominator = local_denominator.clone().float()
    if first.world_size > 1:
        dist.all_reduce(denominator, group=first.group)
    factor = torch.where(
        denominator > 0,
        first.world_size * accumulation_steps / denominator.clamp_min(1e-6),
        0.0,
    )
    norm2 = torch.zeros_like(denominator)
    for shard in optimizer.shards:
        shard.grad.mul_(factor)
        shard.optimizer_grad.copy_(shard.grad)
        norm2.add_(shard.optimizer_grad.square().sum())
    if first.world_size > 1:
        dist.all_reduce(norm2, group=first.group)
    norm = norm2.sqrt()
    if not torch.isfinite(norm).item():
        zero_grad(optimizer)
        return {"grad_norm": float(norm), "updated": False, "denominator": float(denominator)}
    clip = (optimizer.max_grad_norm / (norm + 1e-6)).clamp(max=1.0)
    optimizer.steps += 1
    b1, b2 = optimizer.betas
    for s in optimizer.shards:
        grad = s.optimizer_grad.mul_(clip)
        s.master.mul_(1 - optimizer.lr * optimizer.weight_decay)
        s.m.lerp_(grad, 1 - b1)
        s.v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
        denom = s.v.sqrt().div_(math.sqrt(1 - b2**optimizer.steps)).add_(optimizer.eps)
        s.master.addcdiv_(s.m, denom, value=-optimizer.lr / (1 - b1**optimizer.steps))
        s.param.copy_(s.master)
    zero_grad(optimizer)
    return {"grad_norm": float(norm), "updated": True, "denominator": float(denominator)}


def zero_grad(optimizer):
    for shard in optimizer.shards:
        shard.grad.zero_()
