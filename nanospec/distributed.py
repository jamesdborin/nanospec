"""One process per GPU, synchronous teacher pipeline then NCCL feature transfer."""

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn

from nanospec.qwen import Checkpoint, QwenStage


class Node:
    def __init__(self, train_gpus):
        if train_gpus not in (1, 2, 4):
            raise ValueError("--train-gpus must be 1, 2, or 4")
        self.rank = int(os.environ["RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        if self.world != 8:
            raise ValueError("Launch with torchrun --nproc_per_node=8 on an NVLink node")
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(self.device)
        dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=self.device)
        self.train_gpus = train_gpus
        self.train_group = dist.new_group(list(range(train_gpus)), backend="nccl")
        self.training = self.rank < train_gpus
        self.inference_ranks = list(range(train_gpus, 8))

    def broadcast(self, value, shape, src, dtype=torch.bfloat16):
        if self.rank == src:
            tensor = value.contiguous()
            if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
                raise ValueError("Communication buffer does not match declared shape/dtype")
        else:
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
        dist.broadcast(tensor, src)
        return tensor


class Teacher(nn.Module):
    def __init__(self, node, path, capture):
        super().__init__()
        self.node = node
        self.checkpoint = Checkpoint(path)
        self.config = self.checkpoint.config["text_config"]
        self.capture = tuple(capture)
        n = self.config["num_hidden_layers"]
        ranks = node.inference_ranks
        self.partitions = {
            r: (i * n // len(ranks), (i + 1) * n // len(ranks)) for i, r in enumerate(ranks)
        }
        self.stage = None
        if node.rank in ranks:
            start, end = self.partitions[node.rank]
            self.stage = QwenStage(
                self.checkpoint,
                start,
                end,
                device=node.device,
                first=node.rank == ranks[0],
                last=node.rank == ranks[-1],
            )


@dataclass
class TeacherBatch:
    hidden_states: list[torch.Tensor]
    last_hidden: torch.Tensor
    logits: torch.Tensor | None = None
