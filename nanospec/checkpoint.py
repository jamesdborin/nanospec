"""Rank-local explicit optimizer checkpoints; a checkpoint is complete at metadata.json."""

import json
import uuid
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from nanospec.config import TARGET_MODEL, TARGET_REVISION


def save(path, model, optimizer, objective, step):
    first = model.sharding.context
    # Never overwrite a complete checkpoint with a partially written generation.
    tag = [uuid.uuid4().hex[:12] if first.rank == 0 else None]
    if first.world_size > 1:
        dist.broadcast_object_list(tag, src=0, group=first.group)
    path = Path(f"{path}-step{step}-{tag[0]}")
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "units": {
            name: {field: getattr(unit, field).cpu() for field in ("param", "master", "m", "v")}
            for name, unit in model.sharding.named_shards()
        },
        "optimizer_steps": optimizer.steps,
        "rng": torch.cuda.get_rng_state(),
        "step": step,
    }
    tmp = path / f"rank-{first.rank}.tmp"
    torch.save(payload, tmp)
    tmp.replace(path / f"rank-{first.rank}.pt")
    if first.world_size > 1:
        dist.barrier(group=first.group)
    if first.rank == 0:
        sglang_config = getattr(model, "_sglang_config", None)
        if sglang_config is not None:
            from nanospec.sglang import validate_auxiliary

            validate_auxiliary(sglang_config, model._sglang_auxiliary)
            save_file(model._sglang_auxiliary, path / "sglang_auxiliary.safetensors")
        metadata = {
            "config": model.config.to_dict(),
            "objective": asdict(objective),
            "world_size": first.world_size,
            "step": step,
            "target": TARGET_MODEL,
            "target_revision": TARGET_REVISION,
            "dtype": str(first.param.dtype),
            "optimizer": {
                k: getattr(optimizer, k)
                for k in ("lr", "weight_decay", "betas", "eps", "max_grad_norm")
            },
        }
        if sglang_config is not None:
            metadata["sglang_config"] = sglang_config
        temp = path / "metadata.tmp"
        temp.write_text(json.dumps(metadata, indent=2))
        temp.replace(path / "metadata.json")
    if first.world_size > 1:
        dist.barrier(group=first.group)
    return str(path)


def restore(path, model, optimizer, objective):
    first = model.sharding.context
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    if meta["world_size"] != first.world_size or meta["dtype"] != str(first.param.dtype):
        raise ValueError("Resume currently requires the same training GPU count and dtype")
    if json.dumps(meta["config"], sort_keys=True) != json.dumps(
        model.config.to_dict(), sort_keys=True
    ):
        raise ValueError("Checkpoint configuration mismatch")
    if meta["objective"] != asdict(objective):
        raise ValueError("Checkpoint objective mismatch")
    if meta["target"] != TARGET_MODEL or meta["target_revision"] != TARGET_REVISION:
        raise ValueError("Checkpoint teacher revision mismatch")
    sglang_config = meta.get("sglang_config")
    auxiliary = {}
    if sglang_config is not None:
        from nanospec.sglang import draft_config, validate_auxiliary

        if json.dumps(draft_config(sglang_config).to_dict(), sort_keys=True) != json.dumps(
            model.config.to_dict(), sort_keys=True
        ):
            raise ValueError("Checkpoint SGLang configuration mismatch")
        auxiliary = load_file(path / "sglang_auxiliary.safetensors")
        validate_auxiliary(sglang_config, auxiliary)
    state = torch.load(path / f"rank-{first.rank}.pt", map_location="cpu", weights_only=True)
    if state["step"] != meta["step"]:
        raise ValueError("Checkpoint rank file does not match its metadata")
    for name, unit in model.sharding.named_shards():
        for field, value in state["units"][name].items():
            getattr(unit, field).copy_(value)
    for key, value in meta["optimizer"].items():
        setattr(optimizer, key, value)
    optimizer.steps = state["optimizer_steps"]
    model._sglang_config, model._sglang_auxiliary = sglang_config, auxiliary
    torch.cuda.set_rng_state(state["rng"])
    return state["step"]
