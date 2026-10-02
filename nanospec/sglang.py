"""Lossless local Qwen3 DSPark checkpoint interchange; no SGLang dependency."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from nanospec.config import DraftConfig
from nanospec.dspark import parameter_shapes

_NAMES = {
    "fc.weight": "context_proj.weight",
    "hidden_norm.weight": "context_norm.weight",
    "norm.weight": "final_norm.weight",
}
# Reject new architecture knobs rather than quietly accepting different execution semantics.
_CONFIG_KEYS = set(
    """
architectures model_type attention_bias attention_dropout block_size bos_token_id
confidence_head_with_markov dtype torch_dtype enable_confidence_head eos_token_id head_dim
hidden_act hidden_size initializer_range intermediate_size layer_types markov_head_type
markov_rank mask_token_id max_position_embeddings max_window_layers num_anchors
num_attention_heads num_hidden_layers num_key_value_heads num_target_layers pad_token_id
rms_norm_eps rope_parameters rope_theta rope_scaling rope_is_neox_style sliding_window
target_layer_ids tie_word_embeddings transformers_version use_cache use_sliding_window vocab_size
""".split()
)


def draft_config(config):
    """Translate the supported SGLang schema without confusing target depth and capture count."""
    unknown = config.keys() - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unsupported DSPark configuration fields: {sorted(unknown)}")
    if config.get("architectures") != ["Qwen3DSparkModel"] or config.get("model_type") != "qwen3":
        raise ValueError("Only Qwen3DSparkModel with model_type=qwen3 is supported")
    for key, default in {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "use_sliding_window": False,
        "sliding_window": None,
        "rope_scaling": None,
        "rope_is_neox_style": True,
    }.items():
        if config.get(key, default) != default:
            raise ValueError(f"Unsupported DSPark {key}={config[key]!r}")
    if config.get("layer_types", ["full_attention"] * config["num_hidden_layers"]) != (
        ["full_attention"] * config["num_hidden_layers"]
    ):
        raise ValueError("Only full-attention DSPark layers are supported")
    rope = config.get("rope_parameters") or {}
    if rope.keys() - {"rope_theta", "rope_type", "partial_rotary_factor"}:
        raise ValueError("Unsupported DSPark RoPE parameters")
    if rope.get("rope_type", "default") != "default" or rope.get("partial_rotary_factor", 1) != 1:
        raise ValueError("Only default, full-head NeoX RoPE is supported")
    theta = rope.get("rope_theta", config.get("rope_theta"))
    if theta is None or ("rope_theta" in config and config["rope_theta"] != theta):
        raise ValueError("Missing or conflicting DSPark rope_theta")
    if config["block_size"] < 1 or config["markov_rank"] <= 0:
        raise ValueError("DSPark requires positive block_size and markov_rank")
    captures = tuple(config["target_layer_ids"])
    if len(set(captures)) != len(captures) or not captures:
        raise ValueError("DSPark requires distinct explicit target_layer_ids")
    return DraftConfig(
        **{
            key: config[key]
            for key in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
                "rms_norm_eps",
                "max_position_embeddings",
                "mask_token_id",
                "markov_rank",
            )
        },
        target_hidden_size=config["hidden_size"],
        target_num_hidden_layers=config["num_target_layers"],
        num_target_layers=len(captures),
        target_layer_ids=captures,
        rope_theta=theta,
        tie_word_embeddings=config.get("tie_word_embeddings", False),
    )


def _check_tensors(tensors, shapes, label):
    missing, extra = shapes.keys() - tensors.keys(), tensors.keys() - shapes.keys()
    if missing or extra:
        raise ValueError(
            f"{label}: missing keys {sorted(missing)}; unexpected keys {sorted(extra)}"
        )
    for name, shape in shapes.items():
        value = tensors[name]
        if tuple(value.shape) != shape or value.dtype not in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ):
            raise ValueError(
                f"{label}: unsupported shape/dtype for {name}: {value.shape}/{value.dtype}"
            )


def validate_auxiliary(config, tensors):
    shapes = {}
    for name in ("embed_tokens.weight", "lm_head.weight"):
        if name in tensors:
            shapes[name] = (config["vocab_size"], config["hidden_size"])
    if config.get("enable_confidence_head", True):
        width = config["hidden_size"]
        if config.get("confidence_head_with_markov", True):
            width += config["markov_rank"]
        shapes.update(
            {"confidence_head.proj.weight": (1, width), "confidence_head.proj.bias": (1,)}
        )
    _check_tensors(tensors, shapes, "DSPark auxiliary state")


def _read_weights(path):
    index = path / "model.safetensors.index.json"
    if not index.exists():
        return load_file(path / "model.safetensors")
    mapping = json.loads(index.read_text())["weight_map"]
    weights = {}
    for filename in sorted(set(mapping.values())):
        file = (path / filename).resolve()
        if not file.is_relative_to(path.resolve()):
            raise ValueError("Safetensors index points outside the checkpoint directory")
        shard = load_file(file)
        if weights.keys() & shard.keys():
            raise ValueError("Duplicate tensors in DSPark checkpoint shards")
        if any(mapping.get(key) != filename for key in shard):
            raise ValueError("DSPark shard does not match its safetensors index")
        weights.update(shard)
    if weights.keys() != mapping.keys():
        raise ValueError("DSPark safetensors index contains missing tensors")
    return weights


def _same_config(left, right):
    return json.dumps(left.to_dict(), sort_keys=True) == json.dumps(right.to_dict(), sort_keys=True)


@dataclass
class DraftCheckpoint:
    config: DraftConfig
    state: dict
    sglang_config: dict
    auxiliary: dict

    def validate(self):
        if not _same_config(self.config, draft_config(self.sglang_config)):
            raise ValueError(
                "Nanospec configuration does not match the preserved SGLang configuration"
            )
        _check_tensors(self.state, parameter_shapes(self.config), "DSPark backbone")
        dtype = self.sglang_config.get("dtype", self.sglang_config.get("torch_dtype"))
        if self.sglang_config.get("torch_dtype", dtype) != dtype:
            raise ValueError("Conflicting DSPark dtype and torch_dtype")
        if dtype not in ("bfloat16", "float16", "float32") or any(
            value.dtype != getattr(torch, dtype) for value in self.state.values()
        ):
            raise ValueError("DSPark backbone dtype must match the preserved runtime config")
        validate_auxiliary(self.sglang_config, self.auxiliary)
        return self

    @classmethod
    def load(cls, path):
        """Read a local SGLang checkpoint or an imported nanospec interchange directory."""
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        weights = _read_weights(path)
        if (path / "sglang_config.json").exists():
            original = json.loads((path / "sglang_config.json").read_text())
            auxiliary = load_file(path / "auxiliary.safetensors")
            return cls(DraftConfig(**config), weights, original, auxiliary).validate()
        c = draft_config(config)
        shapes = parameter_shapes(c)
        state, auxiliary = {}, {}
        for name, value in weights.items():
            if name in _NAMES.values():
                raise ValueError(f"Unsupported SGLang backbone name: {name}")
            mapped = _NAMES.get(name, name)
            if mapped in shapes:
                if mapped in state:
                    raise ValueError(f"Duplicate DSPark backbone tensor: {mapped}")
                state[mapped] = value
            else:
                auxiliary[name] = value
        return cls(c, state, config, auxiliary).validate()

    @classmethod
    def from_model(cls, model):
        """Collective on all training ranks; retain the original auxiliary tensors unchanged."""
        if getattr(model, "_sglang_config", None) is None:
            raise ValueError("Export requires a model imported from a supported SGLang checkpoint")
        return cls(
            model.config,
            model.draft_state_dict(),
            model._sglang_config,
            model._sglang_auxiliary,
        ).validate()

    def build_model(self, embedding, lm_head, *, device, dtype=None, **sharding):
        """Build a trainable nanospec drafter using the actual target embedding/head."""
        from nanospec.dspark import DSpark

        self.validate()
        dtypes = {value.dtype for value in self.state.values()}
        if len(dtypes) != 1 or (dtype is not None and dtype not in dtypes):
            raise ValueError(
                "Lossless import requires a uniform backbone dtype and no dtype conversion"
            )
        model = DSpark(
            self.config,
            self.state,
            embedding,
            lm_head,
            device=device,
            dtype=dtypes.pop(),
            **sharding,
        )
        # Not parameters: the CE/KL trainer does not optimize the confidence head.
        model._sglang_config = self.sglang_config
        model._sglang_auxiliary = self.auxiliary
        return model

    def _save(self, path, *, sglang):
        self.validate()
        path = Path(path)
        # Refuse overwrites: stale shards or indices could override the new weights at runtime.
        path.mkdir(parents=True, exist_ok=False)
        if sglang:
            names = {value: key for key, value in _NAMES.items()}
            weights = {names.get(key, key): value for key, value in self.state.items()}
            weights.update(self.auxiliary)
            config = self.sglang_config
        else:
            weights = self.state
            config = self.config.to_dict()
            (path / "sglang_config.json").write_text(
                json.dumps(self.sglang_config, indent=2) + "\n"
            )
            save_file(self.auxiliary, path / "auxiliary.safetensors")
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in weights.items()},
            path / "model.safetensors",
            metadata={"format": "pt"},
        )
        (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def save_sglang(self, path):
        """Write the original runtime schema and every tensor, without casting or quantization."""
        self._save(path, sglang=True)

    def save_nanospec(self, path):
        """Persist the mapped backbone plus the original runtime config and auxiliary state."""
        self._save(path, sglang=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=("import", "export"))
    parser.add_argument("source", help="Local checkpoint directory")
    parser.add_argument("destination", help="New output directory (must not already exist)")
    args = parser.parse_args()
    checkpoint = DraftCheckpoint.load(args.source)
    if args.direction == "import":
        checkpoint.save_nanospec(args.destination)
    else:
        checkpoint.save_sglang(args.destination)


if __name__ == "__main__":
    main()
