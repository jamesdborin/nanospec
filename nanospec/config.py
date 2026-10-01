import json
from dataclasses import asdict, dataclass

TORCHSPEC_REVISION = "076a738fdd7e64be1495338f055e8188784633c4"
TARGET_MODEL = "Qwen/Qwen3.5-27B-FP8"
TARGET_REVISION = "709d2d2a46f204254221f583f45e00f93b578c9e"


@dataclass
class DraftConfig:
    hidden_size: int = 5120
    intermediate_size: int = 15360
    num_hidden_layers: int = 5
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 248320
    target_hidden_size: int = 5120
    target_num_hidden_layers: int = 64
    num_target_layers: int = 5
    target_layer_ids: tuple[int, ...] = (1, 16, 31, 46, 61)
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    max_position_embeddings: int = 262144
    mask_token_id: int = 248063  # <|fim_pad|> in the pinned Qwen3.5 tokenizer
    markov_rank: int = 256
    markov_head_type: str = "vanilla"
    fc_norm: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.hidden_size != self.target_hidden_size:
            raise ValueError("Shared target embedding/head require equal target and draft width")
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2:
            raise ValueError("GQA heads must divide query heads; RoPE head_dim must be even")
        if len(self.target_layer_ids) != self.num_target_layers:
            raise ValueError("Wrong number of target layer IDs")
        if any(i < 0 or i >= self.target_num_hidden_layers for i in self.target_layer_ids):
            raise ValueError("Target layer IDs must index decoder outputs (zero based)")
        if self.markov_head_type != "vanilla" or self.markov_rank < 0:
            raise ValueError("Only vanilla Markov heads are supported")
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError("mask_token_id outside vocabulary")

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

    def to_dict(self):
        return asdict(self)


@dataclass
class Objective:
    block_size: int = 7
    num_anchors: int = 32
    gamma: float = 4.0
    ce: float = 0.1
    kl: float = 0.9

    def __post_init__(self):
        if self.block_size < 1 or self.num_anchors < 1:
            raise ValueError("block_size and num_anchors must be positive")
        if min(self.ce, self.kl) < 0:
            raise ValueError("Loss weights must be nonnegative")
