from __future__ import annotations

import math
import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _rope_variant_utils import float_env, int_env, str_env, load_base


base = load_base("exp014_ar_latent_layerwise_rope_base")


def _round_even(value: int) -> int:
    return value if value % 2 == 0 else value - 1


def _layer_mix(layer_idx: int, num_layers: int, schedule: str) -> float:
    if num_layers <= 1:
        return 1.0
    t = layer_idx / (num_layers - 1)
    if schedule == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * t)
    return t


def _apply_layerwise_rotary(blocks, model_dim: int, num_heads: int, rope_base: float, train_seq_len: int, rope_dims: int):
    head_dim = model_dim // num_heads
    default_dims = rope_dims if rope_dims > 0 else head_dim
    min_dims = _round_even(max(2, min(head_dim, int_env("LAYERWISE_ROPE_MIN_DIMS", default_dims))))
    max_dims = _round_even(max(min_dims, min(head_dim, int_env("LAYERWISE_ROPE_MAX_DIMS", head_dim))))
    min_base = rope_base * float_env("LAYERWISE_ROPE_BASE_MIN_FACTOR", 1.0)
    max_base = rope_base * float_env("LAYERWISE_ROPE_BASE_MAX_FACTOR", 1.0)
    schedule = str_env("LAYERWISE_ROPE_SCHEDULE", "linear")
    for layer_idx, block in enumerate(blocks):
        mix = _layer_mix(layer_idx, len(blocks), schedule)
        dims = int(round(min_dims + (max_dims - min_dims) * mix))
        dims = _round_even(max(2, min(head_dim, dims)))
        layer_base = min_base + (max_base - min_base) * mix
        block.attn.rope_dims = dims
        block.attn.rotary = base.Rotary(head_dim, base=layer_base, train_seq_len=train_seq_len, rope_dims=dims)


class GPTLayerwiseRoPE(base.GPT):
    def __init__(self, *args, **kwargs):
        model_dim = kwargs["model_dim"]
        num_heads = kwargs["num_heads"]
        rope_base = kwargs["rope_base"]
        train_seq_len = kwargs.get("train_seq_len", 2048)
        rope_dims = kwargs.get("rope_dims", 0)
        super().__init__(*args, **kwargs)
        _apply_layerwise_rotary(self.blocks, model_dim, num_heads, rope_base, train_seq_len, rope_dims)


class HessianGPTLayerwiseRoPE(base._HessianGPT):
    def __init__(self, *args, **kwargs):
        model_dim = kwargs["model_dim"]
        num_heads = kwargs["num_heads"]
        rope_base = kwargs["rope_base"]
        train_seq_len = kwargs.get("train_seq_len", 2048)
        rope_dims = kwargs.get("rope_dims", 0)
        super().__init__(*args, **kwargs)
        _apply_layerwise_rotary(self.blocks, model_dim, num_heads, rope_base, train_seq_len, rope_dims)


base.GPT = GPTLayerwiseRoPE
base._HessianGPT = HessianGPTLayerwiseRoPE


if __name__ == "__main__":
    os.environ.setdefault("RUN_ID", "exp014_layerwise_rope")
    base.main()
