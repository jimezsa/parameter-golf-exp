from __future__ import annotations

import argparse
import io
import lzma
import math
from dataclasses import dataclass

import torch

from ternary import pack_ternary_values


@dataclass(frozen=True)
class EntropyConfig:
    num_layers: int
    model_dim: int
    num_heads: int
    num_kv_heads: int
    mlp_mult: float
    zero_prob: float
    seed: int
    max_bytes_per_param: float
    report_formats: tuple[str, ...]
    blocking_format: str

    @property
    def head_dim(self) -> int:
        return self.model_dim // self.num_heads

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def mlp_dim(self) -> int:
        return int(self.mlp_mult * self.model_dim)


def parse_args() -> EntropyConfig:
    parser = argparse.ArgumentParser(description="Experiment 004 ternary LZMA entropy estimator.")
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--mlp-mult", type=float, default=4.0)
    parser.add_argument("--zero-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-bytes-per-param", type=float, default=0.20)
    parser.add_argument("--pack-format", choices=("base3", "two_bit", "both"), default="both")
    parser.add_argument("--blocking-format", choices=("base3", "two_bit"), default="base3")
    args = parser.parse_args()
    report_formats = ("base3", "two_bit") if args.pack_format == "both" else (args.pack_format,)
    if args.blocking_format not in report_formats:
        raise ValueError("blocking-format must be included in the selected pack-format set")
    return EntropyConfig(
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        zero_prob=args.zero_prob,
        seed=args.seed,
        max_bytes_per_param=args.max_bytes_per_param,
        report_formats=report_formats,
        blocking_format=args.blocking_format,
    )


def iter_ternary_weight_specs(cfg: EntropyConfig) -> list[tuple[str, tuple[int, int]]]:
    specs: list[tuple[str, tuple[int, int]]] = []
    for layer_idx in range(cfg.num_layers):
        specs.extend(
            [
                (f"blocks.{layer_idx}.attn.c_q.weight", (cfg.model_dim, cfg.model_dim)),
                (f"blocks.{layer_idx}.attn.proj.weight", (cfg.model_dim, cfg.model_dim)),
                (f"blocks.{layer_idx}.attn.c_k.weight", (cfg.kv_dim, cfg.model_dim)),
                (f"blocks.{layer_idx}.attn.c_v.weight", (cfg.kv_dim, cfg.model_dim)),
                (f"blocks.{layer_idx}.mlp.fc.weight", (cfg.mlp_dim, cfg.model_dim)),
                (f"blocks.{layer_idx}.mlp.proj.weight", (cfg.model_dim, cfg.mlp_dim)),
            ]
        )
    return specs


def sample_ternary(shape: tuple[int, int], zero_prob: float, generator: torch.Generator) -> torch.Tensor:
    if not 0.0 <= zero_prob <= 1.0:
        raise ValueError(f"zero_prob must be in [0, 1], got {zero_prob}")
    values = torch.randint(0, 2, shape, generator=generator, dtype=torch.int64)
    values = values.mul_(2).sub_(1).to(dtype=torch.int8)
    if zero_prob > 0.0:
        zero_mask = torch.rand(shape, generator=generator) < zero_prob
        values[zero_mask] = 0
    return values


def shannon_entropy_bits(neg_count: int, zero_count: int, pos_count: int) -> float:
    total = neg_count + zero_count + pos_count
    entropy = 0.0
    for count in (neg_count, zero_count, pos_count):
        if count <= 0:
            continue
        prob = count / total
        entropy -= prob * math.log2(prob)
    return entropy


def serialize_payload(payload: dict[str, object]) -> tuple[int, int]:
    buf = io.BytesIO()
    torch.save(payload, buf)
    raw = buf.getvalue()
    compressed = lzma.compress(raw, preset=9)
    return len(raw), len(compressed)


def main() -> None:
    cfg = parse_args()
    if cfg.model_dim % cfg.num_heads != 0:
        raise ValueError("model_dim must be divisible by num_heads")
    if cfg.num_heads % cfg.num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    specs = iter_ternary_weight_specs(cfg)
    total_params = sum(math.prod(shape) for _, shape in specs)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(cfg.seed)

    payloads = {fmt: {"w": {}, "m": {}} for fmt in cfg.report_formats}
    neg_count = 0
    zero_count = 0
    pos_count = 0

    for name, shape in specs:
        ternary = sample_ternary(shape, cfg.zero_prob, generator)
        neg_count += int((ternary == -1).sum().item())
        zero_count += int((ternary == 0).sum().item())
        pos_count += int((ternary == 1).sum().item())
        scale_key = f"{name}.scale"
        for pack_format in cfg.report_formats:
            packed_key = f"{name}.q"
            payloads[pack_format]["w"][packed_key] = pack_ternary_values(ternary, pack_format=pack_format).cpu()
            payloads[pack_format]["w"][scale_key] = torch.tensor([1.0], dtype=torch.float32)
            payloads[pack_format]["m"][name] = {
                "type": "ternary",
                "shape": shape,
                "pack_format": pack_format,
            }

    entropy_bits = shannon_entropy_bits(neg_count, zero_count, pos_count)
    print("Experiment 004 LZMA entropy estimator")
    print(
        "config:",
        f"layers={cfg.num_layers}",
        f"model_dim={cfg.model_dim}",
        f"num_heads={cfg.num_heads}",
        f"num_kv_heads={cfg.num_kv_heads}",
        f"mlp_mult={cfg.mlp_mult}",
        f"mlp_dim={cfg.mlp_dim}",
        f"total_ternary_params={total_params}",
        f"blocking_format={cfg.blocking_format}",
        f"threshold_bpp={cfg.max_bytes_per_param:.4f}",
    )
    print(
        "distribution:",
        f"neg={neg_count / total_params:.4f}",
        f"zero={zero_count / total_params:.4f}",
        f"pos={pos_count / total_params:.4f}",
        f"entropy_bits_per_param={entropy_bits:.4f}",
        f"entropy_bytes_per_param={entropy_bits / 8.0:.4f}",
    )

    format_results: dict[str, float] = {}
    for pack_format in cfg.report_formats:
        payload = payloads[pack_format]
        raw_tensor_bytes = sum(t.numel() * t.element_size() for t in payload["w"].values())
        torchsave_bytes, compressed_bytes = serialize_payload(payload)
        bytes_per_param = compressed_bytes / total_params
        format_results[pack_format] = bytes_per_param
        print(
            f"{pack_format}:",
            f"raw_tensor_bytes={raw_tensor_bytes}",
            f"raw_tensor_bpp={raw_tensor_bytes / total_params:.4f}",
            f"torchsave_bytes={torchsave_bytes}",
            f"torchsave_bpp={torchsave_bytes / total_params:.4f}",
            f"lzma_bytes={compressed_bytes}",
            f"lzma_bpp={bytes_per_param:.4f}",
        )

    blocking_bpp = format_results[cfg.blocking_format]
    if blocking_bpp > cfg.max_bytes_per_param:
        raise SystemExit(
            f"ABORT: {cfg.blocking_format} packs to {blocking_bpp:.4f} bytes/param, "
            f"above the {cfg.max_bytes_per_param:.4f} limit. Downsize before burning a run."
        )

    print(
        f"PASS: {cfg.blocking_format} packs to {blocking_bpp:.4f} bytes/param, "
        f"within the {cfg.max_bytes_per_param:.4f} limit."
    )


if __name__ == "__main__":
    main()
