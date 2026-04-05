from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ternary import (
    TernaryLinear,
    pack_ternary_values,
    pack_ternary_weight,
    quantize_ternary_weight,
    unpack_ternary_values,
    unpack_ternary_weight,
)


@dataclass
class BenchmarkConfig:
    steps: int
    warmup_steps: int
    batch_size: int
    seq_len: int
    vocab_size: int
    num_layers: int
    model_dim: int
    num_heads: int
    num_kv_heads: int
    mlp_mult: float
    lr: float
    weight_decay: float
    device: str
    dtype: str
    seed: int

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * (self.model_dim // self.num_heads)

    @property
    def mlp_dim(self) -> int:
        return int(self.mlp_mult * self.model_dim)


class ProjectionBlock(nn.Module):
    def __init__(self, model_dim: int, kv_dim: int, mlp_dim: int, linear_cls: type[nn.Module]):
        super().__init__()
        self.model_dim = model_dim
        self.kv_dim = kv_dim
        self.q_proj = linear_cls(model_dim, model_dim, bias=False)
        self.k_proj = linear_cls(model_dim, kv_dim, bias=False)
        self.v_proj = linear_cls(model_dim, kv_dim, bias=False)
        self.out_proj = linear_cls(model_dim, model_dim, bias=False)
        self.up_proj = linear_cls(model_dim, mlp_dim, bias=False)
        self.down_proj = linear_cls(mlp_dim, model_dim, bias=False)

    def _expand_kv(self, x: Tensor) -> Tensor:
        if self.kv_dim == self.model_dim:
            return x
        return F.pad(x, (0, self.model_dim - self.kv_dim))

    def forward(self, x: Tensor) -> Tensor:
        normed = F.rms_norm(x, (x.size(-1),))
        q = self.q_proj(normed)
        k = self._expand_kv(self.k_proj(normed))
        v = self._expand_kv(self.v_proj(normed))
        attn_like = torch.tanh(q + k + v)
        x = x + self.out_proj(attn_like)
        mlp_in = F.rms_norm(x, (x.size(-1),))
        mlp_hidden = F.leaky_relu(self.up_proj(mlp_in), negative_slope=0.5)
        return x + self.down_proj(mlp_hidden.square())


class DummyGPTLatencyModel(nn.Module):
    def __init__(self, cfg: BenchmarkConfig, linear_cls: type[nn.Module]):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.blocks = nn.ModuleList(
            [ProjectionBlock(cfg.model_dim, cfg.kv_dim, cfg.mlp_dim, linear_cls) for _ in range(cfg.num_layers)]
        )
        self.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))
        logits = self.lm_head(x)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1))


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Latency harness for Experiment 003 BitNet STE overhead.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=11)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    return BenchmarkConfig(
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
    )


def resolve_dtype(cfg: BenchmarkConfig) -> torch.dtype:
    if cfg.dtype == "float32":
        return torch.float32
    if cfg.dtype == "bfloat16":
        return torch.bfloat16
    return torch.bfloat16 if cfg.device.startswith("cuda") else torch.float32


def build_batch(cfg: BenchmarkConfig, device: torch.device) -> tuple[Tensor, Tensor]:
    shape = (cfg.batch_size, cfg.seq_len)
    input_ids = torch.randint(0, cfg.vocab_size, shape, device=device)
    target_ids = torch.randint(0, cfg.vocab_size, shape, device=device)
    return input_ids, target_ids


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_case(name: str, linear_cls: type[nn.Module], cfg: BenchmarkConfig) -> dict[str, float]:
    device = torch.device(cfg.device)
    dtype = resolve_dtype(cfg)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = DummyGPTLatencyModel(cfg, linear_cls).to(device=device)
    if dtype != torch.float32:
        model = model.to(dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    input_ids, target_ids = build_batch(cfg, device)

    first_ternary = None
    for module in model.modules():
        if isinstance(module, TernaryLinear):
            first_ternary = module
            break
    if first_ternary is not None:
        ternary, gamma = quantize_ternary_weight(first_ternary.weight)
        packed_values = pack_ternary_values(ternary)
        unpacked_values = unpack_ternary_values(packed_values, ternary.numel()).reshape_as(ternary)
        if not torch.equal(ternary.cpu(), unpacked_values.cpu()):
            raise RuntimeError("packed ternary value round-trip mismatch")
        packed, gamma = pack_ternary_weight(first_ternary.weight)
        restored = unpack_ternary_weight(packed, gamma, first_ternary.weight.shape)
        quantized = restored.to(dtype=first_ternary.weight.dtype, device=first_ternary.weight.device)
        expected = ternary.to(dtype=quantized.dtype, device=quantized.device) * gamma.to(dtype=quantized.dtype, device=quantized.device)
        if quantized.shape != first_ternary.weight.shape:
            raise RuntimeError("packed ternary round-trip shape mismatch")
        if not torch.allclose(quantized, expected):
            raise RuntimeError("packed ternary weight round-trip mismatch")

    for _ in range(cfg.warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids, target_ids)
        loss.backward()
        optimizer.step()

    maybe_synchronize(device)
    t0 = time.perf_counter()
    loss_value = 0.0
    for _ in range(cfg.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids, target_ids)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().item())
    maybe_synchronize(device)
    elapsed = time.perf_counter() - t0

    if device.type == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    else:
        peak_mem_gb = float("nan")
    step_ms = 1000.0 * elapsed / cfg.steps
    return {
        "loss": loss_value,
        "elapsed_s": elapsed,
        "step_ms": step_ms,
        "peak_mem_gb": peak_mem_gb,
        "name": name,
    }


def main() -> None:
    cfg = parse_args()
    if cfg.model_dim % cfg.num_heads != 0:
        raise ValueError("model_dim must be divisible by num_heads")
    if cfg.num_heads % cfg.num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if cfg.warmup_steps < 0 or cfg.steps <= 0:
        raise ValueError("steps must be > 0 and warmup_steps must be >= 0")
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    torch.set_float32_matmul_precision("high")

    dense = run_case("dense", nn.Linear, cfg)
    ternary = run_case("ternary", TernaryLinear, cfg)
    slowdown_pct = 100.0 * (ternary["step_ms"] / dense["step_ms"] - 1.0)

    print("Experiment 003 BitNet latency benchmark")
    print(
        "config:",
        f"device={cfg.device}",
        f"dtype={resolve_dtype(cfg)}",
        f"steps={cfg.steps}",
        f"warmup_steps={cfg.warmup_steps}",
        f"batch_size={cfg.batch_size}",
        f"seq_len={cfg.seq_len}",
        f"layers={cfg.num_layers}",
        f"model_dim={cfg.model_dim}",
        f"kv_dim={cfg.kv_dim}",
        f"mlp_dim={cfg.mlp_dim}",
    )
    for result in (dense, ternary):
        print(
            f"{result['name']}:",
            f"step_ms={result['step_ms']:.2f}",
            f"elapsed_s={result['elapsed_s']:.2f}",
            f"loss={result['loss']:.4f}",
            f"peak_mem_gb={result['peak_mem_gb']:.2f}" if result["peak_mem_gb"] == result["peak_mem_gb"] else "peak_mem_gb=n/a",
        )
    print(f"slowdown_pct={slowdown_pct:.2f}")

    if slowdown_pct > 25.0:
        print("fallback_hint=STE overhead exceeds 25%; downscale architecture before any full run.")
    else:
        print("fallback_hint=STE overhead is within the 25% guardrail.")


if __name__ == "__main__":
    main()
