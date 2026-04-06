from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_TERNARY_PAD_TRIT = 1
_TERNARY_BASE3_FACTOR = (1, 3, 9, 27)
_TERNARY_2BIT_SHIFTS = (0, 2, 4, 6)
_TERNARY_EPS = 1e-8
_VALID_PACK_FORMATS = ("base3", "two_bit")


def quantize_ternary_weight(weight: Tensor, eps: float = _TERNARY_EPS) -> tuple[Tensor, Tensor]:
    """Round+clip a latent weight tensor onto {-1, 0, 1} with BitNet's abs-mean scale."""
    weight_fp32 = weight.detach().float()
    gamma = weight_fp32.abs().mean().clamp(min=eps)
    ternary = torch.clamp(torch.round(weight_fp32 / gamma), -1.0, 1.0)
    return ternary.to(dtype=torch.int8), gamma.to(dtype=torch.float32)


def _normalize_pack_format(pack_format: str) -> str:
    if pack_format not in _VALID_PACK_FORMATS:
        valid = ", ".join(_VALID_PACK_FORMATS)
        raise ValueError(f"invalid pack_format={pack_format!r}; expected one of {valid}")
    return pack_format


class TernaryQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: Tensor) -> Tensor:
        ternary, gamma = quantize_ternary_weight(weight)
        return ternary.to(dtype=weight.dtype) * gamma.to(dtype=weight.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor]:
        return (torch.clamp(grad_output, -1.0, 1.0),)


class TernaryLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weights_are_quantized = False
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight if self.weights_are_quantized else TernaryQuantizeSTE.apply(self.weight)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight.to(dtype=x.dtype), bias)

    def set_weights_are_quantized(self, enabled: bool = True) -> None:
        self.weights_are_quantized = enabled

    @torch.no_grad()
    def export_quantized(self, pack_format: str = "base3") -> dict[str, Tensor]:
        packed, gamma = pack_ternary_weight(self.weight, pack_format=pack_format)
        return {
            "packed_weight": packed,
            "gamma": gamma.reshape(1),
            "shape": torch.tensor(self.weight.shape, dtype=torch.int64),
        }

    @torch.no_grad()
    def load_packed_weight_(
        self,
        packed: Tensor,
        gamma: Tensor,
        shape: tuple[int, ...] | torch.Size,
        pack_format: str = "base3",
    ) -> None:
        restored = unpack_ternary_weight(packed, gamma, shape, pack_format=pack_format).to(
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        if restored.shape != self.weight.shape:
            raise ValueError(f"shape mismatch: expected {tuple(self.weight.shape)}, got {tuple(restored.shape)}")
        self.weight.copy_(restored)
        self.weights_are_quantized = True


def pack_ternary_values(values: Tensor, pack_format: str = "base3") -> Tensor:
    pack_format = _normalize_pack_format(pack_format)
    flat = values.reshape(-1).to(dtype=torch.int16)
    if flat.numel() == 0:
        return torch.empty(0, dtype=torch.uint8, device=values.device)
    if not torch.all((flat >= -1) & (flat <= 1)):
        raise ValueError("ternary packing expects values in {-1, 0, 1}")
    trits = flat + 1
    pad = (-trits.numel()) % 4
    if pad:
        # Pad with the zero symbol so the tail is neutral after unpacking.
        trits = F.pad(trits, (0, pad), value=_TERNARY_PAD_TRIT)
    chunks = trits.view(-1, 4)
    if pack_format == "base3":
        packed = (
            chunks[:, 0]
            + chunks[:, 1] * _TERNARY_BASE3_FACTOR[1]
            + chunks[:, 2] * _TERNARY_BASE3_FACTOR[2]
            + chunks[:, 3] * _TERNARY_BASE3_FACTOR[3]
        )
    else:
        packed = (
            chunks[:, 0]
            + (chunks[:, 1] << _TERNARY_2BIT_SHIFTS[1])
            + (chunks[:, 2] << _TERNARY_2BIT_SHIFTS[2])
            + (chunks[:, 3] << _TERNARY_2BIT_SHIFTS[3])
        )
    return packed.to(dtype=torch.uint8)


def unpack_ternary_values(packed: Tensor, num_values: int, pack_format: str = "base3") -> Tensor:
    pack_format = _normalize_pack_format(pack_format)
    packed_flat = packed.reshape(-1).to(dtype=torch.int16)
    if packed_flat.numel() == 0:
        return torch.empty(num_values, dtype=torch.int8, device=packed.device)
    if pack_format == "base3":
        w0 = packed_flat % 3
        w1 = (packed_flat // 3) % 3
        w2 = (packed_flat // 9) % 3
        w3 = (packed_flat // 27) % 3
        values = torch.stack((w0, w1, w2, w3), dim=1).reshape(-1)[:num_values]
    else:
        w0 = packed_flat & 0b11
        w1 = (packed_flat >> _TERNARY_2BIT_SHIFTS[1]) & 0b11
        w2 = (packed_flat >> _TERNARY_2BIT_SHIFTS[2]) & 0b11
        w3 = (packed_flat >> _TERNARY_2BIT_SHIFTS[3]) & 0b11
        values = torch.stack((w0, w1, w2, w3), dim=1).reshape(-1)[:num_values]
        if torch.any(values > 2):
            raise ValueError("invalid two_bit ternary payload; encountered reserved code 3")
    return (values - 1).to(dtype=torch.int8)


def pack_ternary_weight(weight: Tensor, pack_format: str = "base3") -> tuple[Tensor, Tensor]:
    ternary, gamma = quantize_ternary_weight(weight)
    return pack_ternary_values(ternary, pack_format=pack_format), gamma


def unpack_ternary_weight(
    packed: Tensor,
    gamma: Tensor,
    shape: tuple[int, ...] | torch.Size,
    pack_format: str = "base3",
) -> Tensor:
    shape = tuple(int(v) for v in shape)
    num_values = math.prod(shape)
    ternary = unpack_ternary_values(packed, num_values, pack_format=pack_format).reshape(shape)
    gamma = gamma.reshape(()).to(dtype=torch.float32, device=packed.device)
    return ternary.to(dtype=torch.float32) * gamma
