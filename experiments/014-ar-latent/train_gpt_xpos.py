from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _rope_variant_utils import float_env, load_base


base = load_base("exp014_ar_latent_xpos_base")


def _apply_scaled_rotary(x, scale, rope_dims: int):
    if rope_dims <= 0:
        return x
    if rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        return base.torch.cat((x_rope * scale, x_pass), dim=-1)
    return x * scale


class RotaryXPos(base.Rotary):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__(dim, base=base, train_seq_len=train_seq_len, rope_dims=rope_dims)
        self.scale_base = float_env("XPOS_SCALE_BASE", 512.0)
        half = max(self.rope_dims // 2, 1)
        dim_pos = base.torch.arange(half, dtype=base.torch.float32)
        scale = (dim_pos + 0.4 * half) / (1.4 * half)
        self.register_buffer("_xpos_scale", scale.clamp_min(1e-4), persistent=False)

    def get_scale(self, seq_len: int, device, dtype):
        positions = base.torch.arange(seq_len, device=device, dtype=base.torch.float32)
        centered = positions - float(seq_len // 2)
        scale = self._xpos_scale.to(device=device).unsqueeze(0).pow(centered.unsqueeze(1) / self.scale_base)
        scale = scale[None, :, None, :]
        scale = base.torch.cat((scale, scale), dim=-1)
        return scale.to(dtype=dtype)


class CausalSelfAttentionXPos(base.CausalSelfAttention):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len):
        super().__init__(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len)
        self.rotary = RotaryXPos(self.head_dim, base=rope_base, train_seq_len=train_seq_len)

    def forward(self, x, q_w, k_w, v_w, out_w, is_causal: bool = True):
        bsz, seqlen, dim = x.shape
        q = base.F.linear(x, base.cast_if_needed(q_w, x.dtype)).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = base.F.linear(x, base.cast_if_needed(k_w, x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = base.F.linear(x, base.cast_if_needed(v_w, x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = base.F.rms_norm(q, (q.size(-1),))
        k = base.F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = base.apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = base.apply_rotary_emb(k, cos, sin, self.rope_dims)
        effective_rope_dims = self.rotary.rope_dims
        if is_causal and effective_rope_dims > 0:
            scale = self.rotary.get_scale(seqlen, x.device, q.dtype)
            q = _apply_scaled_rotary(q, scale, effective_rope_dims)
            k = _apply_scaled_rotary(k, scale.reciprocal().clamp_max(1e4), effective_rope_dims)
        q = q * base.cast_if_needed(self.q_gain, q.dtype)[None, None, :, None]
        y = base.flash_attn_3_func(q, k, v, causal=is_causal)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        return base.F.linear(y, base.cast_if_needed(out_w, x.dtype))


class HessianAttnXPos(base._HessianAttn):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len):
        super().__init__(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len)
        self.rotary = RotaryXPos(self.head_dim, base=rope_base, train_seq_len=train_seq_len)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = base.F.rms_norm(q, (q.size(-1),))
        k = base.F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = base.apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = base.apply_rotary_emb(k, cos, sin, self.rope_dims)
        effective_rope_dims = self.rotary.rope_dims
        if effective_rope_dims > 0:
            scale = self.rotary.get_scale(seqlen, x.device, q.dtype)
            q = _apply_scaled_rotary(q, scale, effective_rope_dims)
            k = _apply_scaled_rotary(k, scale.reciprocal().clamp_max(1e4), effective_rope_dims)
        q = q * base.cast_if_needed(self.q_gain, q.dtype)[None, None, :, None]
        y = base.flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        return self.proj(y.reshape(bsz, seqlen, dim))


base.Rotary = RotaryXPos
base.CausalSelfAttention = CausalSelfAttentionXPos
base._HessianAttn = HessianAttnXPos


if __name__ == "__main__":
    os.environ.setdefault("RUN_ID", "exp014_xpos")
    base.main()
