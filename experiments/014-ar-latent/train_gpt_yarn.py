from __future__ import annotations

import math
import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _rope_variant_utils import float_env, load_base


base = load_base("exp014_ar_latent_yarn_base")


class RotaryYaRN(base.Rotary):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__(dim, base=base, train_seq_len=train_seq_len, rope_dims=rope_dims)
        self.beta_slow = float_env("YARN_BETA_SLOW", 1.0)
        self.beta_fast = float_env("YARN_BETA_FAST", 32.0)
        self._cached_attention_factor = 1.0
        self._cached_factor = 1.0

    def _compute_factor(self, seq_len: int) -> float:
        factor = float_env("YARN_FACTOR", 0.0)
        if factor > 0.0:
            return max(1.0, factor)
        return max(1.0, seq_len / max(self.train_seq_len, 1))

    def _compute_attention_factor(self, factor: float) -> float:
        override = float_env("YARN_ATTENTION_FACTOR", 0.0)
        if override > 0.0:
            return override
        if factor <= 1.0:
            return 1.0
        return 0.1 * math.log(factor) + 1.0

    def _compute_inv_freq(self, seq_len: int, device):
        factor = self._compute_factor(seq_len)
        inv_freq = self.inv_freq.to(device=device)
        if factor <= 1.0:
            return inv_freq, factor
        wavelengths = (2.0 * math.pi) / inv_freq
        low_wavelength = self.train_seq_len / self.beta_fast
        high_wavelength = self.train_seq_len / self.beta_slow
        mix = ((wavelengths - low_wavelength) / max(high_wavelength - low_wavelength, 1e-6)).clamp(0.0, 1.0)
        scaled_inv_freq = inv_freq / factor
        return inv_freq * (1.0 - mix) + scaled_inv_freq * mix, factor

    def forward(self, seq_len: int, device, dtype):
        factor = self._compute_factor(seq_len)
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
            or abs(self._cached_factor - factor) > 1e-9
        ):
            inv_freq, factor = self._compute_inv_freq(seq_len, device)
            t = base.torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = base.torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
            self._cached_factor = factor
            self._cached_attention_factor = self._compute_attention_factor(factor)
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

    def get_attention_factor(self, seq_len: int) -> float:
        factor = self._compute_factor(seq_len)
        if abs(self._cached_factor - factor) > 1e-9:
            self(seq_len, self.inv_freq.device, self.inv_freq.dtype)
        return self._cached_attention_factor


class CausalSelfAttentionYaRN(base.CausalSelfAttention):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len):
        super().__init__(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len)
        self.rotary = RotaryYaRN(self.head_dim, base=rope_base, train_seq_len=train_seq_len)

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
        attn_factor = self.rotary.get_attention_factor(seqlen)
        q = q * attn_factor
        k = k * attn_factor
        q = q * base.cast_if_needed(self.q_gain, q.dtype)[None, None, :, None]
        y = base.flash_attn_3_func(q, k, v, causal=is_causal)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        return base.F.linear(y, base.cast_if_needed(out_w, x.dtype))


class HessianAttnYaRN(base._HessianAttn):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len):
        super().__init__(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len)
        self.rotary = RotaryYaRN(self.head_dim, base=rope_base, train_seq_len=train_seq_len)

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
        attn_factor = self.rotary.get_attention_factor(seqlen)
        q = q * attn_factor
        k = k * attn_factor
        q = q * base.cast_if_needed(self.q_gain, q.dtype)[None, None, :, None]
        y = base.flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        return self.proj(y.reshape(bsz, seqlen, dim))


base.Rotary = RotaryYaRN
base.CausalSelfAttention = CausalSelfAttentionYaRN
base._HessianAttn = HessianAttnYaRN


if __name__ == "__main__":
    os.environ.setdefault("RUN_ID", "exp014_yarn")
    base.main()
