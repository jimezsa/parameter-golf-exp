from __future__ import annotations

import math
import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _rope_variant_utils import int_env, load_base


base = load_base("exp014_ar_latent_headwise_rope_base")


def _rotated_head_counts(num_heads: int, num_kv_heads: int) -> tuple[int, int]:
    group = num_heads // num_kv_heads
    if "ROPE_ROTATE_KV_HEADS" in os.environ:
        rotate_kv_heads = int_env("ROPE_ROTATE_KV_HEADS", num_kv_heads)
    else:
        requested_q_heads = int_env("ROPE_ROTATE_HEADS", num_heads)
        rotate_kv_heads = math.ceil(max(0, requested_q_heads) / group)
    rotate_kv_heads = max(0, min(num_kv_heads, rotate_kv_heads))
    rotate_q_heads = min(num_heads, rotate_kv_heads * group)
    return rotate_q_heads, rotate_kv_heads


def _apply_rotary_to_prefix(x, rotate_heads: int, cos, sin, rope_dims: int):
    if rotate_heads <= 0:
        return x
    if rotate_heads >= x.size(-2):
        return base.apply_rotary_emb(x, cos, sin, rope_dims)
    x_rot = base.apply_rotary_emb(x[..., :rotate_heads, :], cos, sin, rope_dims)
    return base.torch.cat((x_rot, x[..., rotate_heads:, :]), dim=-2)


class CausalSelfAttentionHeadwise(base.CausalSelfAttention):
    def forward(self, x, q_w, k_w, v_w, out_w, is_causal: bool = True):
        bsz, seqlen, dim = x.shape
        q = base.F.linear(x, base.cast_if_needed(q_w, x.dtype)).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = base.F.linear(x, base.cast_if_needed(k_w, x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = base.F.linear(x, base.cast_if_needed(v_w, x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = base.F.rms_norm(q, (q.size(-1),))
        k = base.F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        rotate_q_heads, rotate_kv_heads = _rotated_head_counts(self.num_heads, self.num_kv_heads)
        q = _apply_rotary_to_prefix(q, rotate_q_heads, cos, sin, self.rope_dims)
        k = _apply_rotary_to_prefix(k, rotate_kv_heads, cos, sin, self.rope_dims)
        q = q * base.cast_if_needed(self.q_gain, q.dtype)[None, None, :, None]
        y = base.flash_attn_3_func(q, k, v, causal=is_causal)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        return base.F.linear(y, base.cast_if_needed(out_w, x.dtype))


class HessianAttnHeadwise(base._HessianAttn):
    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = base.F.rms_norm(q, (q.size(-1),))
        k = base.F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        rotate_q_heads, rotate_kv_heads = _rotated_head_counts(self.num_heads, self.num_kv_heads)
        q = _apply_rotary_to_prefix(q, rotate_q_heads, cos, sin, self.rope_dims)
        k = _apply_rotary_to_prefix(k, rotate_kv_heads, cos, sin, self.rope_dims)
        q = q * base.cast_if_needed(self.q_gain, q.dtype)[None, None, :, None]
        y = base.flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        return self.proj(y.reshape(bsz, seqlen, dim))


base.CausalSelfAttention = CausalSelfAttentionHeadwise
base._HessianAttn = HessianAttnHeadwise


if __name__ == "__main__":
    os.environ.setdefault("RUN_ID", "exp014_headwise_rope")
    base.main()
