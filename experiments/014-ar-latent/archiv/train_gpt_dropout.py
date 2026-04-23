from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _rope_variant_utils import float_env, load_base


baseline = load_base("exp014_ar_latent_dropout_base")


def _validate_dropout_prob(name: str, p: float) -> float:
    if not 0.0 <= p < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {p}")
    return p


def _residual_dropout_p() -> float:
    return _validate_dropout_prob("DROPOUT", float_env("DROPOUT", 0.10))


def _embed_dropout_p() -> float:
    return _validate_dropout_prob("EMBED_DROPOUT", float_env("EMBED_DROPOUT", 0.0))


def _attn_dropout_p() -> float:
    return _validate_dropout_prob("ATTN_DROPOUT", float_env("ATTN_DROPOUT", _residual_dropout_p()))


def _mlp_dropout_p() -> float:
    return _validate_dropout_prob("MLP_DROPOUT", float_env("MLP_DROPOUT", _residual_dropout_p()))


baseline.Hyperparameters.dropout = _residual_dropout_p()
baseline.Hyperparameters.embed_dropout = _embed_dropout_p()
baseline.Hyperparameters.attn_dropout = _attn_dropout_p()
baseline.Hyperparameters.mlp_dropout = _mlp_dropout_p()


class BlockDropout(baseline.Block):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        train_seq_len: int,
        layer_idx: int = 0,
        ln_scale: bool = False,
        parallel: bool = False,
        swigelu: bool = False,
    ):
        super().__init__(
            dim,
            num_heads,
            num_kv_heads,
            mlp_mult,
            rope_base,
            qk_gain_init,
            train_seq_len,
            layer_idx=layer_idx,
            ln_scale=ln_scale,
            parallel=parallel,
            swigelu=swigelu,
        )
        self.attn_dropout = baseline.nn.Dropout(_attn_dropout_p())
        self.mlp_dropout = baseline.nn.Dropout(_mlp_dropout_p())

    def forward(
        self,
        x: baseline.Tensor,
        x0: baseline.Tensor,
        q_w: baseline.Tensor,
        k_w: baseline.Tensor,
        v_w: baseline.Tensor,
        out_w: baseline.Tensor,
        up_w: baseline.Tensor,
        down_w: baseline.Tensor,
        is_causal: bool = True,
        t_emb: baseline.Tensor | None = None,
    ) -> baseline.Tensor:
        mix = baseline.cast_if_needed(self.resid_mix, x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        if t_emb is not None:
            x_in = x_in + baseline.cast_if_needed(t_emb, x_in.dtype)
        attn_out = self.attn(
            self.attn_norm(x_in) * self.ln_scale_factor,
            q_w,
            k_w,
            v_w,
            out_w,
            is_causal=is_causal,
        )
        attn_contrib = baseline.cast_if_needed(self.attn_scale, x_in.dtype)[None, None, :] * attn_out
        attn_contrib = self.attn_dropout(attn_contrib)
        if self.parallel:
            mlp_contrib = baseline.cast_if_needed(self.mlp_scale, x_in.dtype)[None, None, :] * self.mlp(
                self.mlp_norm(x_in) * self.ln_scale_factor,
                up_w,
                down_w,
            )
            mlp_contrib = self.mlp_dropout(mlp_contrib)
            x_out = x_in + attn_contrib + mlp_contrib
        else:
            x_out = x_in + attn_contrib
            mlp_contrib = baseline.cast_if_needed(self.mlp_scale, x_out.dtype)[None, None, :] * self.mlp(
                self.mlp_norm(x_out) * self.ln_scale_factor,
                up_w,
                down_w,
            )
            x_out = x_out + self.mlp_dropout(mlp_contrib)
        return x_out


class GPTDropout(baseline.GPT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_dropout = baseline.nn.Dropout(_embed_dropout_p())

    def _forward_hidden(
        self,
        input_ids: baseline.Tensor,
        x_override: baseline.Tensor | None = None,
        is_causal: bool = True,
        t_emb: baseline.Tensor | None = None,
    ) -> baseline.Tensor:
        x = x_override if x_override is not None else self.tok_emb(input_ids)
        x = baseline.F.rms_norm(x, (x.size(-1),))
        x = self.embed_dropout(x)
        x0 = x
        skips: list[baseline.Tensor] = []
        n_enc = self.num_encoder_layers
        rec_start = self.recurrence_start
        rec_end = self.recurrence_end
        do_recur = self.recurrence_depth > 1

        for i in range(n_enc):
            x = self._run_block(i, x, x0, is_causal, t_emb)
            if self.weight_share:
                x = self._run_block(i, x, x0, is_causal, t_emb)
            skips.append(x)

        for i in range(self.num_decoder_layers):
            bi = n_enc + i
            if skips:
                scaled_skip = baseline.cast_if_needed(self.skip_weights[i], x.dtype)[None, None, :] * skips.pop()
                if self.skip_gates is not None:
                    gate = baseline.torch.sigmoid(baseline.cast_if_needed(self.skip_gates[i], x.dtype))[None, None, :]
                    x = baseline.torch.lerp(scaled_skip, x, gate)
                else:
                    x = x + scaled_skip
            x = self._run_block(bi, x, x0, is_causal, t_emb)
            if self.weight_share:
                x = self._run_block(bi, x, x0, is_causal, t_emb)
            if do_recur and bi == rec_end:
                for _extra in range(self.recurrence_depth - 1):
                    for ri in range(rec_start, rec_end + 1):
                        x = self._run_block(ri, x, x0, is_causal, t_emb)

        return self.final_norm(x)


class HessianBlockDropout(baseline._HessianBlock):
    def __init__(
        self,
        dim,
        num_heads,
        num_kv_heads,
        mlp_mult,
        rope_base,
        qk_gain_init,
        train_seq_len,
        layer_idx=0,
        ln_scale=False,
        parallel=False,
        swigelu=False,
        swigelu_hidden_dim=1344,
    ):
        super().__init__(
            dim,
            num_heads,
            num_kv_heads,
            mlp_mult,
            rope_base,
            qk_gain_init,
            train_seq_len,
            layer_idx=layer_idx,
            ln_scale=ln_scale,
            parallel=parallel,
            swigelu=swigelu,
            swigelu_hidden_dim=swigelu_hidden_dim,
        )
        self.attn_dropout = baseline.nn.Dropout(_attn_dropout_p())
        self.mlp_dropout = baseline.nn.Dropout(_mlp_dropout_p())

    def forward(self, x, x0):
        mix = baseline.cast_if_needed(self.resid_mix, x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor)
        attn_contrib = baseline.cast_if_needed(self.attn_scale, x_in.dtype)[None, None, :] * attn_out
        attn_contrib = self.attn_dropout(attn_contrib)
        if self.parallel:
            mlp_contrib = baseline.cast_if_needed(self.mlp_scale, x_in.dtype)[None, None, :] * self.mlp(
                self.mlp_norm(x_in) * self.ln_scale_factor
            )
            mlp_contrib = self.mlp_dropout(mlp_contrib)
            x_out = x_in + attn_contrib + mlp_contrib
        else:
            x_out = x_in + attn_contrib
            mlp_contrib = baseline.cast_if_needed(self.mlp_scale, x_out.dtype)[None, None, :] * self.mlp(
                self.mlp_norm(x_out) * self.ln_scale_factor
            )
            x_out = x_out + self.mlp_dropout(mlp_contrib)
        return x_out


class HessianGPTDropout(baseline._HessianGPT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_dropout = baseline.nn.Dropout(_embed_dropout_p())

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        x = baseline.F.rms_norm(x, (x.size(-1),))
        x = self.embed_dropout(x)
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            if self.weight_share:
                x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                scaled_skip = baseline.cast_if_needed(self.skip_weights[i], x.dtype)[None, None, :] * skips.pop()
                if self.skip_gates is not None:
                    gate = baseline.torch.sigmoid(baseline.cast_if_needed(self.skip_gates[i], x.dtype))[None, None, :]
                    x = baseline.torch.lerp(scaled_skip, x, gate)
                else:
                    x = x + scaled_skip
            x = self.blocks[bi](x, x0)
            if self.weight_share:
                x = self.blocks[bi](x, x0)
            if self.recurrence_depth > 1 and bi == self.recurrence_end:
                for _extra in range(self.recurrence_depth - 1):
                    for ri in range(self.recurrence_start, self.recurrence_end + 1):
                        x = self.blocks[ri](x, x0)
        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits_proj = baseline.F.linear(x_flat, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x_flat)
        logits = self.logit_softcap * baseline.torch.tanh(logits_proj / self.logit_softcap)
        return baseline.F.cross_entropy(logits.float(), targets, reduction="mean")


baseline.Block = BlockDropout
baseline.GPT = GPTDropout
baseline._HessianBlock = HessianBlockDropout
baseline._HessianGPT = HessianGPTDropout


if __name__ == "__main__":
    os.environ.setdefault("RUN_ID", "exp014_dropout")
    baseline.main()
