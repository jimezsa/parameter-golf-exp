# Experiment 015: Pure AR + Attention Residuals

## Paper / Source
- Title: Attention Residuals
- Authors: Kimi Team (Moonshot AI)
- Link: https://arxiv.org/abs/2603.15031
- Key idea: Replace fixed residual accumulation with learned softmax attention over prior layer outputs. Exp 015 now defaults to the practical Block AttnRes variant instead of the slower full-history lane.

## Hypothesis

Pure autoregressive training with Block Attention Residuals should match or beat the diffusion-augmented exp 014 baseline without the Full AttnRes wall-clock tax. All diffusion code (latent MSE, noise schedules, aux diffusion heads) has been stripped — this is a clean AR-only architecture with AttnRes as the sole structural innovation. At 194M+ params, Block AttnRes matched a 1.25x compute baseline in the original paper. Our overhead is still tiny (~5,632 params for one pseudo-query per transformer block). When AttnRes is active, U-Net skip connections are bypassed — AttnRes subsumes them with a unified depth-attention mechanism.

## Changes from Baseline (Exp 014 v6)

- **All diffusion code removed** — no latent MSE loss, no noise schedules, no diffusion heads, no `DIFFUSION_*` env vars
- Pure AR training with next-token prediction only
- `ATTN_RES=1` enabled by default (was opt-in flag in exp 014)
- When active: flat sequential loop through all 11 layers with depth attention, bypasses U-Net encoder/decoder/skip structure
- Per-layer learned pseudo-query `depth_queries` (11 x 512 = 5,632 params)
- `ATTN_RES_IMPL=block` is now the default practical lane; legacy `full`, `stacked`, and `streaming` values still map to the full-history comparison path
- `ATTN_RES_BLOCK_SIZE=3` groups transformer blocks into 4 block summaries on the 11-layer model; this repo counts full transformer blocks, while the paper counts attention+MLP sublayers
- U-Net `skip_weights` / `skip_gates` excluded from training when AttnRes is on
- `depth_queries` in `CONTROL_TENSOR_NAME_PATTERNS` for FP32 precision and quant passthrough
- `_HessianGPT` mirrors the AttnRes forward path for correct GPTQ calibration
- Inherits SwiGeLU (`SWIGELU=1`) from exp 014 v6
- Depth scores now follow the paper more closely: RMSNorm on keys before score projection, and block summaries accumulate residual deltas instead of raw hidden states

## Run Config

### 1xH100 (dev) — from exp 013 v21 (1.1868)

| Parameter | Value |
|-----------|-------|
| `MATRIX_LR` | 0.045 |
| `MIN_LR` | 0.05 |
| `SCALAR_LR` | 0.02 |
| `HEAD_LR` | 0.008 |
| `TIED_EMBED_LR` | 0.03 |
| `MUON_WD` | 0.09 |
| `SWIGELU` | 1 |
| `ATTN_RES` | 1 |
| `ATTN_RES_IMPL` | `block` |
| `ATTN_RES_BLOCK_SIZE` | 3 |
| `LOGIT_SOFTCAP` | 30.0 |
| `SWA_ENABLED` | 1 |
| Steps | ~3600 |

### 8xH100 (final) — from exp 013 v6-8x (1.0964)

| Parameter | Value |
|-----------|-------|
| `MATRIX_LR` | 0.020 |
| `MIN_LR` | 0.00 |
| `SCALAR_LR` | 0.02 |
| `HEAD_LR` | 0.008 |
| `TIED_EMBED_LR` | 0.03 |
| `MUON_WD` | 0.09 |
| `SWIGELU` | 1 |
| `ATTN_RES` | 1 |
| `ATTN_RES_IMPL` | `block` |
| `ATTN_RES_BLOCK_SIZE` | 3 |
| `LOGIT_SOFTCAP` | 30.0 |
| `SWA_ENABLED` | 1 |
| Steps | ~450 |

### Shared Architecture

- 11L / 512d, 8 heads, 4 KV heads
- SP8192 tokenizer, SwiGeLU (hidden_dim=1344)
- MuonEq-R optimizer, XSA on all 11 layers
- AttnRes depth queries: 11 × 512 = 5,632 params

## Run Commands

1xH100 dev:

```bash
RUN_ID=exp015_v1 SEED=1337 SKIP_QUANT=1 ATTN_RES=1 ATTN_RES_IMPL=block ATTN_RES_BLOCK_SIZE=3 SWIGELU=1 \
torchrun --standalone --nproc_per_node=1 \
experiments/015-attn-res/train_gpt.py
```

Block vs legacy full comparison on the real box:

```bash
RUN_ID=exp015_block SKIP_QUANT=1 ATTN_RES=1 ATTN_RES_IMPL=block ATTN_RES_BLOCK_SIZE=3 torchrun --standalone --nproc_per_node=1 experiments/015-attn-res/train_gpt.py
RUN_ID=exp015_full  SKIP_QUANT=1 ATTN_RES=1 ATTN_RES_IMPL=full                          torchrun --standalone --nproc_per_node=1 experiments/015-attn-res/train_gpt.py
```

8xH100:

```bash
RUN_ID=exp015_8x_v1 SEED=1337 SKIP_QUANT=1 ATTN_RES=1 ATTN_RES_IMPL=block ATTN_RES_BLOCK_SIZE=3 SWIGELU=1 \
torchrun --standalone --nproc_per_node=8 \
experiments/015-attn-res/train_gpt.py
```

## Iteration Results

All log paths are relative to this experiment's `results/` directory.

| Version | Val BPB | Post-Quant BPB | Step Time (ms) | Artifact Size | Commit | Log | Description |
|---------|---------|----------------|-----------------|---------------|--------|-----|-------------|
| v1 | — | — | — | — | — | — | Pending: AttnRes + SwiGeLU + wide diffusion, baseline from exp 014 v6 |

- **Baseline to beat**: Exp 014 v6 = 1.1817 (1xH100 pre-quant, SwiGeLU + wide diffusion window, no AttnRes)

## Analysis

_To be updated after first run._

## Status
- [x] Proposed by autoresearch (based on Kimi/Moonshot AI paper)
- [x] Approved by human
- [x] Implemented (AttnRes code carried from exp 014)
- [ ] Tested on 1xH100
- [ ] Analyzed
- [ ] Decision: adopt / discard / iterate
