# Experiment 015: Pure AR + Attention Residuals

## Paper / Source
- Title: Attention Residuals
- Authors: Kimi Team (Moonshot AI)
- Link: https://arxiv.org/abs/2603.15031
- Key idea: Replace fixed residual accumulation with learned softmax attention over all prior layer outputs, letting each layer dynamically select its optimal input combination.

## Hypothesis

Pure autoregressive training with Full Attention Residuals should match or beat the diffusion-augmented exp 014 baseline. All diffusion code (latent MSE, noise schedules, aux diffusion heads) has been stripped — this is a clean AR-only architecture with AttnRes as the sole structural innovation. At 194M+ params, Block AttnRes matched a 1.25x compute baseline in the original paper. Our overhead is ~5,632 params (~11 KB). When AttnRes is active, U-Net skip connections are bypassed — AttnRes subsumes them with a unified depth-attention mechanism.

## Changes from Baseline (Exp 014 v6)

- **All diffusion code removed** — no latent MSE loss, no noise schedules, no diffusion heads, no `DIFFUSION_*` env vars
- Pure AR training with next-token prediction only
- `ATTN_RES=1` enabled by default (was opt-in flag in exp 014)
- When active: flat sequential loop through all 11 layers with depth attention, bypasses U-Net encoder/decoder/skip structure
- Per-layer learned pseudo-query `depth_queries` (11 x 512 = 5,632 params)
- U-Net `skip_weights` / `skip_gates` excluded from training when AttnRes is on
- `depth_queries` in `CONTROL_TENSOR_NAME_PATTERNS` for FP32 precision and quant passthrough
- `_HessianGPT` mirrors the AttnRes forward path for correct GPTQ calibration
- Inherits SwiGeLU (`SWIGELU=1`) from exp 014 v6

## Run Config

- GPU: 1x H100 (dev) / 8x H100 (final)
- Steps: ~3600 (1x) / ~450 (8x)
- Key env vars: `ATTN_RES=1`, `SWIGELU=1`

## Run Commands

1xH100 dev:

```bash
RUN_ID=exp015_v1 SEED=1337 SKIP_QUANT=1 ATTN_RES=1 SWIGELU=1 \
torchrun --standalone --nproc_per_node=1 \
experiments/015-attn-res/train_gpt.py
```

8xH100:

```bash
RUN_ID=exp015_8x_v1 SEED=1337 SKIP_QUANT=1 ATTN_RES=1 SWIGELU=1 \
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
