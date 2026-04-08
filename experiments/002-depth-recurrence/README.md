# Experiment: Depth Recurrence

## Paper / Source
- Title: Depth recurrence with partial middle-layer sharing and learned per-step scalar conditioning
- Authors: Internal experiment spec from scout
- Link: scout agent `spec-002-depth-recurrence.md`
- Key idea: Reuse the middle transformer block stack multiple times so an 11-layer physical model behaves like a deeper model, while keeping early and late layers unique for stability and adding only a tiny learned step-scaling tensor.

## Hypothesis
Looping layers `2..8` with `RECURRENCE_DEPTH=2` should buy effective depth close to an 18-layer model without adding weight matrices, improving local val loss and BPB while staying comfortably inside the artifact budget.

## Changes From Baseline
- Forked from `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py`
- Added `RECURRENCE_DEPTH`, `RECURRENCE_START`, and `RECURRENCE_END` env-configured hyperparameters
- Reworked the banked `GPT` forward path into:
  - early unshared layers
  - recurrent middle block loop
  - late unshared layers
- Added learned `recurrence_step_scales` for per-step, per-layer conditioning
- Preserved absolute RoPE behavior by reusing the same sequence positions on every recurrence step
- Mirrored the recurrence path inside the non-banked Hessian/GPTQ model so calibration and post-quant eval see the same architecture
- Registered `recurrence_step_scales` as a scalar/control tensor so it stays FP32 and is optimized by the scalar AdamW path

## Run Config
- GPU: 1x H100 (dev) / 8x H100 (final)
- Default run from repo root:
```bash
RUN_ID=exp002_depth_recurrence \
RECURRENCE_DEPTH=2 \
RECURRENCE_START=2 \
RECURRENCE_END=8 \
torchrun --standalone --nproc_per_node=1 experiments/002-depth-recurrence/train_gpt.py
```
- Multi-GPU example:
```bash
RUN_ID=exp002_depth_recurrence_8gpu \
RECURRENCE_DEPTH=2 \
RECURRENCE_START=2 \
RECURRENCE_END=8 \
torchrun --standalone --nproc_per_node=8 experiments/002-depth-recurrence/train_gpt.py
```

## Iteration Results

| Version | Val BPB | Post-Quant BPB | Step Time (ms) | Artifact Size | Commit | Description |
|---------|---------|----------------|-----------------|---------------|--------|-------------|
| v10     | 1.2686  | ~1.3537        | —               | >16MB ❌       | —      | Best uncompressed, WARMDOWN=224 |
| v11     | 1.2691  | —              | —               | >16MB ❌       | —      | EMA variant (no gain over v10) |
| v12     | 1.3719  | 1.5015         | —               | 13.4MB ✅      | —      | WARMDOWN=3500, PRUNE_FRAC=0.15 — undertrained |
| v13     | **1.2703** | **1.3600**  | —               | 17.3MB ❌      | —      | WARMDOWN=224, PRUNE_FRAC=0.15 — beats baseline BPB, 1.3MB over limit |
| v14     | 1.2695  | **1.3586**     | —               | 16.6MB ❌      | b34acf6 | WARMDOWN=224, PRUNE_FRAC=0.27 — beats baseline BPB, 0.6MB over limit |
| v15     | TBD     | TBD            | TBD             | TBD           | —      | WARMDOWN=224, PRUNE_FRAC=0.37 — **RUNNING** |

**Key insight:** WARMDOWN_ITERS=224 is critical. Default 3500 decays LR from step 1, collapsing quality. With 224, full LR holds until ~step 890. Pruned zeros compress well via LZMA.

## Analysis
- Depth recurrence with shared middle layers (2..8, depth=2) beats baseline 1.3676 in raw val BPB.
- Remaining challenge is purely compression: fitting artifact under 16MB with int8+lzma.
- If v14 still too large, increase PRUNE_FRAC to 0.35–0.40.

## Status
[x] Proposed by scout
[ ] Approved by professor
[x] Implemented by engineer
[ ] Tested by human
[ ] Analyzed
[ ] Decision: adopt / discard / iterate
