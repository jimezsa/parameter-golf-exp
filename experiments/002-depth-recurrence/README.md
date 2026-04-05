# Experiment: Depth Recurrence

## Paper / Source
- Title: Depth recurrence with partial middle-layer sharing and learned per-step scalar conditioning
- Authors: Internal experiment spec from scout
- Link: `/home/david/.opencolab/projects/default/AGENTS/scout/spec.md`
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

## Results
| Run | BPB | Notes |
|-----|-----|-------|
|     |     | Pending human run |

## Analysis
Implementation is complete. The next useful signal is a real 1xH100 run to check training stability, wallclock impact, and whether GPTQ calibration tolerates repeated block execution without memory issues.

## Status
[x] Proposed by scout
[ ] Approved by professor
[x] Implemented by engineer
[ ] Tested by human
[ ] Analyzed
[ ] Decision: adopt / discard / iterate
