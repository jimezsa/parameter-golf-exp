# Experiment: BitNet Ternary Linear Latency Gate

## Paper / Source
- Title: The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits
- Authors: Hu et al. plus the finalized local implementation spec
- Link: scout agent `spec-003-bitnet.md`
- Key idea: Ternarize the heavy projection weights onto `{-1, 0, 1}` with an abs-mean scale and STE training, then byte-pack four ternary values per `uint8` before LZMA.

## Hypothesis
If a PyTorch-native `TernaryLinear` only adds modest step-time overhead, BitNet-style QAT becomes viable for the real training pipeline and buys a much larger dense trunk under the 16MB artifact budget.

## Changes From Baseline
- Forked `train_gpt.py` from `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py`
- Added `ternary.py` with:
  - `TernaryQuantizeSTE`
  - `TernaryLinear`
  - base-3 pack/unpack helpers for `{-1, 0, 1}` weights
- Added `benchmark_ternary_latency.py`, a standalone 100-step dense-vs-ternary latency harness with baseline-aligned dimensions by default
- Deliberately did not wire ternary layers into the real GPT training path yet

## Run Config
- GPU: 1x H100 for the latency gate
- Default benchmark from repo root:
```bash
python experiments/003-bitnet/benchmark_ternary_latency.py
```
- Heavier H100 stress test:
```bash
python experiments/003-bitnet/benchmark_ternary_latency.py \
  --batch-size 8 \
  --seq-len 1024 \
  --steps 100 \
  --warmup-steps 10 \
  --dtype bfloat16 \
  --device cuda
```

## Results
| Run | BPB | Notes |
|-----|-----|-------|
|     |     | Pending human latency run |

## Analysis
This phase is only the latency gate. The benchmark leaves the embedding/head path as ordinary dense modules and swaps only the projection stack between dense `nn.Linear` and `TernaryLinear`, which isolates STE overhead without tangling it with flash-attention or the existing GPTQ path.

## Status
[x] Proposed by scout
[ ] Approved by professor
[x] Implemented by engineer
[ ] Tested by human
[ ] Analyzed
[ ] Decision: adopt / discard / iterate
