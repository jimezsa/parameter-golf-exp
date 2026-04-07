# Experiment: Mixture of Depths (MoD) over Ternary Layers

## Paper / Source
- **Title**: Mixture-of-Depths: Dynamically allocating compute in transformer-based language models
- **Authors**: Raposo et al., Google DeepMind, 2024
- **Link**: https://arxiv.org/abs/2404.02258
- **Key idea**: Tokens choose to skip layers via a learned router, increasing physical depth while maintaining a fixed compute budget.

## Hypothesis
By routing tokens to skip layers via a learned router, we can increase physical depth while keeping the FLOP/step (wallclock) budget identical. Building on Exp 002 (Depth Recurrence), we can use 18 unique physical ternary layers, recurred 2x, yielding 36 effective layers with routing. This enables deeper computation per token without blowing the 16MB LZMA budget.

## Base Code
- **Fork from**: `experiments/003-bitnet/train_gpt.py` (our current ternary base).
- **Incorporate logic from**: `experiments/002-depth-recurrence/train_gpt.py` (for the 2x recurrence mechanism).

## Changes from Baseline
- **Architecture**: 18 unique ternary layers instead of 24L.
- **Recurrence**: 2x depth recurrence (18 physical layers run twice = 36 effective layers).
- **Routing**: Mixture-of-Depths top-k routing per layer (BF16 routers with stop-gradient for wallclock predictability).
- **Loss**: Aux entropy loss for routing set at 0.01.
- **LZMA Budgeting**: 18 layers * 3.15M params = ~57M unique params. ~57M * 0.18 bytes/param = ~10.3MB LZMA footprint. This leaves a generous ~5.7MB for embeddings and routers.

## Run Config
- **GPU**: 1x H100 (dev) / 8x H100 (final)
- **Steps / Duration**: 10 minutes (wallclock)
- **Key hyperparameters changed**: 18 physical layers, 2x recurrence, top-k routing fraction, aux entropy loss (0.01).

## Results
| Run | BPB | Notes |
|-----|-----|-------|
|     |     |       |

## Analysis
What worked, what didn't, why.

## Status
- [x] Proposed by scout
- [x] Approved by professor
- [x] Implemented by engineer
- [ ] Tested by human
- [ ] Analyzed
- [ ] Decision: adopt / discard / iterate
