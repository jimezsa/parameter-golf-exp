# Experiment: Shared-Head Multi-Token Prediction (MTP-Lite)

## Paper / Source
- **Title**: Better & Faster Large Language Models via Multi-token Prediction
- **Authors**: Meta, 2024
- **Link**: https://arxiv.org/abs/2404.19737
- **Key idea**: Predicting the next several tokens via a shared unembedding head using a lightweight adapter improves sample efficiency.

## Hypothesis
Predicting the next two tokens (t+1 and t+2) improves sample efficiency. A single lightweight transformer layer (512d, 4 heads, 128-dim FFN) delay adapter provides enough capacity to shift representations from t+1 to t+2.

## Base Code
- **Fork from**: `experiments/003-bitnet/train_gpt.py` (our current ternary base).

## Changes from Baseline
- **Architecture**: Add a single lightweight transformer layer (512d, 4 heads, 128-dim FFN) as the delay adapter (~1.3M ternary params ≈ ~234KB LZMA). Stripped at inference, thus zero impact on the final artifact size.
- **Loss**: Add auxiliary loss for $N_{t+2}$ token prediction.
- **Weights**: Set initial $\lambda_2$ (t+2 loss weight) to 0.3 to prevent early instability in ternary STE training.
- **Kill Signal**: Abort the run if t+2 loss > 2x t+1 loss after 5K steps.

## Run Config
- **GPU**: 1x H100 (dev) / 8x H100 (final)
- **Steps / Duration**: 10 minutes (wallclock)
- **Key hyperparameters changed**: Multi-token prediction loss activated, initial $\lambda_2 = 0.3$.

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
