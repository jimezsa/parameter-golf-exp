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

## Iteration Results

| Version | Val BPB | Post-Quant BPB | Step Time (ms) | Artifact Size | Commit | Description |
|---------|---------|----------------|-----------------|---------------|--------|-------------|
| v1      | 1.4090  | 1.9482 (int6+lzma) | 687.76     | 7.41MB        | bafece4 | Initial run. EMA collapses to 1.5512 (SWA starts step 200, poisons average). GPTQ int6 broken on ternary. Only 873 steps in 10 min. |

## Analysis

### v1 (bafece4) — baseline run

**What happened**: val_bpb=1.4090 pre-EMA, but EMA/SWA degrades it to 1.5512 (+0.14 BPB). GPTQ int6+lzma roundtrip is 1.9482 — catastrophic. Only 873 steps in 10 min at 687ms/step.

**Root causes**:
1. **EMA collapse**: SWA starts at step 200, runs to step 873. Most of the 673 averaged steps are still mid-learning (train_loss=4.49 at step 500). Early noisy checkpoints poison the average. Fix: push SWA start to step ~750.
2. **GPTQ int6 broken**: Ternary weights trained with STE don't survive GPTQ int6 post-hoc quantization. Use int8+zlib for dev iterations (per established protocol).
3. **MTP benefit invisible**: 873 steps is too few to evaluate whether the delay adapter (1.18M params, weight=0.3) helps. Need more steps or a faster base.

**Next**: v2 — add `SWA_START_STEP` env var (request codex), set to ~750. Drop `mtp_delay_weight` 0.3 → 0.1.

## Status
- [x] Proposed by scout
- [x] Approved by professor
- [x] Implemented by engineer
- [x] Tested by human
- [x] Analyzed (v1)
- [ ] Decision: adopt / discard / iterate
