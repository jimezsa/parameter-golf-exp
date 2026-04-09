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
| v2      | 1.3922  | 1.4197 (int8+zlib) | 755        | ~14.3MB       | 9c5eb68 | WARMDOWN_ITERS=224 (SWA fix), MTP_DELAY_WEIGHT=0.1. SWA fix gave −0.017 BPB gain. 794 steps. |
| v3      | **1.2686** | 1.3633 (int6+lzma sw) ✅ | 645 | **11.69MB** ✅ | 5b276a4 | MTP_DELAY_ENABLED=0. No delay adapter overhead → 930 steps. Sliding window BPB 1.3633 beats post-quant baseline (1.3700). |

## Analysis

### v1 (bafece4) — baseline run

**What happened**: val_bpb=1.4090 pre-EMA, but EMA/SWA degrades it to 1.5512 (+0.14 BPB). GPTQ int6+lzma roundtrip is 1.9482 — catastrophic. Only 873 steps in 10 min at 687ms/step.

**Root causes**:
1. **EMA collapse**: SWA starts at step 200, runs to step 873. Most of the 673 averaged steps are still mid-learning (train_loss=4.49 at step 500). Early noisy checkpoints poison the average. Fix: push SWA start to step ~750.
2. **GPTQ int6 broken**: Ternary weights trained with STE don't survive GPTQ int6 post-hoc quantization. Use int8+zlib for dev iterations (per established protocol).
3. **MTP benefit invisible**: 873 steps is too few to evaluate whether the delay adapter (1.18M params, weight=0.3) helps. Need more steps or a faster base.

**Next**: v2 — add `SWA_START_STEP` env var (request codex), set to ~750. Drop `mtp_delay_weight` 0.3 → 0.1.

### v2 (9c5eb68) — SWA fix

**What happened**: WARMDOWN_ITERS=224 pushed SWA start to final ~30 seconds. val_bpb improved from 1.4090 → 1.3922 (−0.017). int8+zlib quant BPB 1.4197, artifact ~14.3MB.

**Root causes identified**: SWA was the primary blocker. With it fixed, base val_bpb is 1.3922. Gap to baseline: +0.025.

**Remaining bottleneck**: step time ~755ms/step → only 794 steps. Delay adapter adds one extra forward pass per step (~70ms overhead). Removing it should give ~680ms/step → ~880 steps, potentially improving training saturation.

**Next**: v3 — `MTP_DELAY_ENABLED=0`. Remove delay adapter overhead, measure whether extra steps compensate for removing the auxiliary loss.

### v3 (5b276a4) — no delay adapter

**What happened**: Removing the MTP delay adapter dropped step time from 755ms → 645ms → 930 steps (vs 794). val_bpb improved from 1.3922 → **1.2686** (−0.124 BPB). Beats baseline training BPB (1.3676) by a wide margin. GPTQ int6+lzma artifact 11.69MB. Sliding window BPB: **1.3633** — beats post-quant baseline (1.3700) by 0.0067. Roundtrip BPB: 1.3869 (above baseline by 0.0193, but sliding window is the authoritative metric).

**Root causes**: The 136 extra steps alone don't explain −0.124. The delay adapter with MTP loss may have been adding noise/instability even at weight=0.1. Pure ternary 11L/512d without MTP trains much cleaner.

**Remaining gap**: Late QAT only ran 33 steps (897-930). LATE_QAT_THRESHOLD=0.15 — raising to 0.3 should start QAT ~100 steps earlier, giving ~3× more QAT steps.

**Next**: v4 — `LATE_QAT_THRESHOLD=0.3`. Target: push sliding window BPB further below 1.3633.

## Status
- [x] Proposed by scout
- [x] Approved by professor
- [x] Implemented by engineer
- [x] Tested by human
- [x] Analyzed (v1)
- [ ] Decision: adopt / discard / iterate
