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
| v4      | **1.2680** | 1.3632 (int6+lzma sw) ✅ | 646 | **11.55MB** ✅ | 364aacf | LATE_QAT_THRESHOLD=0.3 → 67 QAT steps (2× v3). Tiny BPB gain (−0.0001). Post-EMA BPB=1.3753 — EMA averaging in early noisy weights throughout training. |
| v5      | **1.2681** | **1.2745** (int6+lzma sw) ✅ | 644 | **12.04MB** ✅ | dbf8877 | EMA_START_STEP=700 — EMA re-initialized at step 700, averages only final ~229 warmdown steps. Post-EMA BPB: **1.2900** (gap 0.107 → 0.019). Roundtrip: 1.2986. Sliding window: **1.2745** — dominant win. |

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

### v4 — LATE_QAT_THRESHOLD=0.3 (67 QAT steps)

**What happened**: Doubling QAT steps (33→67) gave essentially no gain: val_bpb 1.2686→1.2680 (−0.0006), sliding window BPB 1.3633→1.3632 (−0.0001). Post-EMA BPB is 1.3753, a full 0.107 above training BPB (1.2680).

**Root cause**: EMA accumulates from step 0 with decay=0.997 (half-life ~231 steps). Over 929 steps it averages in all early noisy training checkpoints. At step 929, roughly steps 700-929 dominate, but steps 0-700 still contribute meaningful weight. This degrades EMA quality. QAT threshold tuning has hit diminishing returns — the real bottleneck is the EMA averaging early-training noise.

**Next**: v5 — `EMA_START_STEP=700`. Re-initialize EMA at step 700 to the current model weights, then average only the final ~229 warmdown steps. Expected: post-EMA BPB much closer to training BPB (1.2680), potentially better quantization.

**EMA_START_STEP** is now implemented in `train_gpt.py`. Set `EMA_START_STEP=700` for v5.

### v5 — EMA_START_STEP=700

**What happened**: EMA fix fully confirmed. Post-EMA BPB: **1.2900** (v4: 1.3753) — gap collapsed from 0.107 → 0.019. The EMA now averages only the final ~229 warmdown-phase steps. int6+lzma roundtrip BPB: **1.2986**. Sliding window BPB: **1.2745** — the authoritative metric. Artifact: 12.04MB ✅.

**Key insight**: The EMA start step was the dominant bottleneck for quantization quality. Early-training noisy weights were poisoning the averaged model. Starting EMA at step 700 (just before warmdown begins) gives the quantizer a clean, converged model to work with.

**Remaining gap**: Sliding window BPB 1.2745 is well below the pre-GPTQ baseline (1.3676) and even below training BPB (1.2681). The 0.019 residual post-EMA gap (training 1.2681 → EMA 1.2900) is still worth closing. Candidate: EMA_START_STEP=750 or 800 to average only the deepest warmdown steps. Also consider whether the 10-min training budget could be extended to 1000+ steps with further step time optimization.

**Next**: v6 — `EMA_START_STEP=800` to tighten EMA window further. Expect post-EMA BPB to approach training BPB (~1.268) and sliding window BPB to dip below 1.27.

## Status
- [x] Proposed by scout
- [x] Approved by professor
- [x] Implemented by engineer
- [x] Tested by human
- [x] Analyzed (v1–v5)
- [ ] Decision: adopt / discard / iterate (5/10+ iterations — continuing)
