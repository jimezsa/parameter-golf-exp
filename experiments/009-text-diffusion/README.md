# Experiment 009: AR-Diffusion Hybrid (Soft-Masked)

## Paper / Source
- **Titles**:
  1. *A Cheaper and Better Diffusion Language Model with Soft-Masked Noise* (Chen et al., 2023) — https://arxiv.org/abs/2304.04746
  2. *Text Generation with Diffusion Language Models: A Pre-training Approach with Continuous Paragraph Denoise* (Lin et al., 2022) — https://arxiv.org/abs/2212.11685
  3. *A Reparameterized Discrete Diffusion Model for Text Generation* (Zheng et al., 2023) — https://arxiv.org/abs/2302.05737
- **Key idea**: Pure diffusion LMs suffer from sample inefficiency — they predict only a masked subset per step. This hybrid injects a bidirectional soft-masked auxiliary pass on ~25% of steps, adding structural context without doubling step time or changing the primary AR BPB objective.

## Hypothesis
Within a 10-minute wallclock, pure diffusion underfits. An auxiliary bidirectional diffusion pass (25% of steps) improves structural representations and yields better final BPB than a pure AR baseline, while keeping step time within budget.

## Changes from Baseline
Fork from `experiments/008-mtp-lite/train_gpt.py` (MTP-Lite with parameter banking and EMA).

- **Architecture additions**:
  - `mask_embed`: learned embedding for soft token corruption
  - `time_proj`: lightweight projection of continuous timestep $t \sim U(0,1)$, added to the residual stream before each attention block
- **Dual-pass training**:
  - AR pass (primary): causal masking (`is_causal=True`), runs 100% of steps, drives BPB and MTP loss
  - Diffusion pass (auxiliary): bidirectional attention (`is_causal=False`) on soft-masked sequence, runs ~25% of steps
- **Diffusion loss weight**: 0.3

## Run Config
- **GPU**: 1x H100 (dev) / 8x H100 (final)
- **Steps / Duration**: 10 minutes wallclock
- Data prep (run once per pod):
```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```
- Run from repo root (1x H100 dev):
```bash
RUN_ID=exp009_text_diffusion \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 experiments/009-text-diffusion/train_gpt.py
```
- Run for final submission (8x H100):
```bash
RUN_ID=exp009_text_diffusion \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 experiments/009-text-diffusion/train_gpt.py
```

## Key Metrics to Record
1. **val_bpb** — final validation bits per byte (AR pass, primary metric)
2. **sw BPB** — sliding window BPB (authoritative post-quant metric)
3. **Average step time** — mean ms/step from training logs
4. **Total wallclock** — end-to-end training time
5. **Compressed model size** — final artifact size in bytes (target ≤ 16MB)

## Iteration Results

| Version | Val BPB | Post-Quant BPB | Step Time (ms) | Artifact Size | Commit | Description |
|---------|---------|----------------|----------------|---------------|--------|-------------|
| v1      | 1.5064  | GPTQ crash     | 899            | —             | 49fc2fb | Initial run — baseline config, 25% diffusion gate, diffusion weight 0.3 |
| v2      | 1.4641  | 4.1046 (int6+lzma) | 840         | 4.43MB ✅     | 49fc2fb | MTP_DELAY_ENABLED=0, EMA_START_STEP=800, GPTQ Cholesky retry fix |
| v3      | 1.2890  | **1.3320** (int6+lzma) | 763      | **11.47MB** ✅ | b207e41 | EMA_START_STEP=500, DIFFUSION_AUX_PROB=0.15, SWA at step 750 |
| v4      | 1.4830  | 2.0497 (int6+lzma) | 897         | 8.29MB ✅     | 49fc2fb | 13L/512d (+2 layers), VE_LAYERS=11,12, XSA_LAST_N=13 — regression, fewer steps (670 vs 787) |
| v5      | 1.2775  | **1.2887** (int6+lzma) | 688      | **12.37MB** ✅ | 8fb0c2e | Minimal diffusion: DIFFUSION_AUX_PROB=0.05, weight=0.1, 11L. 873 steps, best post-quant |
| v6      | 1.4008  | 1.8406 (int6+lzma) | 688      | 7.65MB ✅ | 0980da8 | DIFF_PROB=0.10, weight=0.15, stop_frac=0.50. Regression — SWA@150 + missing EMA_START_STEP poisoned quant |
| v7      | **1.2763** | **1.2753** (int6+lzma) | 692  | **12.45MB** ✅ | 0980da8 | Scout's schedule: DIFF_PROB=0.08, weight=0.10, stop_frac=0.70, EMA_START_STEP=800. New exp 009 best, near-zero quant degradation |
| v8      | 1.2759  | 1.2751 (int6+lzma) | 692  | 12.0MB ✅ | 7a06d1f | WARMDOWN_ITERS=224 (was 3500 default). Negligible gain over v7 — warmdown schedule irrelevant for this architecture |
| v9      | 1.3565  | 1.5862 (int6+lzma) | ~690 | — | — | Higher LR experiment — SWA fired at step 150, poisoned quant weights |
| v10     | 1.3041  | 4.1051 (int6+lzma) | 925  | 5.28MB | — | 11L/576d width scaling — catastrophic: EMA never activated (only 649 steps at 925ms/step) |
| v11     | 1.3936  | 1.8012 (int6+lzma) | 693  | 7.73MB ✅ | — | QK_GAIN=5.25 experiment — higher gain degraded BPB, no benefit |
| v12     | **1.2742** | **1.2746** (int6+lzma) | **684** | **12.60MB** ✅ | 57a3f4d | Parallel residuals (PARALLEL_START_LAYER=7). Faster steps → more training → new exp 009 best sw BPB |

- **Val BPB**: raw validation bits-per-byte before quantization (AR pass)
- **Post-Quant BPB**: after int8+zlib (or int6+lzma if applicable)
- **Step Time**: average training step time in ms
- **Artifact Size**: compressed model size (target ≤ 16MB)
- **Commit**: short SHA of the code version used
- **Description**: what changed from the previous version

## Analysis

### v1 → v2 Progress
- val_bpb improved from 1.5064 → 1.4641 (−0.04), but still +0.10 above baseline (1.3676)
- `MTP_DELAY_ENABLED=0` helped (known biggest win from exp 008), but less dramatic than exp 008's 1.39→1.27 jump — diffusion auxiliary overhead is eating into the benefit
- **Critical bug:** `EMA_START_STEP=800` but training only reached step 715 (wallclock cap). EMA never started → post_ema BPB is 4.1044 (initial random weights). The int6+lzma roundtrip BPB of 4.1046 reflects this broken EMA, not the actual model quality
- GPTQ Cholesky retry fix worked — no crash this time
- SWA started at step 50 (too early, same poisoning issue as exp 008 v1)
- Artifact size 4.43MB is very compact (well under 16MB limit)

### v2 → v3 Progress
- **Massive improvement**: val_bpb 1.4641 → 1.2890 (−0.175), now beats baseline (1.3676) by 0.079
- EMA fix worked: `EMA_START_STEP=500`, training reached step 787 → EMA collected ~287 steps. Post-EMA BPB 1.3221 (vs v2's broken 4.1044)
- Post-quant int6+lzma: **1.3320** — beats baseline's post-quant reference (1.3700) by 0.038
- Diffusion aux prob reduced 25% → 15%: faster steps (763ms vs 840ms), more steps completed (787 vs 715)
- SWA start at step 750 (late, healthy averaging window)
- Late QAT kicked in at step 754
- Artifact 11.47MB, well under 16MB
- Still trails exp 008 v6 (sw BPB 1.2716) — diffusion overhead costs ~0.06 BPB

### v3 → v4 Regression
- **Clear regression**: val_bpb 1.2890 → 1.4830 (+0.194), back above baseline (1.3676)
- Going wider (13L/512d, 31.7M params vs 11L 27.0M) hurt badly — 897ms/step meant only 670 steps (vs 787 at 11L). Fewer iterations = worse convergence within the 10-min budget
- Post-quant catastrophic: int6+lzma roundtrip BPB 2.0497, sliding window 2.0286 — larger model's weights don't survive 6-bit quantization
- EMA gap moderate (1.4830 → 1.5033 post-EMA), but starting from a much worse base
- **Verdict: revert to 11L.** More layers within the same wallclock budget is counterproductive

### v4 → v5 Breakthrough
- **Major improvement**: val_bpb 1.4830 → 1.2775 (−0.206), post-quant sw BPB **1.2887** — best post-quant result for exp 009
- Reverted to 11L/512d (v3 base), drastically reduced diffusion: `DIFFUSION_AUX_PROB=0.05`, `DIFFUSION_LOSS_WEIGHT=0.10`
- 873 steps at 688ms/step — fastest config yet, more training steps within wallclock
- Post-EMA BPB 1.3028 (gap 0.025) — EMA start working well
- Quant degradation only **0.011** (1.2775 → 1.2887) — minimal diffusion acts as quantization regularizer
- Beats baseline post-quant (1.3676) by **0.079**
- Trails exp 008 v6 (1.2716) by 0.017 — diffusion overhead still costs slightly

### v5 → v6 Regression
- **Catastrophic regression**: val_bpb 1.2775 → 1.4008 (+0.123), sw BPB 1.2887 → 1.8406 (+0.552)
- Config: `DIFF_PROB=0.10` (2× v5), `DIFF_WEIGHT=0.15`, `stop_frac=0.50` (diffusion off at step 412)
- **Root cause: missing critical settings from v5.** SWA started at step 150 (v5 had 850), EMA_START_STEP=800 was not set, MTP heads zeroed out. Early-weight averaging poisoned quantization.
- Diffusion schedule itself (cutoff at 50%) worked as intended for step speed recovery
- **Result is uninformative about diffusion scheduling** — confounded by EMA/SWA regression
- Artifact 7.65MB (compact, but useless at this BPB)

### v6 → v7 Recovery and New Record
- **Full recovery**: val_bpb 1.4008 → 1.2763 (−0.125), sw BPB 1.8406 → **1.2753** (−0.565)
- Scout's diffusion schedule (8% for first 70%, then off) with v5's winning settings: `EMA_START_STEP=800`, `MTP_DELAY_ENABLED=0`, late SWA
- 867 steps at 692ms/step — slight overhead from 8% diffusion (v5 was 688ms, 873 steps)
- Post-EMA gap shrunk to **0.011** (1.2763 → 1.2869) — best EMA performance in exp 009
- **Near-zero quant degradation**: sw BPB 1.2753 vs train BPB 1.2763 — only 0.001 gap. Scheduled diffusion is an even better quant regularizer than constant diffusion
- Gap to exp 008 v6 (1.2716) is now just **0.0037**
- Artifact 12.45MB, comfortably under 16MB

### Cross-Experiment Leaderboard (all int6+lzma sw BPB)

| Rank | Experiment | Post-Quant sw BPB | Artifact |
|------|-----------|-------------------|----------|
| 1 | Exp 008 v6 | **1.2716** | 12.20MB ✅ |
| 2 | **Exp 009 v12** | **1.2746** | 12.60MB ✅ |
| 3 | Exp 009 v7 | 1.2753 | 12.45MB ✅ |
| 4 | Exp 009 v5 | 1.2887 | 12.37MB ✅ |
| 5 | Exp 002 v14 | 1.3586 (int8+lzma) | 15.82MB ✅ |
| 6 | Baseline | 1.3676 | — |

## 8x H100 Submission Run

| Metric | 1x H100 (v7 dev) | 8x H100 (v7 final) | Delta |
|--------|-------------------|---------------------|-------|
| GPUs | 1x H100 SXM | 8x H100 SXM | — |
| Steps | 867 | 6441 | 7.4× |
| Step avg (ms) | 692 | 93.16 | 7.4× faster |
| Batch tokens | 98,304 | 786,432 | 8× |
| val_bpb (raw) | 1.2763 | 1.1396 | −0.137 |
| post_ema val_bpb | 1.2869 | 1.1386 | −0.148 |
| int6 roundtrip BPB | — | 1.1424 | — |
| **int6 sw BPB** | **1.2753** | **1.1189** | **−0.156** |
| int6+lzma model | 12.45MB | 15.03MB | +2.58MB |
| Total submission | — | **15.13MB ✅** | under 16MB |
| Diffusion cutoff | step ~607 (70%) | step 4386 (70%) | — |
| SWA start | — | step 5700 | — |
| Late QAT start | — | step 5884 | — |
| Peak VRAM | — | 45.4 GB/GPU | — |
| Wallclock | 600s | 600s | — |

**Key observations:**
- 7.4× more training steps produces −0.156 BPB improvement (1.2753 → 1.1189 sw)
- Only **0.004 from SOTA** (1.1147) — the closest any experiment has gotten
- Near-zero quant degradation maintained at scale: post-EMA 1.1386 → sw 1.1189 (sw improves over roundtrip as expected)
- Artifact 15.13MB fits comfortably under 16MB limit
- Diffusion schedule scaled correctly: cutoff at step 4386 (70% of 6263 effective steps)

## Status
- [x] Proposed by scout
- [x] Approved by professor
- [x] Implemented by engineer
- [x] Tested by human / autoresearch
- [x] Analyzed
- [x] Decision: **adopt** — v7 is the submission candidate (sw BPB 1.1189, 0.004 from SOTA)
