# Experiment 013: AR Latent Diffusion (Improved Exp 012)

## Motivation

Exp 012 (latent v3) achieved post-quant sw BPB 1.2036 on 1xH100 — our project best — but left significant performance on the table due to weak quantization calibration. The quant gap was 0.048 BPB, far worse than exp 009's ~0.001 gap, because 012 used only 32 batches of training data for GPTQ calibration instead of AR-generated sequences.

The eval phase gets its own separate 10-minute budget (not counted against training). This means full AR self-gen calibration (64+ sequences, ~200s) is legal and free. That alone should cut the quant gap from 0.048 to ~0.005–0.010, yielding an estimated post-quant BPB of **~1.10–1.105** on 1xH100.

## Improvements Over Exp 012

### 1. AR Self-Gen Calibration (primary lever)

- Generate 64+ sequences at eval time for GPTQ calibration (not limited to training data)
- Same technique as exp 009 v7 which achieved near-zero quant degradation (0.001 gap)
- Legal in eval phase — separate 10-min budget
- Expected impact: -0.035 to -0.043 BPB post-quant

## Baseline (inherited from Exp 012 latent v3)

- SP8192 tokenizer, 11L/512d
- WD=0.090, brotli-11 + byte-shuffle
- SDClip, GPTQ int8 embeds, MuonEq-R
- QK-Gain 5.0
- Latent MSE diffusion (AUX_PROB=0.05, STOP_FRAC=0.60)
- LATE_QAT_THRESHOLD=0.15, GPTQ_CALIB_BATCHES=32
- Pre-quant sw BPB: 1.2045 (1xH100)
- Post-quant sw BPB: 1.2036 (1xH100, quant gap 0.0005)

## File Convention

The baseline `train_gpt.py` is **read-only** — never edited directly.
Each version creates a new file forked from the best prior:

- `train_gpt_v1.py` — forked from `train_gpt.py` (baseline)
- `train_gpt_v2.py` — forked from best of v1 or baseline
- `train_gpt_v3.py` — forked from best prior version
- etc.

This keeps diffs clean and rollback trivial. The autoresearch contract's
`editable_file_path` updates per version.

## Run Config

- **GPU**: 1x H100 (dev) / 8x H100 (final)
- **Steps / Duration**: 10 minutes wallclock
- Data prep:

```bash
pip install brotli sentencepiece
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
python3 data/cached_challenge_fineweb.py --variant sp8192
```

- Run (1x H100 dev, replace `_v1` with target version):

```bash
RUN_ID=exp013_v1 \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 experiments/013-ar-latent-diffusion/train_gpt_v1.py
```

- Run (8x H100 final):

```bash
RUN_ID=exp013_vN \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 experiments/013-ar-latent-diffusion/train_gpt_vN.py
```

## Iteration Plan

| Version | File                | Focus                   | Description                                                                                          |
| ------- | ------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------- |
| v1      | `train_gpt_v1.py`   | Fused Cross-Entropy     | Fork baseline, replace `F.cross_entropy()` with Liger Kernel fused CE — 10-20% loss-computation speedup |
| v2      | `train_gpt_v2.py`   | SwiGELU Activation      | Replace LeakyReLU² with SwiGELU — potentially better gradient flow and expressiveness |
| v3      | `train_gpt_v3.py`   | Delayed Diffusion Start | Port delayed diffusion window (25%–60%) from exp 012 v10 — proven +0.002 BPB win |

## HP Sweep Plan (autoresearch handoff)

**Goal:** drive **post-training (pre-quant, 1xH100) sw BPB** below 1.2025 (exp 012 v10). This is NOT the post-quant gap — that is a separate, later lever. Read metric names literally: "post-training BPB" = pre-quant 1xH100 BPB.

Proposals ranked by expected single-lever impact. `autoresearch` should pick them up after v1–v3 land.

### Tier 1 — Optimizer & schedule (highest single-lever impact)

**P1. Peak LR + warmup + cooldown shape**
- Peak LR ∈ {1.5e-3, 2e-3 (assumed current), 2.5e-3, 3e-3}
- Warmup frac ∈ {0.02, 0.05, 0.10}
- Cooldown shape ∈ {linear, cosine, 1-sqrt}
- *Why:* small models are LR-sensitive; cooldown shape matters more than peak at this scale.
- *Expected:* 0.005–0.015 BPB

**P2. MuonEq-R internals**
- Newton-Schulz steps ∈ {5, 6, 7}
- Momentum ∈ {0.90, 0.95, 0.98}
- Embed/head AdamW LR ratio ∈ {0.5×, 1×, 2× Muon LR}
- *Why:* embed/head LR mismatch is a known leader move; ns_steps trades stability for orthogonality.
- *Expected:* 0.005–0.012 BPB

**P3. Batch size × sequence length shape**
- Effective batch ∈ {0.5×, 1×, 2×} (via grad accum)
- Seq length ∈ {1024, 2048, 4096}
- *Why:* token budget is fixed; reshaping it changes gradient noise and diffusion window in tokens.
- *Expected:* 0.003–0.010 BPB

### Tier 2 — Latent-diffusion modeling (not quant)

**P4. Latent dimension**
- ∈ {64, 128, 256 (likely baseline), 384}
- *Why:* capacity-vs-aux-loss tradeoff; v10 may be under- or over-provisioned.
- *Expected:* 0.003–0.010 BPB

**P5. Diffusion loss weight**
- aux weight relative to CE ∈ {0.5, 1.0, 1.5, 2.0}
- *Why:* current weight set when diffusion was a regularizer, not a primary objective.
- *Expected:* 0.003–0.008 BPB

**P6. Noise schedule shape**
- Schedule ∈ {linear, cosine, sigmoid}
- Noise std range ∈ {[0.1, 1.0], [0.2, 1.5], [0.3, 2.0]}
- *Why:* schedule shape interacts with when in training diffusion is active.
- *Expected:* 0.003–0.008 BPB

### Tier 3 — Architecture micro-tweaks

**P7. RoPE base**
- ∈ {10000, 25000, 50000, 100000}
- *Why:* smaller base often helps short-context small models.
- *Expected:* 0.002–0.006 BPB

**P8. Init & residual scaling**
- Depth-scaled init ∈ {1, 1/√(2L), 1/√L}
- Residual gain ∈ {0.5, 0.707, 1.0}
- *Why:* 11L is shallow enough that init-scale effects are visible.
- *Expected:* 0.002–0.005 BPB

**P9. Logit softcap + tied/untied embeds**
- Softcap ∈ {none, 30, 50}
- Tied vs untied output projection
- *Why:* untied costs params (bad for budget) but often gains BPB; softcap stabilizes.
- *Expected:* 0.002–0.006 BPB

### Tier 4 — Regularization

**P10. Z-loss**
- Coefficient ∈ {0, 1e-4, 1e-3, 1e-2}
- *Why:* cheap logit regularizer; small but consistent gain.
- *Expected:* 0.002–0.005 BPB

**P11. EMA decay (eval-time)**
- Decay ∈ {0.9990, 0.9995, 0.9999}
- *Why:* eval typically uses EMA weights — direct pre-quant BPB lever.
- *Expected:* 0.003–0.008 BPB

### Suggested sweep order (~10-run budget)

1. **P1** LR / warmup / cooldown grid (2 runs) — biggest lever
2. **P2** Muon ns_steps + embed-LR ratio (2 runs)
3. **P5** diffusion loss weight (1 run)
4. **P4** latent dim (1 run)
5. **P11** EMA decay (1 run, stackable)
6. **P6** noise schedule (1 run)
7. **P10** z-loss (1 run, stackable)
8. **P3** or **P7** depending on remaining signal (1 run)

## Iteration Policy (autoresearch)

- **Per HP proposal:** 3–5 runs, not 10+.
  - Grid proposals (P1, P2, P4): 3–5 runs to cover the grid. 2 seeds at the best cell if the signal is borderline.
  - Single-knob proposals (P5, P10, P11): 2–3 runs — baseline, hypothesis point, optional third if delta is within noise.
  - Post-hoc / stackable proposals: 0 training runs — evaluate on existing checkpoints.
- **Three-gate structure per proposal:**
  1. *Scout run (1):* does the direction move BPB at all? Keep or kill the proposal.
  2. *Tune runs (1–3):* narrow to the best value.
  3. *Confirm run (1, optional):* second seed at the winner if delta < 0.005 BPB.
- **Experiment-level rule:** the standing **10+ iterations before kill/keep** rule applies to exp 013 as a whole, not to individual HP proposals. Once the best HP stack is picked, run 10+ seeded confirmations before declaring exp 013 ready for 8xH100 promotion.
- **Killed / do-not-re-propose:** Mamba, MoE, ternary, vocab-space diffusion, depth recurrence, parallel residuals, QK-gain 5.25, WD tuning.

## Iteration Results

| Version   | Val BPB | Post-Quant BPB | Step Time (ms) | Artifact (bytes) | Commit | Description |
| --------- | ------- | -------------- | -------------- | ---------------- | ------ | ----------- |
| baseline  | 1.2035  | —              | 778            | —                | 069e7d5 | Exp 012 latent v3 baseline rerun (no quant), confirms reference |
| v1        | 1.2072  | —              | 805            | —                | —       | Liger fused CE — disabled due to torch.compile conflict, +0.003 regression |

## Status

- [x] Forked from exp 012
- [ ] v1: Fused cross-entropy (Liger Kernel)
- [ ] Iteration runs
- [ ] Decision: adopt / discard / iterate
