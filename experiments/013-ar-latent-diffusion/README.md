# Experiment 013: AR Latent Diffusion (Improved Exp 012)

## Motivation

Exp 013 inherits the Exp 012 AR-latent recipe, but the main lever here is calibration rather than architecture churn. The training-side reference to beat is Exp 012 v10 at **1.2025** pre-quant BPB on 1xH100, while Exp 011 latent v3 showed this family can hold a near-zero post-quant gap at **1.2036** sliding-window BPB.

The eval phase gets its own separate 10-minute budget (not counted against training). This means full AR self-gen calibration (64+ sequences, ~200s) is legal and free. That is the cleanest remaining lever to tighten post-quant quality without burning training wallclock.

## Improvements Over Exp 012

### 1. AR Self-Gen Calibration (primary lever)

- Generate 64+ sequences at eval time for GPTQ calibration (not limited to training data)
- Same technique as exp 009 v7 which achieved near-zero quant degradation (0.001 gap)
- Legal in eval phase — separate 10-min budget
- Expected impact: -0.035 to -0.043 BPB post-quant

## Baseline (inherited config)

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
Current throughput probes fork directly from the baseline and are **not stacked**:

- `train_gpt_01_screen.py` — reclaim screening time by skipping quant reserve when `SKIP_QUANT=1` and trimming compile warmup to the minimum compile-safe count
- `train_gpt_02_loader_prefetch.py` — vectorized shard sampling + pinned host buffers + double-buffered CUDA prefetch
- `train_gpt_03_bucketed_allreduce.py` — bucket replicated grads into a few flat all-reduces instead of one NCCL call per tensor
- `train_gpt_04_cyclic_diffusion.py` — deterministic cyclic diffusion schedule instead of per-microstep Python RNG

Legacy `train_gpt_v*.py` files are preserved for older quant / architecture probes, but they are not the current training-throughput plan.

## Run Config

- **GPU**: 1x H100 (dev) / 8x H100 (final)
- **Steps / Duration**: 10 minutes wallclock
- Data prep:

```bash
pip install brotli sentencepiece
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
python3 data/cached_challenge_fineweb.py --variant sp8192
```

- Baseline control:

```bash
RUN_ID=exp013_baseline \
SEED=1337 \
SKIP_QUANT=1 \
torchrun --standalone --nproc_per_node=1 experiments/013-ar-latent-diffusion/train_gpt.py
```

- Variant template:

```bash
RUN_ID=exp013_trial \
SEED=1337 \
SKIP_QUANT=1 \
torchrun --standalone --nproc_per_node=1 experiments/013-ar-latent-diffusion/train_gpt_XX_name.py
```

- 8x H100 follow-up template:

```bash
RUN_ID=exp013_trial_8x \
SEED=1337 \
SKIP_QUANT=1 \
torchrun --standalone --nproc_per_node=8 experiments/013-ar-latent-diffusion/train_gpt_XX_name.py
```

## Iteration Plan

### Current Priority Order

| Rank | File | Focus | Why this branch exists | Expected effect |
| ---- | ---- | ----- | ---------------------- | --------------- |
| 1 | `train_gpt_01_screen.py` | Screen-mode wallclock reclaim | Baseline still burns `GPTQ_RESERVE_SECONDS` and full compile warmup even when `SKIP_QUANT=1`; that is dead time during 1x screening | More training steps inside the same 10-minute cap with no intended model change |
| 2 | `train_gpt_02_loader_prefetch.py` | Loader + H2D overlap | `next_batch()` is still Python-heavy and does synchronous CPU assembly followed by synchronous H2D copies | Lower `step_avg`, better GPU utilization, no intended model change |
| 3 | `train_gpt_03_bucketed_allreduce.py` | Replicated-grad communication | Replicated params are reduced one tensor at a time; that is cheap on 1x and dumb on 8x | Little change on 1x, cleaner scaling on 8x |
| 4 | `train_gpt_04_cyclic_diffusion.py` | Graph-friendlier diffusion duty cycle | Per-microstep Python RNG adds jitter and makes the diffusion path harder to reason about | Small throughput win at best; main risk is BPB drift from schedule change |
| 5 | `P1`-`P11` | HP Sweep Plan (Professor Handoff) | After the throughput branch verdicts, resume the professor-defined HP/calibration experiments tracked below | Direct BPB search space; keep only sweeps that improve post-quant or clearly strengthen the pre-quant baseline |

### Autoresearch Repo Contract

```yaml
repo_path: /home/david/.opencolab/projects/default/parameter-golf-exp
editable_file_path: experiments/013-ar-latent-diffusion/train_gpt_01_screen.py  # swap per branch
run_command: RUN_ID=<run_id> SEED=1337 SKIP_QUANT=1 torchrun --standalone --nproc_per_node=1 experiments/013-ar-latent-diffusion/<file>
metric_rule:
  source: stdout
  pattern: "DIAGNOSTIC post_train val_loss:[0-9.]+ val_bpb:([0-9.]+)"
  direction: lower_is_better
secondary_metrics:
  - "step_avg:([0-9.]+)ms"
  - "stopping_early: wallclock_cap train_time:[0-9.]+ms step:([0-9]+)/"
branch_prefix: autoresearch/exp013-throughput
```

### Keep / Discard Rule

- Keep a branch immediately if `DIAGNOSTIC post_train val_bpb` improves.
- Keep a branch provisionally if `val_bpb` is within `0.002` of baseline and `step_avg` improves by at least `5%` or the run fits materially more steps under the same wallclock cap.
- Discard a branch if it regresses `val_bpb` by more than `0.002`, fails, or shows numerical instability.
- Treat `train_gpt_03_bucketed_allreduce.py` as an 8x-scaling candidate: 1x is just a smoke test, not the real verdict.

### Branch Instructions

- `train_gpt_01_screen.py`
  Why: baseline screening wastes time on quant reserve and extra compile warmup that is thrown away anyway.
  First run: `RUN_ID=exp013_01_screen SEED=1337 SKIP_QUANT=1 torchrun --standalone --nproc_per_node=1 experiments/013-ar-latent-diffusion/train_gpt_01_screen.py`
  First iteration knobs: `SCREEN_WARMUP_STEPS ∈ {1,2,4}`. Keep `SKIP_QUANT=1`.
  What to watch: completed steps under the cap, `step_avg`, and whether `val_bpb` stays effectively unchanged.

- `train_gpt_02_loader_prefetch.py`
  Why: the current loader does per-sequence Python sampling plus synchronous H2D copies; this is the ugliest remaining local throughput leak.
  First run: `RUN_ID=exp013_02_loader SEED=1337 SKIP_QUANT=1 torchrun --standalone --nproc_per_node=1 experiments/013-ar-latent-diffusion/train_gpt_02_loader_prefetch.py`
  Iteration guidance: do one short smoke run first if needed, then standard 10-minute screen. Only edit this file while iterating.
  What to watch: `step_avg`, GPU utilization, and any silent data-loader correctness issues such as shape drift or unstable loss.

- `train_gpt_03_bucketed_allreduce.py`
  Why: baseline does one `all_reduce` per replicated tensor; on 8x that becomes latency tax for no good reason.
  First run: `RUN_ID=exp013_03_bucketed SEED=1337 SKIP_QUANT=1 torchrun --standalone --nproc_per_node=1 experiments/013-ar-latent-diffusion/train_gpt_03_bucketed_allreduce.py`
  Promotion rule: if 1x smoke passes, rerun the same file on 8x before spending time tuning anything else.
  What to watch: no BPB change expected on 1x; the real signal is lower `step_avg` or cleaner scaling on 8x.

- `train_gpt_04_cyclic_diffusion.py`
  Why: this is the only branch that intentionally changes training dynamics; it trades stochastic aux routing for a deterministic duty cycle.
  First run: `RUN_ID=exp013_04_cyclic SEED=1337 SKIP_QUANT=1 torchrun --standalone --nproc_per_node=1 experiments/013-ar-latent-diffusion/train_gpt_04_cyclic_diffusion.py`
  First iteration knobs: `DIFFUSION_CYCLE_RESOLUTION ∈ {20,50,100}`, `DIFFUSION_CYCLE_OFFSET ∈ {0,7,13}`.
  What to watch: `val_bpb` first, throughput second. If BPB regresses, kill it instead of trying to talk yourself into a tiny speed win.

### Evaluation Procedure

1. Run baseline `train_gpt.py` with `SKIP_QUANT=1` if you do not already have a fresh same-seed control on the same machine.
2. Test exactly one branch at a time for a 10-minute 1xH100 screen.
3. Only after a branch survives the 1x screen should `autoresearch` edit that same file for follow-up tuning.
4. For `train_gpt_03_bucketed_allreduce.py`, move to 8x quickly because that branch is about scaling, not single-GPU heroics.

## HP Sweep Plan (Professor Handoff)

This backlog is restored for `autoresearch`. It is separate from the throughput probes above and should be picked up once the current branch screens are resolved or when a modeling-focused iteration is the better bet.

**Goal:** drive **post-training (pre-quant, 1xH100) sw BPB** below `1.2025` (exp 012 v10). This is not the post-quant gap; that remains a separate lever. Read metric names literally: "post-training BPB" means pre-quant 1xH100 BPB.

### Tier 1 — Optimizer & schedule

**P1. Peak LR + warmup + cooldown shape**
- Peak LR ∈ `{1.5e-3, 2e-3, 2.5e-3, 3e-3}`
- Warmup frac ∈ `{0.02, 0.05, 0.10}`
- Cooldown shape ∈ `{linear, cosine, 1-sqrt}`
- Why: small models are LR-sensitive, and cooldown shape can matter more than peak at this scale.
- Expected: `0.005–0.015` BPB

**P2. MuonEq-R internals**
- Newton-Schulz steps ∈ `{5, 6, 7}`
- Momentum ∈ `{0.90, 0.95, 0.98}`
- Embed/head AdamW LR ratio ∈ `{0.5x, 1x, 2x}` Muon LR
- Why: embed/head LR mismatch is a known lever; `ns_steps` trades stability for orthogonality.
- Expected: `0.005–0.012` BPB

**P3. Batch size x sequence length shape**
- Effective batch ∈ `{0.5x, 1x, 2x}` via grad accumulation
- Seq length ∈ `{1024, 2048, 4096}`
- Why: token budget is fixed; reshaping it changes gradient noise and diffusion window in tokens.
- Expected: `0.003–0.010` BPB

### Tier 2 — Latent-diffusion modeling

**P4. Latent dimension**
- ∈ `{64, 128, 256, 384}`
- Why: capacity-vs-aux-loss tradeoff; the inherited setup may be under- or over-provisioned.
- Expected: `0.003–0.010` BPB

**P5. Diffusion loss weight**
- Aux weight relative to CE ∈ `{0.5, 1.0, 1.5, 2.0}`
- Why: the current weight was set when diffusion acted more like a regularizer than a primary objective.
- Expected: `0.003–0.008` BPB

**P6. Noise schedule shape**
- Schedule ∈ `{linear, cosine, sigmoid}`
- Noise std range ∈ `{[0.1, 1.0], [0.2, 1.5], [0.3, 2.0]}`
- Why: schedule shape interacts with when diffusion is active during training.
- Expected: `0.003–0.008` BPB

### Tier 3 — Architecture micro-tweaks

**P7. RoPE base**
- ∈ `{10000, 25000, 50000, 100000}`
- Why: a smaller base often helps short-context small models.
- Expected: `0.002–0.006` BPB

**P8. Init & residual scaling**
- Depth-scaled init ∈ `{1, 1/sqrt(2L), 1/sqrt(L)}`
- Residual gain ∈ `{0.5, 0.707, 1.0}`
- Why: 11 layers is shallow enough that init-scale effects can still show up.
- Expected: `0.002–0.005` BPB

**P9. Logit softcap + tied/untied embeds**
- Softcap ∈ `{none, 30, 50}`
- Tied vs untied output projection
- Why: untied output often helps BPB but costs params; softcap can stabilize logits.
- Expected: `0.002–0.006` BPB

### Tier 4 — Regularization

**P10. Z-loss**
- Coefficient ∈ `{0, 1e-4, 1e-3, 1e-2}`
- Why: cheap logit regularizer with small but sometimes persistent gains.
- Expected: `0.002–0.005` BPB

**P11. EMA decay (eval-time)**
- Decay ∈ `{0.9990, 0.9995, 0.9999}`
- Why: eval usually uses EMA weights, so this is a direct pre-quant BPB lever.
- Expected: `0.003–0.008` BPB

### Suggested Sweep Order (~10-run Budget)

1. `P1` LR / warmup / cooldown grid (2 runs)
2. `P2` Muon `ns_steps` + embed-LR ratio (2 runs)
3. `P5` diffusion loss weight (1 run)
4. `P4` latent dim (1 run)
5. `P11` EMA decay (1 run, stackable)
6. `P6` noise schedule (1 run)
7. `P10` z-loss (1 run, stackable)
8. `P3` or `P7` depending on remaining signal (1 run)

### Iteration Policy (Autoresearch)

- Per HP proposal: `3–5` runs, not `10+`.
- Grid proposals (`P1`, `P2`, `P4`): `3–5` runs to cover the grid, then 2 seeds at the best cell if the signal is borderline.
- Single-knob proposals (`P5`, `P10`, `P11`): `2–3` runs: baseline, hypothesis point, optional third if delta is within noise.
- Post-hoc or stackable proposals: `0` training runs; evaluate on existing checkpoints when possible.
- Three-gate structure per proposal:
  1. Scout run: does the direction move BPB at all?
  2. Tune runs: narrow to the best value.
  3. Confirm run: second seed at the winner if delta is smaller than `0.005` BPB.
- Experiment-level rule: the standing `10+` iterations before kill/keep rule applies to exp 013 as a whole, not to each HP proposal in isolation.
- Killed / do-not-re-propose: Mamba, MoE, ternary, vocab-space diffusion, depth recurrence, parallel residuals, QK-gain `5.25`, WD tuning.

## Iteration Results

All log paths are relative to `experiments/013-ar-latent-diffusion/results/`.

| Variant | Val BPB | Step Time (ms) | Steps @ Cap | Commit | Log | Description |
| ------- | ------- | -------------- | ----------- | ------ | --- | ----------- |
| baseline | 1.2035 | 778 | — | 069e7d5 | — | Exp 012 latent v3 baseline rerun (no quant), confirms reference |
| baseline (fresh, pod 20260416) | 1.2063 | 794.79 | 740 | fdbfcca | [fresh_baseline_20260416.log](results/fresh_baseline_20260416.log) | Same-pod `SKIP_QUANT=1` control for throughput-probe reference; 588.1s train, peak 49.97 GiB, 135.4 MB raw. |
| `train_gpt_01_screen.py` v1-probe | — | 745.47 | 10 | 234a1b5 | [branch01_screen_sw2_20260416_partial10steps.log](results/branch01_screen_sw2_20260416_partial10steps.log) | SIGHUP'd at step 10 during initial nohup setup; first attempt of v1 before official run. |
| `train_gpt_01_screen.py` v1 | 1.2072 | 805 | 731 | 234a1b5 | — | Screen-mode warmup=2, SKIP_QUANT=1, baseline rerun |
| `train_gpt_01_screen.py` v2-attempt | FAIL | — | 0 | 234a1b5 | [branch01_screen_sw2_20260416_v2.log](results/branch01_screen_sw2_20260416_v2.log) | Crashed at startup: tokenizer file `./data/tokenizers/fineweb_8192_bpe.model` missing — data prep had not yet been run on this pod. Re-ran successfully as v2 (SwiGELU) after `cached_challenge_fineweb.py --variant sp8192`. |
| `train_gpt_01_screen.py` v2 | 1.2035 | 779 | 756 | 234a1b5 | — | SwiGELU hidden_dim=1344 probe — inconclusive (later re-run at 1.2078) |
| `train_gpt_01_screen.py` v3 | 1.2036 | 794 | 756 | 234a1b5 | [branch01_screen_sw2_20260416_v3.log](results/branch01_screen_sw2_20260416_v3.log) | Intended delayed-diffusion window 25%–60% — **code change not applied**, effectively a v2 noise re-run; recipe noise floor ~±0.001 BPB. |
| `train_gpt_01_screen.py` v4 | **1.1996** | 761.57 | 788 | 5729718 | [branch01_screen_sw2_20260416_v4_startfrac025.log](results/branch01_screen_sw2_20260416_v4_startfrac025.log) | **Real delayed-diffusion window** `DIFFUSION_START_FRAC=0.25`, `DIFFUSION_STOP_FRAC=0.60` — diffusion startup at step 200 / cutoff at step 467 (confirmed in log). −0.0076 BPB vs v1 baseline on seed=1337; new exp 013 best on this seed. 600.1s train, 49.97 GiB peak, 135.4 MB raw. |
| `train_gpt_01_screen.py` v5 | 1.2033 | 763.43 | 787 | 29225e5 | [branch01_screen_sw2_20260416_v5_seed2026.log](results/branch01_screen_sw2_20260416_v5_seed2026.log) | **Confirm-seed of v4** (`SEED=2026`, all else identical). Diffusion startup at step 199 / cutoff at step 466 (confirmed). −0.0039 vs v1 baseline, +0.0037 vs v4. Seed-averaged v4+v5 = 1.2015 → delayed-diffusion is a real but smaller effect than v4 alone suggested. 600.8s train, 49.9 GiB peak, 135.4 MB raw. |
| `train_gpt_01_screen.py` v6 | 1.2017 (pre-quant) / **FAIL** (post-quant) | 762.59 | 772 | 0d3d436 | [branch01_screen_sw2_20260416_v6_arsg.log](results/branch01_screen_sw2_20260416_v6_arsg.log) | **AR self-gen GPTQ calibration** — training completed cleanly, pre-quant sw val_bpb=1.2017 (seed=1337, within v4/v5 seed noise, confirms training untouched). **Post-quant crashed**: `_build_ar_selfgen_inputs` called `hessian_model.forward_logits(x)` but `_HessianGPT` had no such method — only `forward(input_ids, target_ids)` returning loss. Fix in v6b: add `forward_logits` method returning `(B, T, V)` logits, refactored via shared `_trunk` helper. |
| `train_gpt_01_screen.py` v6b | KILLED | — | 10 | 2e06b7c | [branch01_screen_sw2_20260416_v6b_arsg.log](results/branch01_screen_sw2_20260416_v6b_arsg.log) | **v6 retry, killed at step 10 by user directive ("pls dont run any quant")** before training phase completed. `forward_logits` fix (commit 2e06b7c) was in place and `diffusion_startup:step:196 frac:0.25` confirmed firing, but the run had `SKIP_QUANT=0` — user does not want any quant-pipeline runs during iteration. AR self-gen GPTQ calibration parked; iteration continues on pre-quant BPB only going forward. |
| `train_gpt_01_screen.py` v7 | 1.2014 | 760.66 | 789 | 433d0f2 | [branch01_screen_sw2_20260416_v7_seed42.log](results/branch01_screen_sw2_20260416_v7_seed42.log) | **Third confirm-seed of delayed-diffusion recipe** (`SEED=42`, `DIFFUSION_START_FRAC=0.25`, `DIFFUSION_STOP_FRAC=0.60`, `SKIP_QUANT=1`). Diffusion startup step 200 / cutoff step 468 confirmed. −0.0058 vs v1, +0.0018 vs v4, −0.0019 vs v5. **3-seed delayed-diffusion average (v4+v5+v7) = 1.2014**, vs v1 single-seed baseline 1.2072 → locked **−0.0058 BPB** effect. 600.2s train, 49.9 GiB peak, 135.4 MB raw. |
| `train_gpt_01_screen.py` v8 | 1.1999 | 762.98 | 787 | c47dfa1 | [branch01_screen_sw2_20260416_v8_startfrac020.log](results/branch01_screen_sw2_20260416_v8_startfrac020.log) | **Window-left-edge scan** of locked delayed-diffusion recipe: `DIFFUSION_START_FRAC=0.20` (vs 0.25 default), `DIFFUSION_STOP_FRAC=0.60`, `SEED=1337`, `SKIP_QUANT=1`. Diffusion startup step 161 / cutoff step 465 confirmed (frac:0.20 / frac:0.60). Pre-quant val_bpb=1.19994 → effectively tied with v4 (1.1996, +0.0003 within ±0.001 noise floor); left-edge shift from 0.25→0.20 is BPB-neutral. 600.5s train, peak 49.9 GiB. Next: scan `START_FRAC=0.30` (v8b) to complete the window-edge sweep. |
| `train_gpt_01_screen.py` v8b | **1.1994** | 760.05 | 790 | PENDING | [branch01_screen_sw2_20260416_v8b_startfrac030.log](results/branch01_screen_sw2_20260416_v8b_startfrac030.log) | **Window-right-shift scan** of locked delayed-diffusion recipe: `DIFFUSION_START_FRAC=0.30` (vs 0.25 default), `DIFFUSION_STOP_FRAC=0.60`, `SEED=1337`, `SKIP_QUANT=1`. Diffusion startup step 240 / cutoff step 469 confirmed (frac:0.30 / frac:0.60). Pre-quant val_bpb=1.19943 → effectively tied with v4 (1.1996, −0.0002 within ±0.001 noise floor). **Full window-edge sweep complete (v4=1.1996 @ 0.25, v8=1.1999 @ 0.20, v8b=1.1994 @ 0.30): recipe is robust to `start_frac ∈ [0.20, 0.30]`; no clear winner.** 600.4s train, peak 49.9 GiB. Next: v9 scans `DIFFUSION_AUX_PROB ∈ {0.03, 0.08}` to test mechanism strength. |
| `train_gpt_02_loader_prefetch.py` | pending | pending | pending | — | — | Vectorized loader sampling with double-buffered H2D prefetch |
| `train_gpt_03_bucketed_allreduce.py` | pending | pending | pending | — | — | Coalesced replicated-grad all-reduce path |
| `train_gpt_04_cyclic_diffusion.py` | pending | pending | pending | — | — | Deterministic cyclic diffusion duty cycle |

## Status

- [x] Forked from exp 012
- [x] Added isolated throughput branches from baseline
- [x] Run fresh baseline control with `SKIP_QUANT=1` (1.2063 @ 794.79ms, 740 steps)
- [ ] Screen `train_gpt_01_screen.py`
- [ ] Screen `train_gpt_02_loader_prefetch.py`
- [ ] Smoke test `train_gpt_03_bucketed_allreduce.py`, then move to 8x if clean
- [ ] Screen `train_gpt_04_cyclic_diffusion.py`
- [ ] Keep / discard each branch before any stacking
- [ ] Resume professor HP sweep backlog (`P1`-`P11`) after the throughput branch verdicts
