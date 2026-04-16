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

## Iteration Results

| Variant | Val BPB | Step Time (ms) | Steps @ Cap | Commit | Description |
| ------- | ------- | -------------- | ----------- | ------ | ----------- |
| baseline | 1.2035 | 778 | — | 069e7d5 | Exp 012 latent v3 baseline rerun (no quant), confirms reference |
| `train_gpt_01_screen.py` | pending | pending | pending | — | Skip quant reserve during screening and trim compile warmup |
| `train_gpt_02_loader_prefetch.py` | pending | pending | pending | — | Vectorized loader sampling with double-buffered H2D prefetch |
| `train_gpt_03_bucketed_allreduce.py` | pending | pending | pending | — | Coalesced replicated-grad all-reduce path |
| `train_gpt_04_cyclic_diffusion.py` | pending | pending | pending | — | Deterministic cyclic diffusion duty cycle |

## Status

- [x] Forked from exp 012
- [x] Added isolated throughput branches from baseline
- [ ] Run fresh baseline control with `SKIP_QUANT=1`
- [ ] Screen `train_gpt_01_screen.py`
- [ ] Screen `train_gpt_02_loader_prefetch.py`
- [ ] Smoke test `train_gpt_03_bucketed_allreduce.py`, then move to 8x if clean
- [ ] Screen `train_gpt_04_cyclic_diffusion.py`
- [ ] Keep / discard each branch before any stacking
