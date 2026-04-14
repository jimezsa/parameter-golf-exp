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

### 2. Depth Recurrence (layers 4-5, 3-layer stack)
- Used by all top-5 leaderboard submissions
- Reuses layers 4-5-6 for a second pass — adds depth without parameters
- Expected impact: -0.01 to -0.02 BPB

### 3. Parallel Residuals (GPT-J style from layer 7+)
- Consistent leaderboard gain across top submissions
- MLP and attention in parallel instead of sequential
- Expected impact: -0.005 to -0.010 BPB

### 4. QK-Gain Tune (5.0 → 5.25)
- Minor but free — observed in leaderboard leaders
- Expected impact: -0.001 to -0.003 BPB

## Baseline (inherited from Exp 012 latent v3)

- SP8192 tokenizer, 11L/512d
- WD=0.090, brotli-11 + byte-shuffle
- SDClip, GPTQ int8 embeds, MuonEq-R
- QK-Gain 5.0
- Latent MSE diffusion (AUX_PROB=0.05, STOP_FRAC=0.60)
- LATE_QAT_THRESHOLD=0.15, GPTQ_CALIB_BATCHES=32
- Pre-quant sw BPB: 1.2045 (1xH100)
- Post-quant sw BPB: 1.2036 (1xH100, quant gap 0.0005)

## Run Config
- **GPU**: 1x H100 (dev) / 8x H100 (final)
- **Steps / Duration**: 10 minutes wallclock
- Data prep:
```bash
python3 data/cached_challenge_fineweb.py --variant sp8192
```
- Dependencies:
```bash
pip install brotli
```
- Run (1x H100 dev):
```bash
RUN_ID=exp013_ar_latent_diffusion \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 experiments/013-ar-latent-diffusion/train_gpt.py
```
- Run (8x H100 final):
```bash
RUN_ID=exp013_ar_latent_diffusion \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 experiments/013-ar-latent-diffusion/train_gpt.py
```

## Iteration Plan

| Version | Focus | Description |
|---------|-------|-------------|
| v1 | AR self-gen calibration | Add 64-seq AR calibration during eval phase |
| v2 | Depth recurrence | Add 3-layer recurrence on layers 4-5 |
| v3 | Parallel residuals | GPT-J style from layer 7+ |
| v4 | QK-gain + tuning | Fine-tune QK-gain 5.25, sweep other small knobs |
| v5+ | Integration | Best combo, final tuning |

## Iteration Results

| Version | Val BPB | Post-Quant BPB | Step Time (ms) | Artifact (bytes) | Commit | Description |
|---------|---------|----------------|----------------|-------------------|--------|-------------|
| (pending) | | | | | | |

## Status
- [x] Forked from exp 012
- [ ] v1: AR self-gen calibration
- [ ] v2: Depth recurrence
- [ ] v3: Parallel residuals
- [ ] v4: QK-gain tuning
- [ ] Integration run
- [ ] Decision: adopt / discard / iterate
