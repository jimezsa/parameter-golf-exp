# Experiment 010: MTP + Diffusion Hybrid

## Paper / Source
- Title: Multi-Token Prediction (Meta, 2024) + AR-Diffusion Hybrid (Chen et al. 2023)
- Key idea: Combine exp 008's MTP architecture (best raw training BPB) with exp 009's diffusion regularizer (best quantization resilience). MTP drives training loss down; diffusion prevents quantization degradation.

## Hypothesis
MTP (exp 008 v6) achieves the best training BPB (1.2679) but has measurable quant degradation (sw BPB 1.2716, gap 0.004). Diffusion (exp 009 v7) has near-zero quant degradation (gap 0.001) but slightly worse training BPB (1.2763). Combining both should yield lower sw BPB than either alone: MTP's learning efficiency + diffusion's quant regularization.

## Changes from Baseline
Base: exp 009 v7 train_gpt.py (which already includes both MTP and diffusion code paths).

Config for v1 (env var overrides matching best configs from both winners):
- `MTP_DELAY_ENABLED=0` — delay adapter disabled (exp 008 v3+ finding)
- `EMA_START_STEP=800` — late EMA start (both winners)
- `DIFFUSION_LOSS_WEIGHT=0.10` — lightweight diffusion (exp 009 v7 tuned value)
- `DIFFUSION_AUX_PROB=0.08` — 8% of steps run diffusion (exp 009 v7)
- `DIFFUSION_STOP_FRAC=0.70` — disable diffusion after 70% wallclock (exp 009 v7)

No code changes from exp 009 — this is a config-only fusion experiment.

## Run Config
- GPU: 1x H100 (dev) / 8x H100 (final)
- Steps / Duration: ~900 steps in 10 min wallclock (1x H100)
- Key hyperparameters: see above

- Data prep (run once per pod):
```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```
- Run from repo root (1x H100 dev):
```bash
RUN_ID=exp010_mtp_diff SEED=1337 \
MTP_DELAY_ENABLED=0 \
EMA_START_STEP=800 \
DIFFUSION_LOSS_WEIGHT=0.10 \
DIFFUSION_AUX_PROB=0.08 \
DIFFUSION_STOP_FRAC=0.70 \
torchrun --standalone --nproc_per_node=1 experiments/010-mtp-diffusion/train_gpt.py
```
- Run for final submission (8x H100):
```bash
RUN_ID=exp010_mtp_diff SEED=1337 \
MTP_DELAY_ENABLED=0 \
EMA_START_STEP=800 \
DIFFUSION_LOSS_WEIGHT=0.10 \
DIFFUSION_AUX_PROB=0.08 \
DIFFUSION_STOP_FRAC=0.70 \
torchrun --standalone --nproc_per_node=8 experiments/010-mtp-diffusion/train_gpt.py
```

## Iteration Results

| Version | Val BPB | Post-Quant BPB | Step Time (ms) | Artifact Size | Commit | Description |
|---------|---------|----------------|-----------------|---------------|--------|-------------|
| v1      |         |                |                 |               |        | Initial fusion: exp 008 MTP + exp 009 diffusion regularizer |

- **Val BPB**: raw validation bits-per-byte before quantization
- **Post-Quant BPB**: after int6+lzma (sw eval)
- **Step Time**: average training step time in ms
- **Artifact Size**: compressed model size (target <= 16MB)
- **Commit**: short SHA of the code version used
- **Description**: what changed from the previous version

## Analysis
Pending first run.

## Status
- [x] Proposed by autoresearch (scout greenlit)
- [x] Approved by scout
- [ ] Implemented (code: 009 train_gpt.py copied, config-only changes via env vars)
- [ ] Tested on 1xH100
- [ ] Analyzed
- [ ] Decision: adopt / discard / iterate
