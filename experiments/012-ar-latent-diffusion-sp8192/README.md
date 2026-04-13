# Experiment 012: AR Latent Diffusion SP8192

## Paper / Source
- Inherits latent MSE diffusion approach from Exp 011 (latent v3 — project best)
- Stack: SP8192 + brotli-11 + byte-shuffle + SDClip + MuonEq-R + QK-Gain 5.0
- Key insight: latent-space diffusion (hidden states, not vocab logits) gives zero step-time overhead and near-zero quant degradation (0.0005 BPB gap)

## Hypothesis
Exp 011 latent v3 achieved post-quant sw BPB 1.2036 (15.90MB) with near-zero quant gap. This experiment continues iteration from that baseline to push BPB lower via depth recurrence, parallel residuals, QK-gain tuning, and compression improvements.

## Changes from Exp 011 Latent v3 (baseline)
Starting point: `train_gpt_latent.py` from exp 011 (latent v3 config).

### Baseline config (inherited):
- SP8192 tokenizer
- 11L/512d architecture
- WD=0.090
- Brotli-11 + byte-shuffle compression
- SDClip (k=12.85sigma int6, k=20sigma int8 embeds)
- GPTQ embedding quant at int8
- MuonEq-R optimizer
- QK-Gain 5.0
- Latent MSE diffusion (DIFFUSION_AUX_PROB=0.05, DIFFUSION_STOP_FRAC=0.60)
- LATE_QAT_THRESHOLD=0.15
- GPTQ_CALIB_BATCHES=32
- SWA_ENABLED=0

### Planned iterations:
- Depth recurrence (layers 4-5, top-5 leaderboard feature)
- Parallel residuals (GPT-J style from layer 7+)
- QK-Gain tuning (5.0 vs 5.25 vs 4.5)
- EMA decay tuning
- Compression/pruning optimization

## Run Config
- **GPU**: 1x H100 (dev) / 8x H100 (final)
- **Steps / Duration**: 10 minutes wallclock
- Data prep (run once per pod):
```bash
python3 data/cached_challenge_fineweb.py --variant sp8192
```
- Dependencies:
```bash
pip install brotli
```
- Run from repo root (1x H100 dev):
```bash
RUN_ID=exp012_ar_latent_diffusion_sp8192 \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 experiments/012-ar-latent-diffusion-sp8192/train_gpt.py
```
- Run for final submission (8x H100):
```bash
RUN_ID=exp012_ar_latent_diffusion_sp8192 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 experiments/012-ar-latent-diffusion-sp8192/train_gpt.py
```

## Iteration Results

| Version | Val BPB | Post-Quant BPB | Step Time (ms) | Artifact Size | Commit | Description |
|---------|---------|----------------|----------------|---------------|--------|-------------|
| v1      |         |                |                |               |        | Baseline — identical to exp 011 latent v3 config |

- **Val BPB**: raw validation bits-per-byte before quantization
- **Post-Quant BPB**: after int6+brotli (sliding window)
- **Step Time**: average training step time in ms
- **Artifact Size**: compressed model size (target <= 16MB)
- **Commit**: short SHA of the code version used
- **Description**: what changed from the previous version

## Analysis
Starting from exp 011 latent v3 (post-quant sw BPB 1.2036, 15.90MB). This is the new project baseline.

## Status
[x] Forked from exp 011 latent v3
[ ] v1 baseline verification
[ ] Iterate toward leaderboard competitive (target: <1.10 on 8xH100)
[ ] Decision: adopt / discard / iterate
