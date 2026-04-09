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
| v1      |         |                |                |               |        | Initial run — baseline config, 25% diffusion gate, diffusion weight 0.3 |

- **Val BPB**: raw validation bits-per-byte before quantization (AR pass)
- **Post-Quant BPB**: after int8+zlib (or int6+lzma if applicable)
- **Step Time**: average training step time in ms
- **Artifact Size**: compressed model size (target ≤ 16MB)
- **Commit**: short SHA of the code version used
- **Description**: what changed from the previous version

## Analysis
Pending first run.

## Status
- [x] Proposed by scout
- [x] Approved by professor
- [x] Implemented by engineer
- [ ] Tested by human / autoresearch
- [ ] Analyzed
- [ ] Decision: adopt / discard / iterate
