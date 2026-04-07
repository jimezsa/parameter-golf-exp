# Experiment 000: Baseline Reference

## Paper / Source
- Title: Parameter Golf starter baseline (`baseline/train_gpt.py`)
- Authors: Parameter Golf organizers
- Link: `baseline/train_gpt.py`
- Key idea: Unmodified starter baseline to establish reference val BPB and per-step timing on our hardware before running experiments.

## Hypothesis
No hypothesis — this is a reference run. We need val BPB and average training step time on 1xH100 so all subsequent experiments have a comparable baseline measured on the same hardware.

## Changes from Baseline
None. This is an exact copy of `baseline/train_gpt.py`.

## Run Config
- GPU: 1x H100 (dev)
- Data prep (if not already done):
```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```
- Run from repo root:
```bash
RUN_ID=exp000_baseline \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 experiments/000-baseline/train_gpt.py
```

## Key Metrics to Record
1. **val_bpb** — final validation bits per byte
2. **Average step time** — mean ms/step from training logs
3. **Total wallclock** — end-to-end training time
4. **Compressed model size** — final artifact size in bytes

## Results
| Run | Val BPB | Avg Step Time | Wallclock | Model Size | Notes |
|-----|---------|---------------|-----------|------------|-------|
|     |         |               |           |            | Pending human run |

## Analysis
Pending first run.

## Status
[x] Proposed by professor
[x] Approved by professor
[x] Implemented (unmodified starter baseline copy)
[ ] Tested by human
[ ] Analyzed
[ ] Decision: reference baseline established
