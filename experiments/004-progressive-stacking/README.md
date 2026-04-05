# Experiment: Progressive Layer Stacking

## Paper / Source
- Title: Progressive Layer Stacking & LZMA Entropy Estimator
- Authors: local implementation spec on top of Exp 003 BitNet work
- Link: `/home/david/.opencolab/projects/default/AGENTS/scout/spec-004-progressive-stacking.md`
- Key idea: train a shallow 12-layer model for fast early throughput, then duplicate it into a 24-layer model mid-run once the compression math is validated for ternary weights.

## Hypothesis
If the ternary projection weights really compress to `<= 0.20 bytes/param` after packing plus LZMA, we can afford a deeper 24-layer BitNet trunk inside the 16MB artifact budget and use progressive stacking to recover early-step throughput.

## Changes From Baseline
- Forked the tracked `experiments/003-bitnet/` baseline into `experiments/004-progressive-stacking/`
- Added `estimate_lzma_entropy.py`, a standalone compression blocker for the 24L ternary stack
- Extended `ternary.py` with both `base3` and `two_bit` byte-pack formats so the estimator can compare the spec wording against the actual Exp 003 packing path
- Reworked `train_gpt.py` into a two-phase trainer:
  - starts with a real 12-layer model (`PROGRESSIVE_STACK_LAYERS=12`)
  - rebuilds into a 24-layer model at the 6-minute wallclock trigger (`PROGRESSIVE_STACK_TRANSITION_SECONDS=360`)
  - duplicates banked weights, scalar block parameters, and skip weights with optimizer-state transfer
  - zeroes the inserted odd layers' attention output projections and MLP down projections so the new copies begin as identity layers

## Run Config
- Compression blocker from repo root:
```bash
cd /home/david/.opencolab/projects/default/parameter-golf-exp
python experiments/004-progressive-stacking/estimate_lzma_entropy.py
```
- Full training run from repo root:
```bash
cd /home/david/.opencolab/projects/default/parameter-golf-exp
python experiments/004-progressive-stacking/train_gpt.py
```
- Key defaults for Deliverable 2:
  - `NUM_LAYERS=24`
  - `PROGRESSIVE_STACK_LAYERS=12`
  - `PROGRESSIVE_STACK_TRANSITION_SECONDS=360`
  - `NUM_HEADS=8`
  - `NUM_KV_HEADS=2`
  - `MLP_MULT=4.0`
- Optional single-format checks:
```bash
cd /home/david/.opencolab/projects/default/parameter-golf-exp
python experiments/004-progressive-stacking/estimate_lzma_entropy.py --pack-format base3
python experiments/004-progressive-stacking/estimate_lzma_entropy.py --pack-format two_bit --blocking-format two_bit
```

## Results
| Run | BPB | Notes |
|-----|-----|-------|
| local entropy gate | n/a | `base3 = 0.1917 B/param`, passed the `<= 0.20` blocker |
|     |     | Pending human full training run |

## Analysis
Deliverable 1 cleared locally, so Deliverable 2 is now wired into the trainer. The transition happens at the nearest step after the wallclock trigger, expands the 12-layer checkpoint into a 24-layer checkpoint with deterministic `L_new[2i] / L_new[2i+1]` duplication, transfers optimizer state onto the expanded parameters, and preserves the pre-transition forward path by zero-initializing the inserted layers' output projections.

Local verification:
- `py_compile` passed for `train_gpt.py`, `estimate_lzma_entropy.py`, and `ternary.py`
- A CPU-only harness with a stubbed `flash_attn_interface` successfully expanded a 12-layer state dict into 24 layers and loaded it back into a fresh model
- This runtime does not have the real `flash_attn_interface`, so no local end-to-end training run was possible here

## Status
[x] Proposed by scout
[ ] Approved by professor
[x] Implemented by engineer
[ ] Tested by human
[ ] Analyzed
[ ] Decision: adopt / discard / iterate
