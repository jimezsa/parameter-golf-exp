# Experiment 011: AR-Diffusion SP8192

## Paper / Source
- Inherits diffusion approach from Exp 009 (Chen et al. 2023, Lin et al. 2022, Zheng et al. 2023)
- **Stack rebase** onto the SP8192 meta from the April 2026 leaderboard top 5
- Key references: clarkkev PR #1394 (SP8192 base), bigbag 1.0810 submission (current SOTA)

## Hypothesis
Exp 009's hybrid AR-diffusion produces near-zero quantization degradation (0.001 BPB gap).
The leaderboard top 5 all use SP8192 + brotli-11 + SDClip + higher WD — but show 0.01–0.015 quant loss.
Rebasing our diffusion trick onto the SP8192 stack should compound both advantages:
better base BPB from the tokenizer + better quant survival from diffusion regularization.

## Changes from Exp 009 (SP1024 baseline)
Fork from exp 009 v12 (best 1xH100: sw BPB 1.2746).

### P1 — Stack rebase (already implemented by codex):
- **SP8192 tokenizer** (was SP1024) — single biggest lever
- **WD=0.090** (was 0.04) — enables all-int6 GPTQ via better compression
- **Brotli-11 + byte shuffle** compression (was LZMA)
- **SDClip** (k=12.85σ int6, k=20σ int8 embeds) — principled quantization
- **GPTQ embedding quant** at int8 with Hessian-aware calibration
- **MuonEq-R** — row-normalized Muon optimizer
- **QK-Gain 5.0** (was 1.5)
- **MLP_MULT=4** (matching SOTA)
- **Shuffled shard sampling**
- **Smear disabled** by default
- **Optional skip-gates**
- Diffusion components preserved: mask_embed, time_proj, bidirectional aux pass, scheduled cutoff

### P2 — To iterate:
- QK-Gain tuning (5.0 vs 5.25 vs 4.5)
- Depth recurrence layers 4-5, start at 35-50% training
- EMA decay tuning (0.997 vs 0.9965)
- Warmdown schedule (72% vs current)

### P3 — Novel (scout's "Annealed Handoff"):
- Overlap diffusion warmdown (8%→0% from 60-75%) with SDClip QAT warmup (starting 60%)
- Diffusion-robustness absorbs quantization shock during QAT transition
- Expected: even lower quant degradation than current near-zero

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
RUN_ID=exp011_ar_diffusion_sp8192 \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 experiments/011-ar-diffusion-sp8192/train_gpt.py
```
- Run for final submission (8x H100):
```bash
RUN_ID=exp011_ar_diffusion_sp8192 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 experiments/011-ar-diffusion-sp8192/train_gpt.py
```

## Iteration Results

| Version | Val BPB | Post-Quant BPB | Step Time (ms) | Artifact Size | Commit | Description |
|---------|---------|----------------|----------------|---------------|--------|-------------|
| v1      |         |                |                |               |        | Clean SP8192 rebase, diffusion OFF (pure baseline) |
| v2      |         |                |                |               |        | Add diffusion schedule back (8% aux, off at 70%) |

## Iteration Plan
1. **v1**: Clean run with diffusion disabled — establish SP8192 reference BPB
2. **v2**: Enable diffusion (8% aux prob, off at 70%) — measure delta
3. **v3+**: Scout's Annealed Handoff (overlap diffusion warmdown with QAT warmup)
4. **v4+**: P2 techniques (depth recurrence, EMA tuning, warmdown schedule)
5. Target: beat 1.0810 BPB (current public SOTA)

## Analysis
Pending first run.

## Status
- [x] Proposed by professor + scout
- [x] Approved by professor
- [x] Implemented by engineer (SP8192 rebase)
- [ ] Tested by human
- [ ] Analyzed
- [ ] Decision: pending
