# Experiment: 8k Vocab + Factored Tied Embeddings

## Paper / Source
- Title: Record-mined port from the 8192-BPE ternary track into the 1.1147 BPB dense SOTA baseline
- Authors: Internal experiment port based on scout's approved Experiment 001 spec
- Link: `/home/david/.opencolab/projects/default/AGENTS/scout/findings.md`
- Key idea: Replace the 1024-token tied embedding with an 8192-token factored tied embedding so the model gets the tokenizer/BPB win without blowing the 16 MB artifact budget.

## Hypothesis
The 8k tokenizer was the largest single gain in the old ternary line. Porting it into the dense GPTQ baseline should buy a real BPB drop even after paying for the 11L -> 10L budget rebalance.

## Changes from Baseline
- Forked from `records/track_10min_16mb/jimezsa/2026-04-04_SOTA_fork_FA2/train_gpt.py`
- Defaulted the experiment to `VOCAB_SIZE=8192`, `NUM_LAYERS=10`, `BIGRAM_VOCAB_SIZE=3072`, `BIGRAM_DIM=112`, `EMBED_DIM=128`
- Added factored tied embeddings:
  - `tok_emb`: `8192 x 128`
  - `embed_proj`: `128 -> 512`
  - `embed_proj_rev`: `512 -> 128`
- Wired the factored path through:
  - main training model
  - Hessian/GPTQ calibration model
  - post-quant round-trip eval model
- Kept the existing GPTQ/LZMA routing intact so:
  - `embed_proj` and `embed_proj_rev` stay FP16 passthrough
  - `tok_emb` falls back to Int8 quantization
- Added explicit budget-overrun logging with the approved fallback chain

## Run Config
- GPU: 1x H100 (dev) / 8x H100 (final)
- Data prep:
```bash
python3 data/cached_challenge_fineweb.py --variant sp8192
```
- Default run from repo root:
```bash
RUN_ID=exp001_8k_factored \
TARGET_MB=15.9 \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 experiments/001-8k-factored-emb/train_gpt.py
```
- Multi-GPU example:
```bash
RUN_ID=exp001_8k_factored_8gpu \
TARGET_MB=15.9 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 experiments/001-8k-factored-emb/train_gpt.py
```

## Fallback Chain
If the final `Total submission size int6+lzma` exceeds 16 MB, rerun in this order:

1. `BIGRAM_VOCAB_SIZE=2048`
2. `EMBED_DIM=96`
3. `VOCAB_SIZE=4096`

Notes:
- The script now derives default dataset/tokenizer paths from `VOCAB_SIZE`, so step 3 automatically maps to `fineweb10B_sp4096` and `fineweb_4096_bpe.model` unless you override them.
- Fallback 3 still requires the corresponding assets to exist locally, for example:
```bash
python3 data/cached_challenge_fineweb.py --variant sp4096
```

## Results
| Run | BPB | Notes |
|-----|-----|-------|
|     |     | Pending human run |

## Analysis
Implementation is complete. The remaining work is a real GPU run to measure BPB and artifact size, then decide whether the fallback chain is needed.

## Status
[x] Proposed by scout
[x] Approved by professor
[x] Implemented by engineer
[ ] Tested by human
[ ] Analyzed
[ ] Decision: adopt / discard / iterate
