# Experiment 005: Ternary MoE

## Paper / Source
- Title: Ternary MoE
- Authors: scout implementation spec on top of Exp 003 ternary work
- Link: `/home/david/.opencolab/projects/default/AGENTS/scout/experiments/005-ternary-moe/README.md`
- Key idea: keep the heavy trunk ternary, replace every third FFN with a Top-1 routed MoE block, and pack the resulting model directly with ternary payloads plus LZMA instead of the old dense GPTQ path.

## Hypothesis
If Exp 003 style ternary weights buy us roughly `0.18 B/param` after packing, then a sparse MoE trunk can spend that budget on parameter capacity instead of dense FLOPs. Top-1 routing keeps the active compute close to a dense 18-layer model while MoE layers widen the effective FFN capacity toward the Rank 1 recipe.

## Architecture
- Base trunk: `18L / 512d / 8Q / 2KV / MLP_MULT=4.0`
- MoE placement: every 3rd layer, i.e. layers `3, 6, 9, 12, 15, 18`
- Experts per MoE layer: `4`
- Routing: Top-1 learned router with softmax gating
- Router regularization:
  - load-balance auxiliary loss
  - router z-loss
  - separate router optimizer / LR

## Implementation Notes
- Isolated fork in `experiments/005-ternary-moe/`; Exp 003 files are untouched
- Attention banks and dense FFN banks use ternary STE during training
- MoE expert up/down projections use `TernaryLinear`
- Export path writes `final_model.ternary.ptz` via ternary packing plus LZMA
- Roundtrip eval reloads the packed model in quantized-weight mode so it does not silently requantize to a different scale

## Run Config
From repo root:

```bash
cd /home/david/.opencolab/projects/default/parameter-golf-exp
python experiments/005-ternary-moe/train_gpt.py
```

Useful knobs:

```bash
NUM_LAYERS=18 \
MOE_EVERY_N_LAYERS=3 \
MOE_NUM_EXPERTS=4 \
MOE_ROUTER_LR=0.002 \
MOE_ROUTER_BALANCE_LOSS_WEIGHT=0.01 \
MOE_ROUTER_Z_LOSS_WEIGHT=0.001 \
python experiments/005-ternary-moe/train_gpt.py
```

## Local Verification
- `py_compile` for `train_gpt.py` and `ternary.py`
- No local end-to-end GPU run here; this runtime still depends on the real `flash_attn_interface` and CUDA stack for training

## Status
[x] Proposed by scout
[ ] Approved by professor
[x] Implemented by engineer
[ ] Tested by human
[ ] Analyzed
[ ] Decision: adopt / discard / iterate
