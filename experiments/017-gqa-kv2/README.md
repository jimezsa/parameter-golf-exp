# Experiment 017: GQA KV2 (SOTA Fork)

## Hypothesis

Reducing KV heads from 4 → 2 on top of the current SOTA (codemath3000 @ 1.05651 BPB) saves ~1.44M params across 11 layers with minimal quality loss, potentially improving post-quant BPB by giving the quantizer fewer parameters to compress.

## Base

Forked from **codemath3000's SOTA submission** ([PR #2135](https://github.com/openai/parameter-golf/pull/2135) / [PR #2130](https://github.com/openai/parameter-golf/pull/2130)), not our exp 016 code.

SOTA stack: 11L/512d, 8Q/4KV heads, MLP 4x, layer recurrence (L3-5 frac=0.35), parallel decoder (L8+), XSA, BOS-fixed SmearGate, SparseAttnGate (scale=0.5), Muon+Adam, EMA, GPTQ int6 + LQER asymmetric rank-4, CaseOps tokenizer, token-only n-gram tilt, AsymLogit, phased LoRA TTT.

## Changes from SOTA

- `NUM_KV_HEADS`: 4 → **2** (8 query heads, 2 KV heads = 4:1 grouping)
- KV projection params per layer: 65,536 (was 131,072) — 50% reduction in K+V
- XSA group size: 4 (was 2)
- All other hyperparameters unchanged from SOTA

## Run Commands

Data prep:

```bash
pip install brotli sentencepiece
pip install -r experiments/017-gqa-kv2/requirements.txt
python3 experiments/017-gqa-kv2/prepare_caseops_data.py
```

1xH100 dev:

```bash
SEED=1337 \
SKIP_QUANT=1 \
NUM_GPUS=1 \
torchrun --standalone --nproc_per_node=1 \
experiments/017-gqa-kv2/train_gpt.py
```

8xH100 final:

```bash
SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
experiments/017-gqa-kv2/train_gpt.py
```

## Iteration Results x1 H100

| Version | Pre-Quant BPB | Post-Quant BPB | Step Time (ms) | Artifact Size | Commit | Log | Description |
| ------- | ------------- | -------------- | -------------- | ------------- | ------ | --- | ----------- |

## Iteration Results x8 H100

| Version | Pre-Quant BPB | Post-Quant BPB | Step Time (ms) | Artifact Size | Commit | Log | Description |
| ------- | ------------- | -------------- | -------------- | ------------- | ------ | --- | ----------- |
