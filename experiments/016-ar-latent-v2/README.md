# Experiment 016: AR Latent v2

## Hypothesis

Clean continuation of the AR-latent lane (exp 014 codebase). Fresh iteration table, no legacy sweep baggage.

## Changes from Baseline

Forked from exp 014 `train_gpt.py` as-is. No code changes in v1.

`train_gpt_decode_diffusion.py` now defaults to `DIFFUSION_SEQ_LEN=512`, cropping only the stochastic latent-MSE auxiliary diffusion pass before its transformer forward. Standard validation is run through the eager base model to avoid the compiled-validation recompile failure observed in v2/v3.

Manual submission-code packing:

```bash
python3 experiments/016-ar-latent-v2/pack_submission.py \
experiments/016-ar-latent-v2/train_gpt.py \
experiments/016-ar-latent-v2/train_gpt_packed.py \
--minify
```

## Run Commands

- Data prep:

```bash
pip install brotli sentencepiece
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
python3 data/cached_challenge_fineweb.py --variant sp8192
```

1xH100 baseline screen:

```bash
RUN_ID=exp016_1x_baseline \
SEED=1337 \
SKIP_QUANT=1 \
torchrun --standalone --nproc_per_node=1 \
experiments/016-ar-latent-v2/train_gpt.py
```

8xH100 baseline screen:

```bash
RUN_ID=exp016_8x_baseline \
SEED=1337 \
SKIP_QUANT=1 \
torchrun --standalone --nproc_per_node=8 \
experiments/016-ar-latent-v2/train_gpt.py
```

Full GPTQ baseline:

```bash
RUN_ID=exp016_gptq_full \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 \
experiments/016-ar-latent-v2/train_gpt.py
```

## Run Config

- GPU: 1x H100 (dev) / 8x H100 (final)
- Tokenizer: SP8192
- Arch: 11L/512d, latent-MSE diffusion, MuonEq-R

## Iteration Results x1 H100

| Version | Val BPB | Post-Quant BPB | Step Time (ms) | Artifact Size | Commit | Log | Description |
| ------- | ------- | -------------- | -------------- | ------------- | ------ | --- | ----------- |

## Iteration Results x8 H100

| Version | Val BPB | Post-Quant BPB | Step Time (ms) | Artifact Size | Commit | Log | Description |
| ------- | ------- | -------------- | -------------- | ------------- | ------ | --- | ----------- |
| v2 | FAIL | — | — | — | 5070904 | — | `train_gpt_encode_diffusion.py` x8 run crashed during pre-quant validation in compiled `forward` after the diffusion crop wiring change; stack reached `eval_val -> _eval_loss_static_or_eager -> forward`. |
| v3 | FAIL | — | — | — | 4f35c16 | — | `train_gpt_encode_diffusion.py` x8 run still crashed during pre-quant validation; `train_model` was still routing `eval_val` through the DDP/compiled model and hit `FailOnRecompileLimitHit`. |
