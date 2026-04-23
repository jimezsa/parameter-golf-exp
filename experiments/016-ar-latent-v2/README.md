# Experiment 016: AR Latent v2

## Hypothesis

Clean continuation of the AR-latent lane (exp 014 codebase). Fresh iteration table, no legacy sweep baggage.

## Changes from Baseline

Forked from exp 014 `train_gpt.py` as-is. No code changes in v1.

## Run Config

- GPU: 1x H100 (dev) / 8x H100 (final)
- Tokenizer: SP8192
- Arch: 11L/512d, latent-MSE diffusion, MuonEq-R

## Iteration Results x1 H100

| Version | Val BPB | Post-Quant BPB | Step Time (ms) | Artifact Size | Commit | Log | Description |
|---------|---------|----------------|-----------------|---------------|--------|-----|-------------|

## Iteration Results x8 H100

| Version | Val BPB | Post-Quant BPB | Step Time (ms) | Artifact Size | Commit | Log | Description |
|---------|---------|----------------|-----------------|---------------|--------|-----|-------------|
