# Experiment 014: AR Latent Baseline

## Purpose

Exp 014 keeps the `ar-latent` naming while pulling the new quantization lane out of Exp 013 into a clean baseline. Training keeps the proven loader-prefetch harness, but the quantization and diffusion-stability path changes:

- GPTQ calibration uses self-generated AR trajectories instead of clean teacher-forced batches
- GPTQ returns to clip-sigma scaling
- diffusion-specific parameters get stricter decay than the generic scalar/head groups
- the default diffusion cooldown ends earlier so the model gets a longer pure-AR tail before quantization

Exp 013 remains the historical scratchpad for throughput probes and HP sweeps. Exp 014 is the lean baseline lane.

## Baseline File

- `train_gpt.py` is the single baseline entrypoint for this experiment.

## Locked Baseline Defaults

Shared defaults:

- 11L / 512d, SP8192 tokenizer, latent-MSE diffusion, MuonEq-R, brotli + byte-shuffle
- `SCALAR_LR=0.02`
- `HEAD_LR=0.008`
- `TIED_EMBED_LR=0.03`
- `DIFFUSION_AUX_PROB=0.03`
- `DIFFUSION_START_FRAC=0.25`
- `DIFFUSION_STOP_FRAC=0.60`
- `WARMDOWN_FRAC=0.667`
- `WARMUP_STEPS=20`
- `MUON_WD=0.090`
- `LATE_QAT_THRESHOLD=0.15`
- `SWA_ENABLED=1`
- `GPTQ_CALIB_BATCHES=64`
- `SELFGEN_PROMPT_LEN=32`
- `SELFGEN_SEQ_LEN=256`
- `SELFGEN_BATCH_SIZE=8`
- `SELFGEN_TEMPERATURE=0.8`
- `DIFFUSION_SCALAR_WD=0.08`
- `DIFFUSION_HEAD_WD=0.08`
- `MATRIX_CLIP_SIGMAS=12.85`
- `EMBED_CLIP_SIGMAS=20.0`

Scale-aware defaults inside `train_gpt.py`:

- 1xH100 baseline: `MATRIX_LR=0.045`, `MIN_LR=0.05`
- 8xH100 baseline: `MATRIX_LR=0.020`, `MIN_LR=0.00`

If you need to override either profile, pass env vars explicitly. Otherwise the file auto-selects the 8x matrix/min-LR pair when `WORLD_SIZE=8`.

## Run Commands

1xH100 baseline screen:

```bash
RUN_ID=exp014_1x_baseline \
SEED=1337 \
SKIP_QUANT=1 \
torchrun --standalone --nproc_per_node=1 \
experiments/014-ar-latent/train_gpt.py
```

8xH100 baseline screen:

```bash
RUN_ID=exp014_8x_baseline \
SEED=1337 \
SKIP_QUANT=1 \
torchrun --standalone --nproc_per_node=8 \
experiments/014-ar-latent/train_gpt.py
```

Full GPTQ baseline:

```bash
RUN_ID=exp014_gptq_full \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 \
experiments/014-ar-latent/train_gpt.py
```

Triton loss variant dependency:

```bash
python3 -m pip install triton
```

Use this before running `experiments/014-ar-latent/train_gpt_h100_bf16forward_tritonloss.py`.

## Iteration Results

| Version | Val BPB | Post-Quant BPB | Step Time (ms) | Artifact Size | Commit | Description |
|---------|---------|----------------|-----------------|---------------|--------|-------------|
| v1 | 1.1881 | — | 166 | — | — | Baseline: DIFFUSION_STOP_FRAC=0.40, all other HPs from exp 013 v21 |
| v2 | 1.1870 | — | 166 | — | — | DIFFUSION_STOP_FRAC=0.50 |
| v3 | 1.1867 | — | 166 | — | — | DIFFUSION_STOP_FRAC=0.60, reproduces exp 013 v21 (1.1868). Gate 1 passed. |
| v4 | 1.2798 | 1.2825 (sw) / 1.2980 (rt) | 1566 | 15,998,434 | — | Gate 2: full GPTQ w/ AR self-gen. **Pod bottleneck**: only 355 steps (vs ~3600 expected). Gap=+0.0027 sw, but undertrained — not directly comparable. Hessians 16.8s, AR calib 123.7s, quant 46.0s. |
| v5 | 1.1827 | — | 170 | — | a2e3c4d | SwiGeLU activation (SWIGELU=1), DIFFUSION_STOP_FRAC=0.60. Beats v3 by 0.0040. |
| v6 | 1.1817 | — | 170 | — | da9f1a0 | Wider diffusion window: DIFFUSION_START_FRAC=0.15, STOP=0.60. Beats v5 by 0.0010. |
| v7 | 1.1918 | — | 796 | — | — | **Undertrained** (755/~3600 steps). Throttled pod (345 MHz GPU clock). Invalid — do not compare. |
| v8 | FAIL | — | FAIL | — | 41dbbd6 | `train_gpt_h100.py` initial 1xH100 auto-accum variant OOM before step 0. Single-rank auto chose `GRAD_ACCUM_STEPS=1` while keeping `TRAIN_BATCH_TOKENS=786432`, triggering a 3.00 GiB compile-time allocation and exhausting 80 GB H100 memory. |
| v9 | FAIL | — | FAIL | — | 2a08b4c | `train_gpt_h100.py` second 1xH100 auto-batch attempt still OOMed before step 0. Token-cap auto chose `local_batch_tokens=196608` / `micro_batch_seqs=96`, and Inductor failed allocating a `(96, 2048, 2048)` BF16 buffer (~768 MiB). |
| v10 | — | — | 1048 | — | 68ce02e | Throughput-only screen of `train_gpt_h100_bf16forward_tritonloss.py`. The Triton loss path ran, but step time regressed versus the BF16-forward baseline (`1048.22 ms` vs `737.33 ms`). |
| v11 | — | — | 938 | — | de08646 | Throughput-only screen after replacing the chunked Python-loop backward with a Triton `dlogits` kernel plus GEMM gradients. Step time improved to `937.59 ms`, but is still slower than the BF16-forward baseline (`737.33 ms`). |
| v12 | FAIL | — | FAIL | — | a990e3b | `train_gpt_h100_bf16forward_tritonloss.py` no-`grad_logits` Triton backward failed during `torch.compile` with `BackendCompilerFailed`: `a leaf Variable that requires grad is being used in an in-place operation`. |

## Notes

- The earlier `DIFFUSION_STOP_FRAC=0.60` recipe was the best validated Exp 013 training baseline. Exp 014 intentionally shortens that default to `0.40` because this lane is targeting quantization stability, not just pre-quant BPB.
- The first real gate for Exp 014 is straightforward: fresh 1x pre-quant BPB plus one full post-quant run to measure whether AR self-gen calibration actually closes the gap enough to justify the new lane.
