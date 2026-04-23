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

## RoPE Variant Entrypoints

- `train_gpt_xpos.py`: xPos-style rotary scaling on top of the Exp 014 baseline.
- `train_gpt_layerwise_rope.py`: layerwise RoPE schedule that can ramp `ROPE_DIMS` and optionally `ROPE_BASE` across depth.
- `train_gpt_headwise_rope.py`: headwise RoPE that rotates only a configurable prefix of attention head groups.
- `train_gpt_yarn.py`: YaRN-style rotary rescaling for longer-context probes.

## Regularization Variant Entrypoints

- `train_gpt_dropout.py`: dropout-regularized Exp 014 wrapper with env-backed residual and optional embedding dropout.

## Decoded Variant Entrypoints

- `train_gpt_decode_diffusion.py`: readable decode-script variant with the Exp 014 latent-MSE diffusion aux path added back in; diffusion is capped to one micro-step per optimizer step so grad accumulation does not multiply wallclock cost.

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

- Data prep:

```bash
pip install brotli sentencepiece
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
python3 data/cached_challenge_fineweb.py --variant sp8192
```

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

Manual submission-code packing:

```bash
python3 experiments/014-ar-latent/pack_submission.py \
experiments/014-ar-latent/train_gpt.py \
experiments/014-ar-latent/train_gpt_packed.py \
--minify
```

Use this when you need to pack the current `train_gpt.py` entrypoint without rerunning training. Full GPTQ runs already write a wrapped code artifact to `SUBMISSION_CODE_PATH` (`logs/{RUN_ID}.train_gpt_submission.py` by default); `--minify` is optional and falls back to basic minification if `python-minifier` is unavailable.

Triton loss variant dependency:

```bash
python3 -m pip install triton
```

Use this before running `experiments/014-ar-latent/train_gpt_h100_bf16forward_tritonloss.py`.

## Iteration Results x1 H100

| Version | Val BPB | Post-Quant BPB | Step Time (ms) | Artifact Size | Commit | Description                                            |
| ------- | ------- | -------------- | -------------- | ------------- | ------ | ------------------------------------------------------ |
| v1      | 1.1881  | —              | 166            | —             | —      | Baseline: 1. SWIGELU=12, ROPE_DIMS=32, NUM_KV_HEADS=8. |

## Iteration Results x8 H100

| Version | Val BPB | Post-Quant BPB | Step Time (ms) | Artifact Size | Commit | Description                                            |
| ------- | ------- | -------------- | -------------- | ------------- | ------ | ------------------------------------------------------ |
| v1      | 1.0957  | 1.09482789     | 166            | 17164353      | —      | Baseline: 1. SWIGELU=12, ROPE_DIMS=32, NUM_KV_HEADS=8. |

### Human Config Sweep (v14–v40, Apr 19–22)

Runs v14–v40 are manual config sweeps run by jimezsa on the pod. Baseline config: 11L/512d, KV4, rope16, SWA=on, SwiGeLU=off, 36.2M params. Post-quant metric is sliding-window BPB unless noted. Artifact size constraint: ≤16,777,216 bytes.

| Version | Val BPB | Post-Quant BPB            | Step Time (ms) | Artifact Size  | Commit | Description                                                                                              |
| ------- | ------- | ------------------------- | -------------- | -------------- | ------ | -------------------------------------------------------------------------------------------------------- |
| v14     | 1.1955  | —                         | 844            | —              | —      | gptg script baseline. KV4, rope16, SWA=on. 711 steps.                                                    |
| v15     | 1.1888  | 1.1811 (sw) / 1.1971 (rt) | 752            | 16,098,918     | —      | gptg script v2. KV4, rope16, SWA=on. 782 steps.                                                          |
| v16     | 1.1866  | 1.1787 (sw) / 1.1948 (rt) | 794            | **17,181,683** | —      | gptg + KV8 + rope32 + SwiGeLU. 38.7M params. 741 steps. **Over 16MB.**                                   |
| v17     | FAIL    | —                         | 767            | —              | —      | Baseline w/ quant enabled. Only 10 steps completed, killed.                                              |
| v18     | 1.1889  | 1.1812 (sw) / 1.1971 (rt) | 755            | 16,098,557     | —      | Baseline full GPTQ. KV4, rope16, SWA=on. 780 steps. Gate 2 reference.                                    |
| v19     | 1.1906  | 1.1828 (sw) / 1.1989 (rt) | 766            | 15,955,469     | —      | SwiGeLU only. KV4, rope16. 35.8M params. 768 steps. Fits 16MB.                                           |
| v20     | 1.1866  | 1.1787 (sw) / 1.1947 (rt) | 784            | **17,321,252** | —      | KV8 + rope32. 39.1M params. 751 steps. **Over 16MB.**                                                    |
| v21     | 1.1861  | —                         | 781            | —              | —      | KV8 + rope32 + SWA=on. Pre-quant only. 39.1M params. 753 steps.                                          |
| v22     | 1.1867  | 1.1976 (sw) / 1.2138 (rt) | 783            | 15,998,618     | —      | Labeled "no_swa" but cfg shows SWA=on. Quant gap +0.0109. 751 steps.                                     |
| v23     | 1.2227  | 1.2142 (sw) / 1.2302 (rt) | 1261           | **17,182,829** | —      | KV8 + rope32 + SwiGeLU + recurrence. 38.7M params. 467 steps. Slow + **over 16MB.**                      |
| v24     | 1.3377  | —                         | 2471           | **17,317,732** | —      | KV8 + rope32, SWA=off. **Undertrained** (238 steps, very slow). Invalid.                                 |
| v25     | 1.1882  | —                         | 815            | —              | —      | KV8 + rope32, SWA=off. Pre-quant only. 39.1M params. 737 steps.                                          |
| v26     | 1.1860  | —                         | 798            | —              | —      | KV8 + rope32, SWA=off. Pre-quant only. 39.1M params. 753 steps.                                          |
| v27     | 1.1858  | —                         | 792            | —              | —      | KV8 + rope32, SWA=on. Pre-quant only. 39.1M params. Best KV8 pre-quant. 758 steps.                       |
| v28     | 1.1862  | —                         | 808            | —              | —      | KV8 + rope32 + SwiGeLU, SWA=on. Pre-quant only. 38.7M params. 744 steps.                                 |
| v29     | 1.1881  | —                         | 765            | —              | —      | Headwise RoPE variant, SWA=off. Pre-quant only. 785 steps.                                               |
| v30     | 1.1920  | —                         | 777            | —              | —      | Layerwise RoPE variant, SWA=off. Pre-quant only. 773 steps.                                              |
| v31     | FAIL    | —                         | —              | —              | —      | xPoS rope64 sb512 full. Crashed/incomplete.                                                              |
| v32     | 1.1899  | —                         | 760            | —              | —      | xPoS + rope64, SWA=off. Pre-quant only. 791 steps.                                                       |
| v33     | 1.1899  | —                         | 761            | —              | —      | xPoS + rope64 + sb1024, SWA=off. Pre-quant only. 789 steps.                                              |
| v34     | 1.1879  | —                         | 759            | —              | —      | xPoS + rope16, SWA=off. Pre-quant only. 791 steps.                                                       |
| v35     | 1.1871  | —                         | 755            | —              | —      | YaRN + rope16, SWA=off. Pre-quant only. 796 steps.                                                       |
| v36     | 1.1898  | —                         | 772            | —              | —      | Layerwise rope 32→64 cosine, SWA=off. Pre-quant only. 778 steps.                                         |
| v37     | 1.1893  | —                         | 773            | —              | —      | Layerwise rope 32→64 cosine + rope4, SWA=off. Pre-quant only. 777 steps.                                 |
| v38     | 1.2042  | —                         | 786            | —              | —      | Dropout=0.10, SWA=off. Pre-quant only. 764 steps. Hurts pre-quant.                                       |
| v39     | 1.2025  | 1.2079 (rt only)          | 790            | 16,098,169     | —      | Dropout=0.05, SWA=off. Pre-quant only + partial quant. 745 steps.                                        |
| v40     | 1.3142  | —                         | 2181           | —              | —      | No diffusion + recurrence. Only 276 steps at 2181ms. Massive regression.                                 |
| v41     | 1.1997  | 1.2119 (sw) / 1.2282 (rt) | 941            | 16,142,512     | —      | KV8 + rope32 + SwiGeLU + embed_bits=6. **Fits 16MB.** 626 steps. SWA=off.                                |
| v42     | 1.3267  | 1.3172 (sw) / 1.3325 (rt) | 2412           | **17,093,525** | —      | KV8 + rope32 + SwiGeLU + embed_bits=8 + full GPTQ. **Undertrained** (244 steps). **Over 16MB.** SWA=off. |
| v43     | 2.4710  | 2.4773 (rt) / 2.4859 (sw) | ~2329          | **16,071,105** | —      | `train_gpt_encode_diffusion.py` full GPTQ. Layer looping (num_loops=2). **EMA broken** (pre-EMA 1.3886 → post-EMA 2.4710). 253 steps. Over 16MB. |
| v44     | 2.4101  | 2.4156 (rt) / 2.4206 (sw) | ~2329          | 16,017,925     | —      | `train_gpt_encode_reference.py` full GPTQ. Layer looping + parallel residuals + QK-gain 5.0. **EMA broken** (pre-EMA 1.3710 → post-EMA 2.4101). 253 steps. |

### Reference Runs (exp 013)

| Run ID                    | Val BPB | Steps | Step Time (ms) | Description                                                 |
| ------------------------- | ------- | ----- | -------------- | ----------------------------------------------------------- |
| exp013_baseline           | 1.1872  | 794   | 757            | Exp 013 baseline reproduction on new pod                    |
| exp013_baseline (run 2)   | 1.1886  | 783   | 767            | Second baseline run, slightly worse                         |
| exp013_swigelu_rope32_KV8 | 1.1850  | 756   | 794            | SwiGeLU + KV8 + rope32 in exp 013. Best pre-quant in sweep. |

### Key Findings from Config Sweep

1. **KV8 + rope32 gives best pre-quant** (v27: 1.1858) but **all KV8 configs exceed 16MB** when quantized (17.2–17.3MB). Need parameter-budget reduction to fit.
2. **Baseline full GPTQ works** (v18): pre=1.1889, sw=1.1812, artifact=16.1MB. Quant gap=+0.0077 (rt) or -0.0077 better with sliding window vs pre-quant.
3. **gptg_base_v2** (v15) is the best post-quant under 16MB: sw=1.1811, beating baseline v18 by 0.0001.
4. **SwiGeLU alone** (v19) hurts pre-quant (1.1906 vs 1.1889) but fits 16MB. Not worth it standalone.
5. **Dropout hurts** (v38/v39): +0.015–0.017 regression.
6. **No diffusion** (v40): catastrophic regression, 1.3142. Confirms diffusion is load-bearing.
7. **Recurrence** (v23): slow (1261ms) and poor (1.2227). Confirms kill decision.
8. **xPoS/YaRN/layerwise-rope** (v32–v37): marginal or neutral vs baseline. Not worth pursuing.
9. **Best overall post-quant**: v16/v20 at 1.1787 (sw), but over 16MB. Need ~1.3MB artifact reduction.

## Notes

- The earlier `DIFFUSION_STOP_FRAC=0.60` recipe was the best validated Exp 013 training baseline. Exp 014 intentionally shortens that default to `0.40` because this lane is targeting quantization stability, not just pre-quant BPB.
- The first real gate for Exp 014 is straightforward: fresh 1x pre-quant BPB plus one full post-quant run to measure whether AR self-gen calibration actually closes the gap enough to justify the new lane.
