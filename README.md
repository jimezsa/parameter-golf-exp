# parameter-golf-exp

This repo contains a minimal Parameter Golf baseline experiment setup. The main training entrypoint in this repo is `baseline/train_gpt.py`.

> Note: The original baseline reference for this repo is in [baseline/README.md](baseline/README.md).

## Run a Full Experiment on a GPU Server

These steps are adapted from `baseline/README.md`, but updated for this repo layout.

### 1. Prepare the machine

Use a CUDA machine with at least 1 NVIDIA GPU. Run all commands below from the repo root.

These instructions assume you are already running on the official Parameter Golf image, so the required Python and CUDA dependencies are preinstalled.

### 2. Download the baseline dataset and tokenizer

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

This downloads:

- `data/datasets/fineweb10B_sp1024/`
- `data/tokenizers/fineweb_1024_bpe.model`

### 3. Launch a 1-GPU baseline run

```bash
RUN_ID=baseline_sp1024_1gpu \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 baseline/train_gpt.py
```

By default, the baseline keeps the original ~10 minute wallclock cap. During training it prints `train_loss`, and at validation/export time it prints `val_loss`, `val_bpb`, and the compressed artifact size.

### 4. Launch a multi-GPU run

`baseline/train_gpt.py` expects `WORLD_SIZE` to divide `8`, so valid values are `1`, `2`, `4`, or `8`.

Example on 8 GPUs:

```bash
RUN_ID=baseline_sp1024_8gpu \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 baseline/train_gpt.py
```

````

For periodic validation during training, set for example:

```bash
VAL_LOSS_EVERY=200
````

### 5. Outputs from a full run

The training script writes artifacts in the current working directory:

- `logs/<RUN_ID>.txt`
- `final_model.pt`
- `final_model.int8.ptz`

The final score to watch is printed in the `final_int8_zlib_roundtrip` lines near the end of the log. That is the post-quantization validation result for the compressed artifact.
