# Experiment 009: AR-Diffusion Hybrid (Soft-Masked)

## Paper / Source
- **Titles**:
  1. *A Cheaper and Better Diffusion Language Model with Soft-Masked Noise* (Chen et al., 2023) - [Link](https://arxiv.org/abs/2304.04746)
  2. *Text Generation with Diffusion Language Models: A Pre-training Approach with Continuous Paragraph Denoise* (Lin et al., 2022) - [Link](https://arxiv.org/abs/2212.11685)
  3. *A Reparameterized Discrete Diffusion Model for Text Generation* (Zheng et al., 2023) - [Link](https://arxiv.org/abs/2302.05737)
- **Key idea**: Pure Text Diffusion models suffer from sample inefficiency because they predict a masked subset of tokens at a uniform timestep $t$, unlike Autoregressive (AR) models which receive dense gradients for all $N$ tokens simultaneously. By combining Soft-Masked Diffusion with an AR-Diffusion Hybrid approach, we impart rich, bidirectional structural context probabilistically (e.g., 25% of steps) without sacrificing the exact next-token BPB metric optimization and wallclock constraints.

## Hypothesis
Within a 10-minute training wallclock, pure diffusion will underfit. However, injecting an auxiliary bidirectional diffusion pass (using soft-masked embeddings and time projections) on ~25% of training steps will improve structural representations without doubling the step time, leading to a better final BPB compared to the pure AR baseline.

## Base Code
- **Fork from**: `experiments/008-mtp-lite/train_gpt.py` (MTP-Lite with parameter banking and EMA).

## Changes from Baseline
- **Architecture**:
  - `mask_embed`: A learned embedding for corrupting input tokens.
  - `time_proj`: A lightweight projection of a continuous timestep $t \sim U(0,1)$ added to the residual stream before attention in each block.
- **Dual Head / Dual Pass**:
  - **AR Pass (Primary)**: Standard causal masking (`is_causal=True`) computed on 100% of steps for exact BPB scoring and MTP loss.
  - **Diffusion Pass (Auxiliary)**: Runs probabilistically (~25% of steps). Uses bidirectional attention (`is_causal=False`) on a soft-masked sequence to predict the uncorrupted tokens.
- **Compute Constraint**: Diffusion pass is limited to ~25% execution rate to ensure the training script completes within the 10-minute 8xH100 wallclock limit.

## Run Config
- **GPU**: 1x H100 (dev) / 8x H100 (final)
- **Steps / Duration**: 10 minutes (wallclock)
- **Key hyperparameters changed**:
  - `is_causal` dynamic toggle added to `CausalSelfAttention` and `Block`.
  - Probabilistic auxiliary diffusion step (25% frequency).
  - Diffusion loss weight: 0.3.

## Status
- [x] Proposed by scout
- [x] Approved by professor
- [x] Implemented by engineer
- [ ] Tested by human / autoresearch
- [ ] Analyzed
- [ ] Decision: adopt / discard / iterate
