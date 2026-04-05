# Experiment 003 — BitNet b1.58 Ternary Quantization Spec

## 1. Paper Grounding
**Core Concept (Hu et al. "The Era of 1-bit LLMs"):** BitNet b1.58 constrains weights to a ternary alphabet `{-1, 0, 1}`. 
- **Quantization Function:** The latent continuous weights $W$ are quantized to $W_q$ using an absolute-mean scaling factor: 
  $\gamma = \frac{1}{n} \sum |W|$
  $W_q = \text{RoundClip}\left(\frac{W}{\gamma + \epsilon}, -1, 1\right)$
- **Gradients:** Backpropagation routes around the non-differentiable `RoundClip` via the Straight-Through Estimator (STE), passing gradients directly to the continuous $W$. 
- **Activations:** Typically 8-bit, but for Parameter Golf where artifact size (weights) is the primary constraint and compute is bounded, keeping activations at native BF16/FP16 while ternarizing weights maximizes implementation speed without penalizing the 16MB budget.

## 2. Architecture Changes
- **Ternary Targets:** Apply ternary quantization exclusively to the heavy $O(N^2)$ components:
  - Attention: `Q`, `K`, `V`, and `Out` projections.
  - MLP: `Up`, `Down`, and `Gate` projections.
- **Full Precision Exceptions:** Keep the Embedding layer, LayerNorms, biases, and the final classification head in FP32/BF16 (or GPTQ int6). These components hold critical representational manifold topology and cost negligible artifact space.
- **Optimizer Interaction (Muon):** Muon uses orthogonalization (Newton-Schulz) on 2D weight gradients. 
  - *Compatibility:* Muon is fully compatible. It calculates the orthogonalized update using $\nabla_{W_q} L$ (via STE) and applies this update to the latent, continuous $W$. 
  - *Hybrid Split:* Continue to use Muon for the 2D ternary projections (updating their latent FP representations) and AdamW for 1D tensors (Norms, biases, scales).
- **Depth Recurrence Interaction (Exp 002):** When a ternary layer block is looped, quantization error is mathematically identical across passes (no new noise injected per loop). The continuous latent weights simply aggregate the STE gradients from multiple BPTT steps. This actually *stabilizes* the STE by providing denser gradient signals to the latent weights.

## 3. Compression Impact Math
- **Current Baseline:** int6 = 6 bits/param. LZMA packs this, but entropy limits actual compression.
- **Ternary Entropy:** The alphabet `{-1, 0, 1}` natively carries $\log_2(3) \approx 1.58$ bits of entropy.
- **LZMA Synergy:** Ternary weights exhibit massive structural sparsity (lots of 0s) and repetitiveness. By byte-packing them (e.g., $3^4 = 81$ states per byte, cleanly mapping 4 weights into 1 byte) before LZMA, LZMA can compress them down to $\sim 0.15 - 0.2$ bytes per parameter.
- **Scaling Headroom:** 16MB / 0.2 bytes $\approx 80M$ parameters. 
  - Our 11L / 512d baseline is heavily under-budget now. 
  - We can safely scale the dense trunk up to **18L / 768d** or even **24L / 512d**, matching or exceeding the capacity of Rank 2 while staying under 16MB.

## 4. Training Recipe Specifics
- **Training Paradigm:** **Quantization-Aware Training (QAT) from Scratch.**
  - *Late QAT fails for ternary.* Compressing a fully trained continuous model to 1.58 bits causes catastrophic representation collapse. BitNet *must* be trained with STE from Step 1 so the network learns to route logic through the discrete pathways.
- **Learning Rate Schedule:** The continuous latent weights need a relatively higher learning rate because the `RoundClip` function naturally acts as a strong regularizer. Increase peak LR by ~1.5x over the baseline schedule, and use a longer warmup (e.g., 10% of total steps) to let the latent weights distribute around the quantization thresholds.
- **Wallclock Impact:** Simulating ternary weights during training (FP latent $\rightarrow$ AbsMean $\rightarrow$ RoundClip $\rightarrow$ BF16 cast for matmul) adds a non-trivial memory-bandwidth overhead. Unfused, expect a ~15-20% slowdown on the 10-minute H100 budget. 

## 5. GPTQ Interaction
- **Ternary Layers:** GPTQ is completely obsoleted for the `{-1, 0, 1}` layers. The BitNet latent weights simply get their final `RoundClip` frozen into the artifact.
- **Non-Ternary Layers:** Retain the current Late-QAT/GPTQ pipeline exclusively for the Embedding and Head layers (quantizing them to int6 or int8) to squeeze the last few bytes, though skipping GPTQ on these entirely may be viable if the ternary layers save enough space.

## 6. Risk Assessment
- **Instability at 20k steps:** BitNet typically requires longer training horizons to overcome initial STE gradient variance. At our short ~10-minute training budget, the ternary network might underfit compared to a dense baseline.
- **Wallclock Timeout:** If the PyTorch-native implementation of `W / W.abs().mean()` per-forward pass is too slow, we will breach the 10-minute limit.
  - *Fallback:* We may need a fused `TernaryLinear` Triton kernel just for the forward pass, or we back off to 2-bit or 3-bit GPTQ if wallclock strictly fails.