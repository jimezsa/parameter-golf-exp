# Gemini Research Brief: BitNet/Ternary Parameter Density Maximization

## Context

We're going all-in on BitNet 1.58-bit ternary quantization as our primary strategy to beat SOTA (1.1147 BPB). The hypothesis is pure parameter density: ternary weights compress ~4x better than int6 GPTQ, letting us fit a much larger pure transformer into 16MB. Mamba/hybrid long-context is killed. seq_len stays at 1024.

Rank 1 (1.0149 BPB) already validates this route — they use BitNet + depth recurrence. We need to understand exactly how to replicate and exceed that.

## Existing Work

We already have:
- `experiments/003-bitnet/ternary.py`: TernaryQuantizeSTE, TernaryLinear, base-3 pack/unpack
- `experiments/003-bitnet/benchmark_ternary_latency.py`: latency gate harness (not yet run on H100)
- `experiments/003-bitnet/spec-003-bitnet.md`: initial spec from scout covering QAT recipe, compression math, GPTQ interaction, Muon compatibility
- The spec estimates ~15-20% slowdown from unfused ternary forward pass and suggests a Triton kernel fallback

## What I Need From You

### 1. BitNet Training Recipe: From-Scratch QAT vs Post-Training Quantization

The spec says "late QAT fails for ternary" and we must train with STE from step 1. Verify this claim with evidence:
- What BPB/perplexity gap do papers report between from-scratch BitNet QAT and post-training ternary quantization at <100M params?
- Is there a hybrid approach (train dense for N% of steps, then switch to STE for remaining)? Some papers suggest phased quantization reduces instability.
- At our ~20K training steps budget (10 min on 8xH100), is from-scratch QAT feasible or do we need more steps for ternary to converge?
- What about OneBit (Xu et al. 2024) or other ternary training improvements since the original BitNet paper?

### 2. Rank 1 Architecture Reverse Engineering

Rank 1 (1.0149 BPB) uses BitNet + depth recurrence. I need you to dig into this:
- How does depth recurrence (weight sharing across layers) interact with ternary weights? Our spec says "quantization error is identical across passes" — is this actually beneficial or does it amplify representation collapse?
- What's the optimal ratio: how many unique ternary layer blocks, looped how many times? (e.g., 6 unique blocks × 3 loops = 18 effective layers, but only 6 blocks worth of parameters)
- Does rank 1 use any activation quantization, or just weight ternary?
- Does rank 1 use standard STE or a modified estimator (clipped STE, polynomial STE, etc.)?

### 3. Optimal Architecture for 16MB Ternary Budget

Given ternary compression at ~0.15-0.2 bytes/param after base-3 packing + LZMA:
- What's the maximum parameter count we can fit? The spec says ~80M at 0.2 B/param. Can we push closer to 0.15 B/param with optimized packing?
- Design 2-3 concrete architectures that maximize BPB at seq_len=1024:
  - (A) Deep narrow: e.g., 24L/512d — more layers, more representational depth
  - (B) Wide moderate: e.g., 18L/768d — wider representations, fewer layers
  - (C) Depth recurrence: e.g., 6 unique blocks × 4 loops = 24 effective layers — maximum depth per byte
- For each: total params, estimated compressed size, expected tokens/sec on 8xH100 at seq_len=1024
- Which architecture family does existing evidence favor for small-scale ternary models?

### 4. Training Throughput: Can We Fit Enough Steps?

This is the make-or-break question. The spec estimates 15-20% slowdown from unfused STE.
- On 8xH100 at seq_len=1024, batch_size per current SOTA config: how many training steps can we fit in 10 minutes for each of the 3 architectures above?
- Compare against current SOTA step count at the same wallclock. If ternary gets significantly fewer steps, the parameter density advantage may be eaten by underfitting.
- Is there a fused Triton kernel for ternary forward pass? Check: (a) Microsoft's BitBLAS, (b) T-MAC, (c) any Triton implementations in the wild. What speedup do they claim?
- Alternative: could we train in FP16 for 80% of steps and switch to ternary STE for the final 20%? What's the quality tradeoff?

### 5. Ternary Compression Pipeline

Current SOTA uses GPTQ int6 → LZMA. Ternary needs a different pipeline:
- Confirm: base-3 packing (4 values per byte, 81 states) before LZMA is optimal, or is there a better encoding?
- What about mixed precision: ternary for projections, int6 GPTQ for embeddings/head, FP16 for norms. Estimate the full artifact size breakdown for each architecture option.
- LZMA compression ratio on ternary weights: what ratio should we expect? The spec says 0.15-0.2 B/param — validate this against any published numbers.
- Rank 1 presumably solved this pipeline. Any clues from their submission about packing format?

### 6. Interaction with Existing SOTA Techniques

Our current SOTA stack includes: LeakyReLU², BigramHash 3072×112, XSA-all, Partial RoPE, SmearGate, VE128, EMA+SWA, Parallel Muon + Parameter Banking.
- Which of these are compatible with ternary weights? Specifically:
  - BigramHash: the hash embedding is separate from projections — should be fine, but confirm
  - XSA (cross-sequence attention): any issues with ternary Q/K/V projections?
  - EMA + SWA: averaging ternary weights — do you average the latent FP weights and then re-quantize, or average the ternary values directly?
  - Parameter Banking: this is a Muon-specific technique. Compatible with STE?
- Which techniques should we DROP for the ternary version? (e.g., GPTQ is partially replaced, Late QAT changes)

### 7. Risk Matrix

Rank the following risks by severity and propose mitigations:
1. STE gradient variance causing training instability at ~20K steps
2. Wallclock timeout from unfused ternary ops
3. Ternary representation collapse at 80M params (is there a model size where ternary stops scaling?)
4. LZMA compression ratio worse than expected on real ternary weights
5. Depth recurrence amplifying quantization error

## Deliverables

Write full analysis to `parameter-golf-exp/experiments/003-bitnet/research_ternary_density.md`.

Include:
- Evidence-backed answers to all 7 sections
- Recommended architecture (pick one of A/B/C or propose D)
- Concrete training recipe: LR schedule, warmup, STE variant, optimizer config
- Full compression pipeline spec
- Go/no-go recommendation with clear criteria

If any section lacks published evidence, say so explicitly and propose a concrete experiment to fill the gap.
