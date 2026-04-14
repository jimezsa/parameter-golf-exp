# Experiment 013: Levenberg-Marquardt Optimizer

## Motivation

Levenberg-Marquardt (LM) interpolates between Gauss-Newton (fast convergence near optima) and gradient descent (robust far from optima) via an adaptive damping parameter lambda. Classical LM is infeasible for full-scale transformers, but our model is small enough (~20M params, 11L/512d) that approximate second-order methods become tractable within the 10-min wallclock.

The hypothesis: better per-step convergence from curvature-aware updates can offset the higher per-step cost, yielding lower BPB in the same wallclock budget.

## Literature Foundation

1. **Classical Levenberg-Marquardt**: Damped least-squares — update rule `(J^T J + lambda*I)^{-1} J^T r`. Lambda adapts: decreases when loss improves (more Newton-like), increases when it doesn't (more GD-like).

2. **K-FAC (Kronecker-Factored Approximate Curvature)**: Practical second-order for neural networks. Approximates Fisher as Kronecker product of input/gradient covariances per layer. Compatible with mini-batch training. LM-style damping applied per block.

3. **Shampoo / SOAP**: Preconditioned methods that maintain per-layer second-order statistics. Shown to converge faster than Adam on transformers at small-to-medium scale.

4. **AdaHessian**: Diagonal Hessian approximation via Hutchinson trace estimation. Lightweight, drop-in replacement for Adam.

## Approach Candidates

### A: K-FAC + LM Damping (recommended first try)
- Kronecker-factored curvature for attention and MLP blocks
- LM-style adaptive damping per layer
- Amortize curvature updates every N steps (e.g., every 10 steps)
- Expected overhead: ~30-50% per step, but fewer steps to converge

### B: Diagonal LM (lightweight)
- Diagonal Hessian via Hutchinson estimator
- LM damping on diagonal
- Minimal overhead (~15%), modest convergence improvement

### C: Hybrid LM + Muon
- Keep Muon for the bulk bank params (already well-tuned)
- Apply LM-style optimizer to embedding, head, and scalar params
- Lowest risk, but limited upside

## Baseline (inherited from Exp 012)

- SP8192 tokenizer, 11L/512d
- WD=0.090, brotli-11 + byte-shuffle
- SDClip, GPTQ int8 embeds, MuonEq-R
- QK-Gain 5.0
- Latent MSE diffusion (AUX_PROB=0.05, STOP_FRAC=0.60)
- LATE_QAT_THRESHOLD=0.15, GPTQ_CALIB_BATCHES=32
- Best post-quant sw BPB: 1.2036 (1xH100)

## Run Config
- **GPU**: 1x H100 (dev) / 8x H100 (final)
- **Steps / Duration**: 10 minutes wallclock
- Data prep:
```bash
python3 data/cached_challenge_fineweb.py --variant sp8192
```
- Dependencies:
```bash
pip install brotli
```
- Run (1x H100 dev):
```bash
RUN_ID=exp013_levenberg_marquardt \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 experiments/013-levenberg-marquardt/train_gpt.py
```
- Run (8x H100 final):
```bash
RUN_ID=exp013_levenberg_marquardt \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 experiments/013-levenberg-marquardt/train_gpt.py
```

## Iteration Results

| Version | Val BPB | Post-Quant BPB | Step Time (ms) | Artifact (bytes) | Commit | Description |
|---------|---------|----------------|----------------|-------------------|--------|-------------|
| (pending) | | | | | | |

## Status
- [x] Forked from exp 012
- [ ] Select LM variant (K-FAC, diagonal, or hybrid)
- [ ] Implement optimizer
- [ ] v1 baseline comparison run
- [ ] Iterate toward BPP improvement
- [ ] Decision: adopt / discard / iterate
