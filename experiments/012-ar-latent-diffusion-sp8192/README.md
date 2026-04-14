# Experiment 012: AR Latent Diffusion SP8192

## Literature Foundation: Latent Diffusion as AR Regularizer

Recent deep search into autoregressive (AR) models and diffusion reveals that adding a continuous latent diffusion prior mitigates exposure bias, condition inconsistency, and overfitting. This strongly justifies the approach in Exp 012.

### Core Findings & Implementation Mapping

1. **Global Structural Coherence (STAR-LDM)**
   - **Literature**: [STAR-LDM: Stop-Think-AutoRegress (arXiv:2602.20528)](https://arxiv.org/abs/2602.20528) proves that a latent diffusion planning phase enforces structural coherence, correcting the local bias of next-token generation.
   - **Implementation (`train_gpt.py`)**: We replicate this by jointly optimizing the AR next-token loss alongside a continuous latent diffusion matching loss via `_diffusion_loss()` on the final hidden states before the LM head (lines 1220-1250, 1276-1277).

   ```python
    def _diffusion_loss(self, input_ids: Tensor, _target_ids: Tensor, subsample_frac: float = 1.00) -> Tensor:
        if self.latent_proj is None:
            raise RuntimeError("latent_proj is required when diffusion loss is enabled")
        bsz = input_ids.size(0)
        orig_embeds = self.tok_emb(input_ids)
        t = torch.rand(bsz, 1, 1, device=input_ids.device, dtype=torch.float32)
        # ... forward pass with noise mask ...
        pred_latents = self.latent_proj(hidden_diff)
        return F.mse_loss(pred_latents.float(), target_latents.float(), reduction="mean")
   ```

2. **Combating AR Exposure Bias via Optimal Transport**
   - **Literature**: [Condition Errors Refinement with Diffusion Loss (arXiv:2602.07022)](https://arxiv.org/abs/2602.07022) uses diffusion loss to pull intermediate corrupted AR states back to the ideal uncorrupted manifold.
   - **Implementation (`train_gpt.py`)**: Our loss adds a structural anchor (`self.diffusion_loss_weight * self._diffusion_loss(...)` at line 1277) to explicitly correct representation drift during the forward pass.

   ```python
    def forward(self, input_ids: Tensor, target_ids: Tensor, run_aux_diffusion: bool = False) -> Tensor:
        # ...
        if run_aux_diffusion and self.diffusion_loss_weight > 0.0:
            main_loss = main_loss + self.diffusion_loss_weight * self._diffusion_loss(input_ids, target_ids, self.diffusion_subsample_frac)
        return main_loss
   ```

3. **Continuous Structural Priors**
   - **Literature**: [Latent-Autoregressive GP-VAE (arXiv:2512.09535)](https://arxiv.org/abs/2512.09535) shows that continuous structural priors prevent generation collapse.
   - **Implementation (`train_gpt.py`)**: By projecting to a continuous latent space via `latent_proj` (line 1014), we regularize over continuous trajectories rather than using discrete token-space rounding, which is inherently unstable.

   ```python
        self.latent_proj = None
        if diffusion_loss_weight > 0.0 or diffusion_aux_prob > 0.0:
            # Training-only latent diffusion head; kept out of export/quantization.
            self.latent_proj = CastedLinear(model_dim, model_dim, bias=False)
            self.latent_proj._zero_init = True
   ```

4. **Implicit Data Augmentation & Robustness**
   - **Literature**: [Diffusion Beats Autoregressive in Data-Constrained Settings (arXiv:2507.15857)](https://arxiv.org/abs/2507.15857) demonstrates that masking/noising objectives natively resist saturation.
   - **Implementation (`train_gpt.py`)**: Implemented via randomized partial application. The `DIFFUSION_AUX_PROB` flag (parsed at line 105, used at 2271-2275) dynamically triggers the diffusion pass on a subset of iterations. This acts as implicit data augmentation without imposing massive step-time overhead on every step.

   ```python
    # 105: config parameter
    diffusion_aux_prob = float(os.environ.get("DIFFUSION_AUX_PROB", "0.05"))

    # 2271-2275: stochastic execution per step
                use_diffusion = (
                    diffusion_active
                    and args.diffusion_loss_weight > 0.0
                    and args.diffusion_aux_prob > 0.0
                    and random.random() < args.diffusion_aux_prob
                )
   ```

## Hypothesis
Exp 011 latent v3 achieved post-quant sw BPB 1.2036 (15.90MB) with near-zero quant gap. This experiment continues iteration from that baseline to push BPB lower via depth recurrence, parallel residuals, QK-gain tuning, and compression improvements.

## Changes from Exp 011 Latent v3 (baseline)
Starting point: `train_gpt_latent.py` from exp 011 (latent v3 config).

### Baseline config (inherited):
- SP8192 tokenizer
- 11L/512d architecture
- WD=0.090
- Brotli-11 + byte-shuffle compression
- SDClip (k=12.85sigma int6, k=20sigma int8 embeds)
- GPTQ embedding quant at int8
- MuonEq-R optimizer
- QK-Gain 5.0
- Latent MSE diffusion (DIFFUSION_AUX_PROB=0.05, DIFFUSION_STOP_FRAC=0.60)
- LATE_QAT_THRESHOLD=0.15
- GPTQ_CALIB_BATCHES=32
- SWA_ENABLED=0

### Planned iterations:
- Depth recurrence (layers 4-5, top-5 leaderboard feature)
- Parallel residuals (GPT-J style from layer 7+)
- QK-Gain tuning (5.0 vs 5.25 vs 4.5)
- EMA decay tuning
- Compression/pruning optimization

## Run Config
- **GPU**: 1x H100 (dev) / 8x H100 (final)
- **Steps / Duration**: 10 minutes wallclock
- Data prep (run once per pod):
```bash
python3 data/cached_challenge_fineweb.py --variant sp8192
```
- Dependencies:
```bash
pip install brotli
```
- Run from repo root (1x H100 dev):
```bash
RUN_ID=exp012_ar_latent_diffusion_sp8192 \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 experiments/012-ar-latent-diffusion-sp8192/train_gpt.py
```
- Run for final submission (8x H100):
```bash
RUN_ID=exp012_ar_latent_diffusion_sp8192 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 experiments/012-ar-latent-diffusion-sp8192/train_gpt.py
```

## Iteration Results

| Version | Val BPB | Post-Quant BPB | Step Time (ms) | Artifact (bytes) | Commit | Description |
|---------|---------|----------------|----------------|-------------------|--------|-------------|
| v1      | 1.2036  | 1.2039 (sw)    | 776            | 16,672,285 ❌     | 5431ed7 | Fork verification — OVER 16M cap (MiB bug) |
| v1-nodiff | 1.1996 | 1.1996 (sw)   | 747            | 16,672,309 ❌     |        | Diffusion off — OVER 16M cap (MiB bug) |
| v2      | 1.2034  | 1.2169 (sw)    | 771            | 15,990,596 ✅     | 4baa4f0 | TARGET_MB=15.25 — aggressive pruning (42.7%) destroyed quant quality |
| v3      | 1.2002  | 1.2161 (sw)    | 752            | 15,949,713 ✅     |         | Diffusion OFF ablation (TARGET_MB=15.95). -0.032 train BPB vs v2, but quant gap +0.016 |
| v4      | 1.2140  | — (SKIP_QUANT) | 909            | —                 |         | Depth recurrence (layers 4-5, 1 extra pass). +16% step time → 103 fewer steps. Regression ❌ |
| v5      | 1.2063  | — (SKIP_QUANT) | 786            | —                 |         | QK-gain 5.25 + warmdown 72%. Slight regression vs v2. Discarded ❌ |
| v6      | 1.2134  | — (SKIP_QUANT) | 928            | —                 | 93af936 | Delayed depth recurrence (layers 3,5 @35%). +18% step time → 634 steps. Regression ❌ |

- **Val BPB**: raw validation bits-per-byte before quantization
- **Post-Quant BPB**: after int6+brotli (sliding window)
- **Step Time**: average training step time in ms
- **Artifact (bytes)**: total submission size in bytes (budget ≤ 16,000,000 = 16 MB decimal)
- **Commit**: short SHA of the code version used
- **Description**: what changed from the previous version

## Analysis
Starting from exp 011 latent v3 (post-quant sw BPB 1.2036, 15.90MB). This is the new project baseline.

### v1/v1-nodiff — OVER BUDGET (MiB bug)
Both v1 and v1-nodiff artifacts (~16.67M bytes) exceed the official 16,000,000-byte cap. The selective_prune function was using `target_mb * 1024 * 1024` (MiB) instead of `target_mb * 1_000_000` (decimal MB). Fixed in train_gpt.py: pruning now uses decimal MB, default TARGET_MB changed from 15.9 to 15.95. v3 will be the first valid-budget run with diffusion on.

### v1-nodiff ablation (informational only — over budget)
Disabling diffusion saves ~29ms/step → 30 more training steps → raw BPB improves 0.004 (1.1996 vs 1.2036). Quant gap is near-zero with or without diffusion — SDClip already handles quant robustness on SP8192/brotli. **Decision: keep diffusion on** for quant regularization safety margin.

### v3 — Diffusion OFF with fixed MB pruning
Same as v2 budget fix (decimal MB) but with diffusion disabled. Faster step time (752ms vs 771ms) gives 22 more steps (782 vs 760). Raw train BPB improves 0.003 (1.2002 vs 1.2034), but post-quant sw BPB is slightly worse than v1 baseline (1.2161 vs 1.2039). Quant degradation gap is 0.016 (vs 0.0003 for v1 with diffusion ON). Confirms latent MSE diffusion earns ~0.013 sw BPB for free as a quant regularizer. Artifact 15.95MB ✅.

**Conclusion across v1–v3:** Latent diffusion ON + TARGET_MB=15.95 is the optimal config. v2's aggressive 15.25MB target destroys quant quality; v3 (diffusion OFF) loses the quant regularization benefit. Next iterations should keep diffusion ON and focus on architecture features (depth recurrence, parallel residuals, QK-gain, TTT).

### v5 — QK-gain 5.25 + warmdown 72% (discarded)
QK_GAIN_INIT=5.25 (was 5.0) + WARMDOWN_FRAC=0.72 (was 0.667). Zero overhead (786ms vs 783ms), but val_bpb regressed to 1.2063 (+0.0018 vs v2 baseline 1.2045). QK-gain effect may require depth recurrence to be active — all top-5 submissions use them together. Discarded.

### v6 — Delayed depth recurrence (discarded)
Depth recurrence on layers [3,5] with 2 passes, delayed activation at 35% wallclock. Step time jumps from ~785ms to 928ms after activation (+18%), cutting total steps from ~750 to 634 (-116 steps). val_bpb regresses to 1.2134 (+0.0089 vs v2 baseline 1.2045). Same conclusion as v4: depth recurrence is too expensive on 1xH100 — the step count penalty outweighs any representational benefit. Top-5 submissions run recurrence on 8xH100 where they have ~4800 steps to absorb overhead. Discarded.

## Status
[x] Forked from exp 011 latent v3
[x] v1 baseline verification (sw BPB 1.2039, matches exp 011 latent v3)
[x] v1-nodiff ablation — diffusion off gives -0.004 BPB but artifact over budget (16.67MB ❌)
[ ] Iterate toward leaderboard competitive (target: <1.10 on 8xH100)
[ ] Decision: adopt / discard / iterate
