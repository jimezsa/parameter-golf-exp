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

| Version | Val BPB | Post-Quant BPB | Step Time (ms) | Artifact Size | Commit | Description |
|---------|---------|----------------|----------------|---------------|--------|-------------|
| v1      | 1.2036  | 1.2039 (sw)    | 776ms          | 15.90MB ✅    |        | Fork verification — matches exp 011 latent v3 |

- **Val BPB**: raw validation bits-per-byte before quantization
- **Post-Quant BPB**: after int6+brotli (sliding window)
- **Step Time**: average training step time in ms
- **Artifact Size**: compressed model size (target <= 16MB)
- **Commit**: short SHA of the code version used
- **Description**: what changed from the previous version

## Analysis
Starting from exp 011 latent v3 (post-quant sw BPB 1.2036, 15.90MB). This is the new project baseline.

## Status
[x] Forked from exp 011 latent v3
[x] v1 baseline verification (sw BPB 1.2039, matches exp 011 latent v3)
[ ] Iterate toward leaderboard competitive (target: <1.10 on 8xH100)
[ ] Decision: adopt / discard / iterate
