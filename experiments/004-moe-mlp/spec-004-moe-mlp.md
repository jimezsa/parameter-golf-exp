# Specification: Experiment 004 - Sparse MoE MLP

## Overview
This specification details the implementation of **Experiment 004: Sparse MoE MLP**, targeting the 1.1147 BPB Parameter Golf SOTA. This strategy leverages Mixture of Experts (MoE) under severe constraints (11L, 512d, 16MB GPTQ Int6, 10-minute training limit). 

**Source Base:** Fork from `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py`
**Target Location:** `parameter-golf-exp/experiments/004-moe-mlp/`

---

## 1. Architectural Modifications

### 1.1. MoE Block Structure
Replace the existing dense MLP blocks with a Sparse MoE architecture.
- **Total Experts:** Configurable (default `n_experts=8`).
- **Expert Configuration:** 
  - **Shared Experts:** `n_shared=1` (always active, no routing overhead). Isolates common syntax to survive extreme quantization.
  - **Routed Experts:** `n_routed=7` (from the default total of 8).
- **Routing Strategy:** Top-K routing with `top_k=1` for the routed experts.
- **Hyperparameters to Expose:** Ensure `n_experts`, `n_shared`, and `top_k` are easily configurable from the command line or config file to allow quick sweeps (e.g., 4/6/8 total experts).

### 1.2. Load Balancing (Gating)
- **Mechanism:** Implement **Normalized Sigmoid Gating** (DeepSeekMoE style).
- **Constraint:** **NO auxiliary load-balancing loss.** Standard auxiliary losses take too long to converge within our ~20k step budget, risking early expert collapse. The normalized sigmoid natively encourages distribution without fighting the primary objective.

---

## 2. Optimization Strategy (Hybrid)

Muon's Newton-Schulz iterations on highly inactive experts or sparse outputs can destabilize training. Thus, a hybrid optimization approach is required:
- **Muon Optimizer:** Apply *strictly* to the 2D weight matrices (the experts' MLP weights). Muon's geometric matrix updates naturally balance updates across dimensions, ensuring even conditioning.
- **AdamW Optimizer:** Apply to all 1D/0D parameters (Router weights, Embeddings, LayerNorms). Applying Muon to the router will destabilize the gating logic.

---

## 3. Quantization & Calibration (CRITICAL RISK)

Standard GPTQ destroys rare experts due to representation imbalance in the calibration set.
- **Requirement:** Implement **Expert-Balanced GPTQ Calibration** (Expert-Balanced Self-Sampling).
- **Mechanism:** Ensure that the calibration dataset forces proportional representation for every expert. The routing passes must occur during calibration to uniformly sample.
- **Profiling Mandate:** GPTQ calibration time is the primary wallclock risk. **Profile the calibration phase immediately.** If expert-balanced calibration exceeds 60 seconds of our 10-minute budget:
  1. Reduce the number of calibration passes.
  2. Implement an approximation method for calibration sampling.

---

## 4. Logging & Telemetry

- **Expert Load Distribution:** Add logging to track the activation frequency (load distribution) of each routed expert. This is critical for detecting expert collapse early in the run since we are omitting the auxiliary balancing loss.
- **Gradient Norms:** Monitor and log gradient norms of the experts specifically, ensuring that the Muon optimizer's sparse matrix condition does not blow up the updates.

---

## 5. Implementation Deliverables
1. `train_gpt.py` (forked and modified with MoE, Hybrid Optimizer, and Load Logging).
2. `gptq_calib.py` (or equivalent) modified for Expert-Balanced Calibration with strict time profiling.
3. Updated launch scripts to support configurable MoE arguments.