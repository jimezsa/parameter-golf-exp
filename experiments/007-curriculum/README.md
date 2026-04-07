# Experiment: Progressive Data Curriculum (Difficulty Ordering)

## Paper / Source
- **Title**: QuRating: Selecting High-Quality Data for Training Language Models / DoReMi
- **Authors**: Various (2023, 2024)
- **Link**: N/A
- **Key idea**: Sorting the training data sequence from short/simple text to long/complex text to accelerate early convergence.

## Hypothesis
Sorting the FineWeb-EDU sequence from simple to complex text accelerates convergence in the critical first 3 minutes, achieving a deeper final optimum without any parameter or compute overhead.

## Base Code
- **Fork from**: `experiments/003-bitnet/train_gpt.py` (our current ternary base).

## Changes from Baseline
- **Data Loading**: Implement offline/dataloader sorting based on two proxy metrics for simplicity:
  1. Token entropy (character-level Shannon entropy of the text).
  2. Unique token ratio (unique tokens / total tokens per document).
  *Note: We avoid external model perplexity to eliminate computational overhead.*
- **Schedules**: Test 3 different curriculum schedules (e.g., linear pacing, step-wise) against 1 random baseline control.
- **Architecture**: No model architecture changes.

## Run Config
- **GPU**: 1x H100 (dev) / 8x H100 (final)
- **Steps / Duration**: 10 minutes (wallclock)
- **Key hyperparameters changed**: Dataloader sampling order.

## Results
| Run | BPB | Notes |
|-----|-----|-------|
|     |     |       |

## Analysis
What worked, what didn't, why.

## Status
- [x] Proposed by scout
- [x] Approved by professor
- [x] Implemented by engineer
- [ ] Tested by human
- [ ] Analyzed
- [ ] Decision: adopt / discard / iterate
