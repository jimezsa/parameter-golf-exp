# Experiment Template

Copy this structure when creating a new experiment.

## Folder Naming

`<number>-<short_name>` — e.g., `1-depth_recurrence`, `2-sparse_moe`

Numbers are sequential. Name should be descriptive but short (snake_case).

## Required Files

```
experiments/<number>-<short_name>/
  README.md          # Experiment card (see below)
  train_gpt.py       # Modified training script (fork from SOTA or previous best)
  results/           # Training logs, metrics, artifacts from runs
```

## README.md Template

```markdown
# Experiment: <Name>

## Paper / Source
- Title:
- Authors:
- Link:
- Key idea (1-2 sentences):

## Hypothesis
What we expect to improve and why.

## Changes from Baseline
Concise list of what was modified in train_gpt.py.

## Run Config
- GPU: 1x H100 (dev) / 8x H100 (final)
- Steps / Duration:
- Key hyperparameters changed:

## Iteration Results

| Version | Val BPB | Post-Quant BPB | Step Time (ms) | Artifact (bytes) | Commit | Description |
|---------|---------|----------------|-----------------|-------------------|--------|-------------|
| v1      |         |                |                 |                   |        | Initial run |

- **Val BPB**: raw validation bits-per-byte before quantization
- **Post-Quant BPB**: after int8+lzma (or int6+lzma if applicable)
- **Step Time**: average training step time in ms
- **Artifact (bytes)**: total submission size in bytes (budget ≤ 16,000,000 = 16 MB decimal)
- **Commit**: short SHA of the code version used
- **Description**: what changed from the previous version

## Analysis
What worked, what didn't, why. Update this after each significant iteration.

## Status
[ ] Proposed by scout
[ ] Approved by professor
[ ] Implemented by engineer
[ ] Tested by human
[ ] Analyzed
[ ] Decision: adopt / discard / iterate
```

## Workflow

1. **Scout** proposes experiment with paper source and hypothesis in README.md
2. **Professor** reviews, refines, approves
3. **Engineer** implements in train_gpt.py (fork from SOTA or previous best)
4. **Human** runs on 1x H100 pod, saves logs to results/
5. **Analyst** reviews results, writes analysis
6. **Professor** decides: adopt into next baseline, discard, or iterate
