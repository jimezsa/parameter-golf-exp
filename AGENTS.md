# AGENTS.md — parameter-golf-exp

Repo-scoped rules for any agent running experiments here. Canonical owner: **autoresearch** specialist.

## Iteration Table Is Mandatory

**Always document every run in the active experiment's iteration results table. No exceptions.**

- After every run completes — success, failure, crash, or OOM — immediately append a row to the experiment's `README.md` under `## Iteration Results` **before** planning the next iteration.
- Columns: `Version | Val BPB | Post-Quant BPB | Step Time (ms) | Artifact Size | Commit (short SHA) | Description`.
- For failed or crashed runs, record the version, mark metrics as `FAIL` or `—`, and describe the failure mode in the Description.
- See `experiments/TEMPLATE.md` for the canonical format.
- Commit the README update alongside (or immediately after) the code change that produced the run.

Incomplete tables are not acceptable — they are the experiment's permanent record.

## Iteration Policy

- Per HP proposal: **3–5 runs** (scout / tune / confirm gates).
- Per experiment as a whole: **10+ honest iterations** before a final kill/keep verdict.
- Single-shot discard is not acceptable.

## Current Focus

Exp 013 — AR-Latent-Diffusion. See `experiments/013-ar-latent-diffusion/README.md` for the HP sweep plan (P1–P11), iteration policy, and baseline config.

## Killed Directions (do not re-propose)

Mamba, MoE, ternary, vocab-space diffusion, depth recurrence, parallel residuals, QK-gain 5.25, WD tuning.
