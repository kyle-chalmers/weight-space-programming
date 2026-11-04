# weight-space-programming: project instructions

> IMPORTANT: Everything in this repo is public-facing, so do not place any sensitive info here and make sure to distinguish between what should be internal-facing info (e.g. secrets, PII, recording guides/scripts), and public-facing info (instructions, how-to guides, actual code utilized). If there is information the assistant needs across sessions but should not be published, put it in the `.internal/` folder which is ignored by git per the `.gitignore`.

## What this repo is

A replication, on one Mac, of the Qwen3.5-0.8B loop-fixing edit from *Sparks of Weight-Space Programming*, plus random
controls and a blind attempt by Claude to make its own edit. `README.md` is the overview; `results.md` is the full
write-up.

## Rules for any session working here

- Activate `.venv` for every Python command (`./setup.sh` creates it).
- Never commit `models/`, `vendor/`, `probes/` or any `.safetensors` file. Edits are shared as `manifest.json` only.
- GSM8K test problems are for before/after scoring only. Any session that builds an edit works from `--split practice`
  and never opens `results/*-test*.json`.
- Every number in `results.md` comes from a result JSON summary or a manifest in this repo. Keep it that way.
