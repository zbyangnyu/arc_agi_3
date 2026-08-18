# PRP-WM agent guide

## Default scope

- Start from the files named by the task. For unfamiliar work, read
  `docs/architecture.md`, then inspect only the relevant source and paired test.
- Do not recursively list, search, summarize, or read `runs/` unless the task
  names a run or explicitly asks for raw experiment analysis.
- Prefer the small canonical artifacts in `results/` and the reports in `docs/`
  before opening raw run output.
- Keep searches scoped to `prp_wm/`, `scripts/`, `tests/`, `docs/`, `configs/`,
  and `results/`. The root `.ignore` removes generated artifacts from ordinary
  ripgrep searches; use `rg --no-ignore` only for intentional artifact work.
- Do not scan every script or report to learn the project. Use the routing below.

## Repository map

- `prp_wm/`: reusable library and model implementation.
- `scripts/`: experiment, training, evaluation, and audit entry points.
- `tests/`: tests paired by feature/module name.
- `docs/`: specifications, experiment reports, and research plans.
- `configs/`: frozen reproducibility inputs.
- `results/`: compact reference outputs and manifests.
- `runs/`: generated raw outputs, traces, logs, and checkpoints; excluded by
  default.

## Task routing

- Stage 0 / GF(2): `gf2.py`, `schema.py`, `evaluation.py`,
  `reproducibility.py`; tests with the same names.
- RuleGrid environment and oracle evaluation: `rulegrid.py`,
  `rulegrid_evaluation.py`, `random_geometry_protocol.py`.
- Neural particle model and latent executors: `neural.py`, `latent_rules.py`,
  `matched_executor.py`, `routed_executor.py`.
- Causal/version-space inference: `causal_filter.py`, `causal_rules.py`,
  `discrete_causal_rules.py`, `public_version_k4.py`,
  `unstructured_causal_rules.py`.
- GRAM and sequential search: `gram_causal_rules.py`, `gram_smc.py`,
  `stratified_gram.py`.
- RL and RuleGame: `rl.py`, `rulegame.py`, `rulegame_rl.py`.
- Experiment behavior normally lives in one `scripts/<name>.py` file plus its
  matching `tests/test_<name>.py`.

## Verification

- Required project version: Python 3.12; the frozen Stage 0-A reference requires
  exactly Python 3.12.3.
- Run the narrowest relevant test first:
  `python -m pytest -q tests/test_<feature>.py`.
- After cross-module changes, run `python -m pytest -q`.
- On the current workstation, `/Users/yangzhenbang/anaconda3/bin/python3`
  provides Python 3.12 with pytest and PyTorch. `/opt/homebrew/bin/python3`
  is the exact Python 3.12.3 Stage 0-A reference interpreter.
- Set `PYTHONDONTWRITEBYTECODE=1` when running checks to avoid new bytecode
  artifacts.

## Change boundaries

- Treat `configs/` and `results/` as reproducibility artifacts; update them only
  when the task explicitly changes a frozen experiment or reference output.
- Never rewrite or delete raw experiment artifacts as a side effect of a code
  task.
- Preserve public/simulator-only data boundaries described in the relevant
  tests and research protocol.
