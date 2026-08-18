# PRP-WM architecture and context map

This is the short routing document for humans and agents. It describes where to
look; the research conclusions and experiment history remain in `README.md` and
the topic reports in this directory.

## Repository shape

| Path | Role | Current scale |
| --- | --- | ---: |
| `prp_wm/` | Core Python package | 24 source files, about 13.5k lines |
| `scripts/` | Train/evaluate/audit experiment entry points | 43 files, about 34.3k lines |
| `tests/` | Unit and experiment regression tests | 35 files, about 9.5k lines |
| `docs/` | Specifications, reports, and plans | 20 topic documents |
| `configs/` | Frozen experiment configuration | 1 compact JSON |
| `results/` | Curated reference results and manifests | 5 compact JSON files |
| `runs/` | Generated raw outputs and checkpoints | 188 directories, about 631 MB |

`runs/` is artifact storage, not part of the code architecture. Ordinary
searches should exclude it. For an experiment question, read the relevant report
or compact result first, then open only the named run.

## Functional layers

```text
Stage 0: minimal belief-update scaffold
  gf2.py -> schema.py -> evaluation.py / reproducibility.py

RuleGrid: symbolic environment and controlled evaluation
  rulegrid.py
    |- rulegrid_evaluation.py
    |- pilot.py
    |- random_geometry_protocol.py
    |- rl.py
    `- rulegame.py -> rulegame_rl.py

Learned rule hypotheses and execution
  neural.py
    `- latent_rules.py
         |- causal_filter.py
         |- causal_rules.py -> discrete_causal_rules.py
         |    |- gram_causal_rules.py -> gram_smc.py / stratified_gram.py
         |    |- public_version_k4.py
         |    `- unstructured_causal_rules.py
         |- matched_executor.py
         `- routed_executor.py
```

## Read paths by task

| Task | Start with | Usually pair with |
| --- | --- | --- |
| Exact Stage 0 behavior | `prp_wm/gf2.py`, `schema.py` | `tests/test_gf2.py` |
| Reproducibility contract | `prp_wm/reproducibility.py` | `tests/test_reproducibility.py`, `configs/`, `results/` |
| RuleGrid mechanics | `prp_wm/rulegrid.py` | `tests/test_rulegrid.py` |
| RuleGrid gate metrics | `prp_wm/rulegrid_evaluation.py` | `tests/test_rulegrid.py` |
| Neural particle model | `prp_wm/neural.py` | `tests/test_neural.py` |
| Latent executor behavior | `prp_wm/latent_rules.py` | `tests/test_latent_rules.py` |
| Version-space inference | matching `causal_*` or `public_version_k4.py` module | same-name test |
| GRAM proposal/search | matching `gram_*` or `stratified_gram.py` module | same-name test |
| RL / RuleGame | `rl.py` or `rulegame*.py` | same-name test |
| One experiment | exact `scripts/<experiment>.py` | matching test, then named report/result |

## Context hot spots

The largest reusable modules are `public_version_k4.py`, `rulegrid.py`,
`neural.py`, `gram_causal_rules.py`, and `latent_rules.py`. Read the relevant
symbols or line ranges instead of loading these files wholesale.

The scripts directory is intentionally an entry-point layer, but several runners
are larger than the reusable modules. A code task should inspect only the runner
named by the task and the library functions it calls.

## Artifact convention

- `results/` is the compact, canonical surface for reference outputs.
- `docs/` records interpretation, limitations, and go/no-go decisions.
- `runs/` contains raw per-run detail, progress streams, checkpoints, and logs.
- New experiments should expose a small summary artifact separately from large
  per-task records so routine analysis does not require reading multi-megabyte
  JSON files.
