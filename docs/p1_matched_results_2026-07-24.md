# P1 matched capacity-vs-locality experiment

Date: 2026-07-24  
Decision: **NO-GO; narrow the claim and repair outcome-partition fidelity**

## Question

Does the earlier oracle-local improvement come from factor-local structure, or
only from extra parameters / a favorable seed?

The formal comparison uses two executors with the same 41,564 parameters,
four decoder branches, deterministic public canonical router, spatial masks,
decoder initialization, data, optimizer and branch forward count:

- **E1 matched-wider-global:** every axis branch reads the full three-factor tuple.
- **E2 matched-factor-local:** an axis branch reads only its own factor; nuisance
  factors are replaced by the fixed mean reference.

Thus the manipulated variable is the factor-conditioning graph.

## Frozen protocol

- Model seeds: `2026072501`, `2026072502`, `2026072503`.
- Active-task offsets: `0`, `2560`, `5120`.
- 320 updates per run; no early stopping.
- Six randomized geometry panels/update: three singleton plus three pair axis
  sets.
- All 64 factor codes per panel: 1,920 panels and 122,880 geometry examples/run.
- The same 64 training geometry seeds are traversed for five complete cycles in
  a seed-specific fixed permutation.
- Frozen initial teacher categorical KL weight: 10,000.
- Held-out geometry seeds start at 200,000 and are disjoint from training.
- Formal executor gate: 192 held-out tasks, all 64 codes per task.
- B2: 48 paired scenarios/seed-run at budget 2.
- RR audit: 2,304 deduplicated semantic sequences/branch.

## Main results

### Per-seed executor and held-out geometry

| Seed | Condition | Active hard gate | Held-out singleton exact-map | All-64 exact panel rate | Fiber Huber | Triple NLL/cell |
|---:|---|:---:|---:|---:|---:|---:|
| 2026072501 | E1 wider-global | **fail** | 56.45% | 0.00% | 0.5178 | 8.09e-6 |
| 2026072501 | E2 factor-local | pass | 52.08% | 12.50% | **0** | 3.95e-6 |
| 2026072502 | E1 wider-global | pass | 56.58% | 0.00% | 0.5453 | 8.17e-6 |
| 2026072502 | E2 factor-local | pass | 54.17% | 12.50% | **0** | 1.86e-6 |
| 2026072503 | E1 wider-global | pass | 55.53% | 0.00% | 0.5760 | 5.72e-6 |
| 2026072503 | E2 factor-local | pass | 53.13% | 8.33% | **0** | 2.49e-6 |
| Mean | E1 wider-global | 2/3 pass | **56.18%** | 0.00% | 0.5464 | 7.33e-6 |
| Mean | E2 factor-local | **3/3 pass** | 53.13% | **11.11%** | **0** | **2.77e-6** |

E2 is easier to optimize in all three seeds: its final geometry-supervision
loss averages 0.2883 versus 0.4875 for E1. This training advantage does not
translate into better mean per-code held-out exact-map accuracy.

### B2 and cross-axis belief

| Seed | E1 B2 | E2 B2 | E1 RR reversal / P99 drop | E2 RR reversal / P99 drop |
|---:|---:|---:|---:|---:|
| 2026072501 | not opened: hard gate failed | 96.875% | not opened | 0 / 3.81e-6 nat |
| 2026072502 | **100%** | 96.875% | 0 / 2.322 nat | 0 / 3.81e-6 nat |
| 2026072503 | **100%** | 96.875% | 0 / 2.124 nat | 0 / 1.91e-6 nat |

E2 B2 mean and worst are both 96.875%. The worst-seed gate of 95% passes, but
the mean gate of 98% fails. E2 also does not beat E1 on paired B2.

E2 does decisively solve the nuisance-belief problem: all audited reversal
rates are zero and RR P99 drop is at numerical-noise scale, while both
evaluable E1 seeds exceed the 0.5-nat P99 gate by more than fourfold.

## Gate decision

| Pre-registered gate | Result |
|---|---|
| Old hard gate passes in 3/3 seeds | E1 fails (2/3); E2 passes (3/3) |
| RR reversal ≤ 0.1% | E2 passes: 0 in all three seeds |
| RR P99 drop ≤ 0.5 nat | E2 passes; E1 fails in both evaluable seeds |
| B2 mean ≥ 98%, worst ≥ 95% | E2 mean fails; worst passes |
| E2 beats matched E1 on cross-axis and B2 | Cross-axis yes; B2 no |
| E2 triple NLL degradation ≤ 0.05 nat/cell | Pass; E2 is better |

Overall decision: **NO-GO for the claim that factor-locality is the main
remaining B2 variable.**

## What the experiment establishes

1. **The model was not simply too small.** Matching E1 to 41,564 parameters
   does not remove cross-axis odds corruption and is seed-sensitive on the
   active belief gate.
2. **Factor-local structure is real and useful.** E2 produces exact nuisance
   invariance, 3/3 stable belief gates, lower triple NLL and lower training
   geometry loss.
3. **Factor-locality is not sufficient for acquisition.** E2 remains at
   96.875% B2 for all three seeds and has lower mean held-out singleton
   exact-map accuracy than E1.
4. **The residual error is an outcome-partition error.** The RR/B2 audits
   repeatedly identify `collision:v1`: the factor-local model merges the true
   four outcome classes into two or three learned classes. Aggregate partition
   F1 is 0.9643, 0.9818 and 0.9643 across E2 seeds.

## Next experiment: P1.1 partition fidelity

Keep E2 frozen as the architecture and change only the training target:

1. Add a simulator-derived, training-only outcome-partition loss on canonical
   probe panels. It should reward equality for hypotheses with the same next
   grid and enforce a margin between hypotheses with different next grids.
2. Stratify minibatches by semantic probe key and oversample
   `collision:v1`; retain the current proper categorical NLL.
3. Compare proper-NLL-only versus proper-NLL plus partition loss with identical
   data, parameters and seeds.
4. Re-run the same active gate, RR, partition F1 and 48-scenario B2 protocol.
5. Require 3/3 active gates, partition F1 = 1 on every semantic probe, RR P99
   ≤ 0.5 nat, and B2 mean ≥ 98% before proceeding to a learned router.

Do not increase model size, add JEPA, or move to Push-T yet. The current
failure is already localized to a small discrete outcome partition, so those
changes would confound the diagnosis.

## Reproducibility

- Implementation: `prp_wm/matched_executor.py`
- Training runner: `scripts/run_counterfactual_locality_finetune.py`
- Full repository test suite after implementation: **276 passed**
- Formal checkpoints and JSON results:
  - `runs/p1_matched_wider_global_seed202607250{1,2,3}/`
  - `runs/p1_matched_factor_local_seed202607250{1,2,3}/`
  - `runs/p1_b2_matched_{wider_global,factor_local}_seed202607250{1,2,3}`
  - `runs/p1_rr_matched_{wider_global,factor_local}_seed202607250{1,2,3}/`

E1 seed 1 B2/RR is intentionally absent because the downstream runners reject
checkpoints that fail the active-prefix hard gate.
