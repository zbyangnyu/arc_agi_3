# Nuisance learned 2×2 bridge: full experiment

## Protocol

The experiment crosses:

- EE: exact belief / exact outcome model
- LE: learned belief / exact outcome model
- EL: exact belief / learned outcome model
- LL: learned belief / learned outcome model

It evaluates 192 base public environments (64 groups × 3 query axes),
three hidden query modes per environment, and three candidate-order seeds:
1,728 seeded hidden-task instances per condition and budget.

The budget is fixed: exactly `B` probes are executed, with no early stopping.
All four conditions share tasks, candidate order, feedback, factor bank, and
canonicalization. Belief and executor inference are performed once per public
environment and broadcast bitwise over its three hidden modes.

## Validity checks

- 14 focused tests passed.
- The exact/exact control reproduced its theoretical values:
  B0 tie-aware accuracy `1/3`, B1 accuracy `1`.
- All 2,304 public-environment/condition groups had an identical first action
  across their three hidden modes.
- Selector boundary, 64-code bank, posterior normalization, finite learned
  likelihood, and true-code/canonical-feedback alignment checks passed.
- Learned MAP grids were used only for acquisition partitions; posterior
  updates used the proper full-grid likelihood.

## Main results

Tie-aware fixed-budget accuracy:

| Budget | EE | LE | EL | LL |
|---|---:|---:|---:|---:|
| B0 | 33.333% | 33.333% | 33.333% | 33.333% |
| B1 | 100% | 100% | 100% | 100% |
| B2 | 100% | 100% | 100% | 99.653% |
| B3 | 100% | 100% | 99.942% | 99.537% |

The balanced hidden-mode construction makes B0 top-1 accuracy uninformative.
The calibrated B0 metric is more revealing:

- exact belief mean true-query probability: `0.33333`
- learned belief mean true-query probability: `0.28960`
- learned belief mass on the exact 48-code support: `0.86879`
- symbolic-to-learned KL: `0.41135` nats

The learned B0 true-query probability is `0.30797` for collision, `0.30757`
for trigger, and only `0.25325` for relation. Thus the learned prior is already
weakest on relation, although the strong atomic query probe hides this at B1.

Every condition selected a query-atomic probe first and solved all 1,728
seeded tasks at B1. This establishes one-step assimilation on this controlled
menu; it does not establish difficult multi-step active rule discovery.

## Where the learned update fails

Collision and trigger remain perfect after B1. Every observed error is on:

`query axis = relation`, `hidden mode = Relation.NONE`.

Relation-only accuracy in the main batch-16 run:

| Budget | EE | LE | EL | LL |
|---|---:|---:|---:|---:|
| B2 | 100% | 100% | 100% | 98.958% |
| B3 | 100% | 100% | 99.826% | 98.611% |

All failed trajectories passed through `nuisance-axis-0`. This matches the
executor audit:

| Candidate category | Exact-grid rate | Exact-partition rate | Pair F1 |
|---|---:|---:|---:|
| nuisance-axis-0 | 68.23% | 50.00% | 0.649 |
| query-atomic | 80.73% | 66.67% | 0.778 |
| nuisance-axis-1 | 93.23% | 83.33% | 0.937 |
| neutral | 90.63% | 0.00% | 0.907 |

These are high-confidence reversals, not ties. In one representative
`Relation.NONE` task, the query probe first placed essentially all posterior
mass on the true query value. A subsequent nuisance-axis-0 observation received
approximately:

- log predictive likelihood under `Relation.REPEL`: `-14.15`
- log predictive likelihood under the true `Relation.NONE`: `-50.98`

The roughly 37-nat spurious Bayes factor overwhelmed the correct query
evidence and flipped the posterior. This is direct evidence of a
misspecified, entangled learned likelihood: a nuisance observation can provide
false evidence about an already identified query factor.

## Robustness boundary

A complete batch-size-64 sensitivity run preserved:

- B0 and B1 conclusions
- all outcome-partition metrics
- localization to relation/NONE and nuisance-axis-0
- the qualitative negative learned-belief × learned-outcome interaction

But LL failures changed from `6/8` at B2/B3 to `2/4`. After B1, the query is
already solved and remaining acquisition scores are nearly tied; small numeric
changes can alter which unnecessary probe fixed-budget evaluation forces next.
Therefore the exact post-B1 failure percentage is not a stable benchmark
number. The stable scientific result is the existence and localization of
large false-likelihood counterexamples.

An entropy threshold of `1e-3` would stop every recorded trajectory at B1 and
score 100%, but that is a post-hoc observation, not a held-out evaluation of a
pre-registered stopping policy.

## Conclusion and next experiment

The current bottleneck is not first-action selection. It is causal modularity
and calibration of the learned outcome likelihood, amplified by an imperfect
learned prior.

The next experiment should:

1. pre-register and evaluate an entropy/evidence-based stopping rule;
2. remove strong atomic query probes and require at least two overlapping
   partial probes, so acquisition genuinely depends on the belief;
3. run a forced counterfactual stability audit: after identifying the query,
   apply every nuisance probe independently and measure the change in true-query
   log odds;
4. train the executor with a factor/intervention consistency objective so a
   probe of one mechanism cannot change likelihood ratios for an unrelated
   mechanism.

This is the clearest place to introduce a factorized JEPA/causal-world-model
objective around the current GRAM-style hypothesis belief.
