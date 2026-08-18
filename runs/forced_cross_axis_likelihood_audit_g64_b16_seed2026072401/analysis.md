# Forced cross-axis likelihood locality audit

## Outcome

The preregistered decision is **support-factorized-jepa-executor**. Under RR, cross-axis observations reverse a confident query belief in 2.08% of the 1,536 semantic cross-axis paths, and the higher-method P99 log-odds drop is 35.7118 nats. Under PP these become 0.00% and 1.907e-06 nats.

The fixed-geometry query positive control has raw top-1 accuracy 100.00%, with 91.67% of query cases assigning at least 0.95 probability to the true query factor.

## Learned-path decomposition (cross-axis paths)

| Branch | Query likelihood | Forced likelihood | Reversal rate | P99 drop (higher, nats) | Max drop (nats) | Mean marginal TV |
|---|---|---|---:|---:|---:|---:|
| EE | exact | exact | 0.00% | 0 | 0 | 0 |
| RR | raw | raw | 2.08% | 35.7118 | 45.5634 | 0.0372 |
| RP | raw | oracle projected | 0.00% | 3.5784 | 4.4131 | 0.0097 |
| PR | oracle projected | raw | 2.08% | 34.6338 | 43.2150 | 0.0257 |
| PP | oracle projected | oracle projected | 0.00% | 1.907e-06 | 1.907e-06 | 3.636e-08 |

RP removes catastrophic reversals but retains a non-zero tail, while PR remains close to RR. This localizes the dominant failure to raw forced-observation likelihoods; raw query likelihoods can still induce cross-factor correlations that explain the remaining RP tail.

## Failure concentration and calibration

All 32 RR reversals come from 16 unique programs on `relation -> collision`, specifically forced `cross:collision:v1`. They cover collision PASS, PUSH, relation SWAP, NONE, and every trigger value; both relation query variants fail. The PR reversal identities are exactly the same as RR.

The query projection leaves the query posterior metric essentially unchanged (projected-minus-raw posterior NLL 2.219e-08 nats), while true-code conditional full-grid NLL changes by -2.8940 nats. The latter is the preregistered gate quantity.

The weakest atomic outcome partitions are collision v1 (pair F1 0.3551) and trigger v1 (0.5590). Only collision v1 causes reversals, so partition mismatch co-localizes with the failure but is not sufficient for it.

Neutral probes cause no high-confidence reversals. Their RR P99 drop is 1.5550 nats, versus zero under RP/PP, revealing smaller raw non-local likelihood drift.

## Robustness

| Run | Split | Seed | Batch | Valid | Decision | RR reversal | RR P99 | PP reversal | PP P99 | Semantic equality |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---|
| primary | forced-cross-axis-likelihood-audit-heldout-v1 | 2026072401 | 16 | yes | support-factorized-jepa-executor | 2.08% | 35.7118 | 0.00% | 1.907e-06 | reference |
| comparison_1 | forced-cross-axis-likelihood-audit-heldout-v1 | 2026072401 | 64 | yes | support-factorized-jepa-executor | 2.08% | 35.7118 | 0.00% | 1.907e-06 | exact |
| comparison_2 | forced-cross-axis-likelihood-audit-heldout-v2 | 2026072402 | 16 | yes | support-factorized-jepa-executor | 2.08% | 35.7118 | 0.00% | 1.907e-06 | exact |

All JSON inputs passed a recursive finite-number check. Semantic equality ignores candidate positions and split/seed/batch provenance; it does not ignore any model metric.

## Worst 20 RR cross-axis paths, joined across learned branches

| # | Axis pair | True program | Query probe | Forced probe | RR drop | RP drop | PR drop | PP drop | RR reversal | PP reversal |
|---:|---|---|---|---|---:|---:|---:|---:|---|---|
| 1 | relation->collision | C=PUSH, T=TOGGLE, R=NONE | query:relation:v0 | cross:collision:v1 | 45.5634 | 1.0073 | 43.2150 | 0 | yes | no |
| 2 | relation->collision | C=PUSH, T=DELETE, R=NONE | query:relation:v0 | cross:collision:v1 | 45.5634 | 1.0073 | 43.2150 | 0 | yes | no |
| 3 | relation->collision | C=PUSH, T=SPAWN, R=NONE | query:relation:v0 | cross:collision:v1 | 45.5634 | 1.0073 | 43.2150 | 0 | yes | no |
| 4 | relation->collision | C=PUSH, T=RECOLOR, R=NONE | query:relation:v0 | cross:collision:v1 | 45.5634 | 1.0073 | 43.2150 | 0 | yes | no |
| 5 | relation->collision | C=PUSH, T=TOGGLE, R=SWAP | query:relation:v1 | cross:collision:v1 | 37.6348 | 0 | 42.1151 | 0 | yes | no |
| 6 | relation->collision | C=PUSH, T=DELETE, R=SWAP | query:relation:v1 | cross:collision:v1 | 37.6348 | 0 | 42.1151 | 0 | yes | no |
| 7 | relation->collision | C=PUSH, T=SPAWN, R=SWAP | query:relation:v1 | cross:collision:v1 | 37.6348 | 0 | 42.1151 | 0 | yes | no |
| 8 | relation->collision | C=PUSH, T=RECOLOR, R=SWAP | query:relation:v1 | cross:collision:v1 | 37.6348 | 0 | 42.1151 | 0 | yes | no |
| 9 | relation->collision | C=PASS, T=TOGGLE, R=SWAP | query:relation:v1 | cross:collision:v1 | 37.1876 | 0.5142 | 34.5758 | 0 | yes | no |
| 10 | relation->collision | C=PASS, T=DELETE, R=SWAP | query:relation:v1 | cross:collision:v1 | 37.1876 | 0.5142 | 34.5758 | 0 | yes | no |
| 11 | relation->collision | C=PASS, T=SPAWN, R=SWAP | query:relation:v1 | cross:collision:v1 | 37.1876 | 0.5142 | 34.5758 | 0 | yes | no |
| 12 | relation->collision | C=PASS, T=RECOLOR, R=SWAP | query:relation:v1 | cross:collision:v1 | 37.1876 | 0.5142 | 34.5758 | 0 | yes | no |
| 13 | relation->collision | C=PUSH, T=TOGGLE, R=SWAP | query:relation:v0 | cross:collision:v1 | 35.7118 | 0 | 34.6338 | 0 | yes | no |
| 14 | relation->collision | C=PUSH, T=DELETE, R=SWAP | query:relation:v0 | cross:collision:v1 | 35.7118 | 0 | 34.6338 | 0 | yes | no |
| 15 | relation->collision | C=PUSH, T=SPAWN, R=SWAP | query:relation:v0 | cross:collision:v1 | 35.7118 | 0 | 34.6338 | 0 | yes | no |
| 16 | relation->collision | C=PUSH, T=RECOLOR, R=SWAP | query:relation:v0 | cross:collision:v1 | 35.7118 | 0 | 34.6338 | 0 | yes | no |
| 17 | relation->collision | C=PUSH, T=TOGGLE, R=NONE | query:relation:v1 | cross:collision:v1 | 35.3836 | 0 | 42.7458 | 0 | yes | no |
| 18 | relation->collision | C=PUSH, T=DELETE, R=NONE | query:relation:v1 | cross:collision:v1 | 35.3836 | 0 | 42.7458 | 0 | yes | no |
| 19 | relation->collision | C=PUSH, T=SPAWN, R=NONE | query:relation:v1 | cross:collision:v1 | 35.3836 | 0 | 42.7458 | 0 | yes | no |
| 20 | relation->collision | C=PUSH, T=RECOLOR, R=NONE | query:relation:v1 | cross:collision:v1 | 35.3836 | 0 | 42.7458 | 0 | yes | no |

## Interpretation limits

- PP is an **oracle causal projection**, computed by factor-fiber log-mean-exp after full-grid likelihood evaluation. Its near-zero query-marginal drift is a mathematical consequence of the intervention and a mechanistic rescue test—not evidence that the current learned model can discover or apply the projection.
- The 64 palette/order groups are canonically identical invariance repeats, not 64 independent environments. Accordingly this report gives finite exhaustive rates over semantic paths and no palette-level confidence interval.
- The audit uses the two fixed, previously validated atomic probe geometries per axis. It establishes locality on this bank, not generalization to arbitrary or compositional geometries.
