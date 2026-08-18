# Geometry-randomized executor protocol

This protocol is an isolated follow-up to the fixed-template executor audit. It
does not modify `rulegrid.py`, `pilot.py`, `latent_rules.py`, `neural.py`, or an
existing training runner, and it does not start training.

## Causal question

The intended future experiment asks whether a privileged factor-conditioned
executor can learn local mechanisms from randomized singleton and pair events,
then compose them on unseen randomized triple layouts. It still receives the
benchmark-supplied collision/trigger/relation code and canonical palette roles;
it does not test discovery of those variables.

| Split | Public event scope | Targets | Geometry seeds |
|---|---|---|---|
| Train | Three singleton types and three unordered pair types | All 64 privileged factor codes | Train-only seed domain |
| Eval | Triple composition only | All 64 privileged factor codes | Disjoint eval-only seed domain |

## Geometry generation

- Collision and relation events sample all four public movement directions.
- Their object geometry is either a single cell or a connected two-cell domino
  perpendicular to motion, which prevents overlap with the one-step target.
- Trigger events retain RuleGrid's public eastward
  `trigger | payload | socket` convention but randomize legal positions.
- Zero to four inert distractor cells are sampled outside every potential write
  envelope. They enlarge nuisance geometry support without changing a rule.
- Composite panels are rejection-sampled until the full potential write
  envelopes of their local events are pairwise disjoint.
- Composite action atom order is randomized. Geometry hashes sort the action
  set first, so a different atom order cannot masquerade as a new geometry.

Every accepted panel is simulated under all 64 programs. Acceptance requires:

1. exactly `4^m` distinct outcomes for `m` selected axes;
2. four distinct outcomes for every selected axis value conditional on every
   assignment of the other selected axes;
3. pairwise-disjoint unions of cells written by the selected mechanisms;
4. no simulator error for any factor code.

## Leakage boundary

Model inputs contain exactly `state`, public `action`, and the intentionally
privileged `factor_code`. Split, geometry seed, geometry hash, panel kind, axis
names, task IDs, and probe IDs remain audit metadata and are absent from the
model record. Geometry hashes are computed only from the grid and a
canonicalized public action set.

The manifest fails closed if train or eval hashes repeat, if the train/eval
hash intersection is non-empty, or if any semantic invariant fails. A separate
coverage gate checks that both splits contain all four movement directions,
both motion shapes, and at least four distinct anchors per axis.

## Executable artifacts

- Generator: `prp_wm/random_geometry_protocol.py`
- Manifest audit: `scripts/audit_random_geometry_executor_protocol.py`
- Invariants: `tests/test_random_geometry_protocol.py`

Example audit command:

```bash
/Users/yangzhenbang/anaconda3/bin/python3 \
  scripts/audit_random_geometry_executor_protocol.py \
  --output runs/random_geometry_protocol/manifest.json
```

The next implementation should consume `RandomGeometryDataset.iter_examples`
in a new training script, train on its train iterator only, freeze the resulting
checkpoint, and evaluate only its eval iterator. Geometry seed ranges and the
manifest hash should be recorded in that checkpoint before any optimization.
