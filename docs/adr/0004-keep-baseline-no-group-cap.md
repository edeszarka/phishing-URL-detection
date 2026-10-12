# 0004: Keep baseline; no per-group cap

## Status
Accepted. Unsuccessful change: the duilawyeryork.com exclusion did not help.

## Context
duilawyeryork.com is the second-largest platform-level group (257 rows, all
phishing; ADR 0002). An earlier comparison in `results/baseline_oof.json`
(recall at FPR <= 1% 0.4101 with the group in training vs 0.4671 without) was a
measurement artefact. The "without" run re-split the folds and re-chose its
threshold, so it changed three things at once. Do not cite `baseline_oof.json`
for this effect.

A paired experiment (`results/group_ablation.json`) holds the folds, the scored
rows and the threshold rule fixed, and removes the group only from training.
It repeats this over fold seeds 0-4, with 2,000 platform-level group resamples
per seed. Difference in recall at FPR <= 1%, all 8,325 rows scored:

| fold seed | baseline | without duilawyeryork | difference | group-bootstrap 95% CI |
|---|---|---|---|---|
| 0 | 0.4373 | 0.4383 | 0.001 | -0.01415 to 0.01835 |
| 1 | 0.4471 | 0.4582 | 0.0111 | -0.00979 to 0.01561 |
| 2 | 0.487 | 0.4931 | 0.0061 | -0.00723 to 0.02219 |
| 3 | 0.5602 | 0.5602 | 0.0 | -0.00977 to 0.02105 |
| 4 | 0.4383 | 0.4331 | -0.0052 | -0.01105 to 0.00789 |

All intervals include 0 and the sign changes across seeds. Scoring only the
8,068 rows outside the group gives the same picture.

## Decision
Keep the baseline unchanged (ADR 0003): train on all cleaned rows, with no group
excluded and no per-group cap.

## Alternatives
- Exclude duilawyeryork.com from training: no measurable effect (above).
- Cap rows per platform-level group: not tried. With a cap of 50, 7 groups and
  745 training rows would be affected. Left as a next step.

## Consequences
- The primary model, the threshold 0.74956 and ADR 0003 are unchanged.
- Baseline recall at FPR <= 1% ranges 0.4373-0.5602 across the five fold seeds.
  This split sensitivity is larger than any variant difference, and the
  group-bootstrap CIs do not include it.
- Configurations compared on OOF: 3 (baseline, stripped-input probe,
  duilawyeryork exclusion), each with 1 hyperparameter configuration. The
  per-group cap was not run.
