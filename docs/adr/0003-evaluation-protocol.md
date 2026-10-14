# 0003: Evaluation protocol, fixed before any test file is scored

## Status
Accepted. Written before `data/test.csv` or `data/external_test.csv` has been
scored; no test or external label has been read.

## Context
Grouped OOF results (platform-level folds, ADR 0002; `results/oof_headline.json`),
8,325 cleaned rows (4,068 phishing, 4,257 clean), 5,130 groups:

| point | threshold | TP / FP | TPR | FPR | precision at 1:1000 |
|---|---|---|---|---|---|
| FPR <= 1% | 0.74956 | 1,779 / 42 | 0.4373 | 0.00987 | 0.0425 |
| FPR <= 0.1% | 0.84127 | 848 / 4 | 0.2085 | 0.00094 | 0.1817 |

- Intervals at the 1% point: group bootstrap FPR 0.00169-0.02363, precision at
  1:1000 0.018-0.209; exact binomial FPR 0.00712-0.01331, precision 0.031-0.060.
- Intervals at the 0.1% point: exact binomial FPR 0.000256-0.002404, precision at
  1:1000 0.076-0.464; group bootstrap precision upper bound 1.0. The point rests on
  4 false positives and its threshold was chosen on the same OOF rows.
- False positives cluster by platform: at the 1% point 26 of 42 are blogspot.com
  (91 clean rows), 5 microsoftonline.com (all 5 of its clean rows).
- Per fold model at the pooled 1% threshold, FPR ranges 0.0 to 0.0317 and each
  fold's own 1% threshold ranges 0.632 to 0.786.

## Decision
1. **Primary model**: the baseline (char 3-5-gram TF-IDF, `lowercase=False`, raw
   URL, logistic regression C=1), one hyperparameter config, refit on all 8,325
   cleaned rows.
2. **Operating threshold**: 0.74956, the OOF threshold for FPR <= 1%. The
   0.84127 point (FPR <= 0.1%) is reported only as an "evidence needed" scenario.
3. **Threshold carry-over**: the final model is refit on all rows, and the OOF
   probability value is copied unchanged. The per-fold spread above is the
   evidence of how far that transfer can move FPR. It cannot be checked on
   training data. The FPR measured on the test sets is the check, and the
   threshold is not re-tuned on test results.
4. **Probability**: no calibration. `probability` is the logistic-regression
   score for the cleaned training population (48.9% phishing), not for 1:1000
   traffic. Deployment precision is derived from TPR and FPR, not from probability.
5. **Uncertainty**: group bootstrap over platform-level groups (2,000 resamples,
   seed 0) plus exact Clopper-Pearson intervals on row counts. Raw TP and FP
   counts are reported with every rate.
6. **Shortcut probe**: the same pipeline on URLs without the leading scheme and
   `www.` (`without_scheme_and_www`), refit on all rows. Its thresholds are fixed
   now from its own OOF: 0.69256 (FPR <= 1%) and 0.88479 (FPR <= 0.1%). Its only
   role is to show how much test performance depends on the scheme and `www.`
   prefix. It cannot replace the primary model, whatever its test results.
7. **Overlap with training** (URLs only), reported per level and used to split
   test metrics by the strongest level: exact URL; normalised URL (external-set
   rule); normalised with path case kept; hostname; hostname plus first path
   segment (near-duplicate); tenant-level group; platform-level group.
8. **Test labels** are read only in `src/evaluate.py`, run once per set. Any rerun
   after seeing results is logged with its reason. The duilawyeryork.com
   with/without split is repeated on both sets. Unlabelled URL features of the
   test sets (`www.`, scheme, IP share) are compared descriptively after the
   model is frozen, and that comparison is disclosed.

## Alternatives
- Operating at FPR <= 0.1%: closer to what 1:1000 traffic needs, but rests on 4
  false positives and an in-sample threshold.
- Re-choosing the threshold on the final model: there is no held-out data left
  to choose it on without touching test labels.
- Sigmoid or isotonic calibration: does not change TPR/FPR at a threshold and
  adds a fitted step with only 4-42 events in the tail.
- Selecting between primary and probe on test results: that would turn the test
  sets into validation data.

## Consequences
- The headline claim is limited to recall at a fixed OOF threshold with its
  measured test FPR, and precision at 1:1000 derived from both, with intervals.
- At the operating point, 95.75% of alerts at 1:1000 prevalence would
  be false (precision 0.0425), so the result supports review prioritisation, not
  blocking.
- Measured test effect (single run at 59a648f, `results/evaluation.json`): at
  0.74956, test TP/FP 875/5 (TPR 0.6123, FPR 0.0035), external 304/0 (TPR 0.2127,
  FPR 0, exact joint-95% upper bound 0.003062). FPR stayed inside the per-fold OOF
  range on both sets; the threshold was not re-tuned. Test no-overlap rows give
  TPR 0.4519, same-site rows 0.7734, so the test headline is inflated by overlap.
  The probe lowers AUC from 0.9127 to 0.8053 on external and from 0.9714 to 0.9645
  on test.

## Note, 2026-10-14 (before any test or external label was read)
Two details fixed before the run, without changing the decisions above:

- **Headline interval for deployed precision**: the platform-group bootstrap
  percentile interval on precision at 1:1000 itself. It resamples TPR and FPR
  together and respects grouping. Clopper-Pearson ranges are cross-checks: alpha
  0.025 per rate (joint coverage >= 95% by Bonferroni; OOF range at the 1% point
  0.0295-0.0629) and 0.05 per rate (as above, 0.031-0.060, kept for comparison).
  Both ignore grouping and pair interval ends. If the bootstrap FPR interval has
  zero width (e.g. no false positives), it is flagged and the exact FPR upper bound
  is used. `src/evaluate.py --dry-run` reproduces `results/oof_headline.json`.
- **Overlap strata** for decision 7, from strictest level per URL
  (`results/overlap.json`): same page (exact, normalised, normalised with path
  case), same site (host plus first path segment, hostname), same group (tenant
  or platform group), none.
