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
- Measured test effect: not yet measured.
