# URL phishing classifier: report

**OOF** = out-of-fold predictions on cleaned `train.csv`: every training row scored by
a model that did not see its platform group (5-fold grouped CV, ADR 0002). Test and
external numbers come from a single run (`results/evaluation_runs.jsonl`) at thresholds
fixed beforehand (ADR 0003). Brackets name the file each number comes from.

## 1. Result for the manager

The model is char 3-5-gram TF-IDF + logistic regression on the raw URL. The threshold is
0.74956, chosen on OOF for FPR <= 1%. I stand behind the **external set** and the **test
rows with no training overlap**, not the test headline [results/evaluation.json]:

| evaluation | AUC | TP / FP | TPR | FPR | precision at 1:1000, 95% interval |
|---|---|---|---|---|---|
| OOF (validation) | 0.9421 | 1,779 / 42 | 0.4373 | 0.00987 | 0.0425 (0.018-0.209) |
| test, all 2,858 | 0.9714 | 875 / 5 | 0.6123 | 0.0035 | 0.149 (0.080-0.468) |
| test, no overlap (1,508) | 0.9472 | 249 / 4 | 0.4519 | 0.00418 | 0.098 (0.049-0.315) |
| external, all 2,858 | 0.9127 | 304 / 0 | 0.2127 | 0 (<= 0.0031) | >= 0.058 (exact bound) |

OOF row: `results/oof_headline.json`. Intervals are platform-group bootstrap; with 0
false positives that collapses, so external uses the exact Clopper-Pearson bound
(joint 95%, alpha 0.025 per rate).

**Supported:** the model ranks phishing above clean on both sources (AUC >= 0.91);
recall at a low-FPR threshold does not transfer (about one phishing URL in five on the
separate collection); at 1:1000 most alerts would be false on every set. **Not
supported:** that test numbers predict future traffic, or a deployed precision on
unseen sources above the ~0.06 lower bound.

**What each set tells us:**
- **OOF:** performance on platforms not seen in training. It is the only data that
  chose the threshold.
- **Test: same feeds as training, and overlapping** [results/overlap.json].
  - 1,350 of 2,858 URLs share at least a platform group with training. 145 are the same
    page after the external-set normalisation (0 exact matches), so test was not
    filtered against training.
  - Same-site rows (813) reach TPR 0.7734 with 0 false positives. No-overlap rows give
    TPR 0.4519, close to OOF. **Observation:** the test headline is optimistic because
    of overlap.
- **External: a different source.** 2,589 of 2,858 URLs match training at no level.
  Yet even on no-overlap rows TPR is 0.1224, against 0.4519 for no-overlap test rows,
  so overlap does not explain the gap.
  **Hypothesis:** the external URLs differ in style, notably in scheme/`www.`
  conventions (probe evidence in section 2). Not tested further.
- **What the external normalisation misses:** it compares whole URLs, so the same page
  with other query parameters, fragments or encoding, and sibling pages on one host or
  platform, pass. External still shares 24 URLs at hostname level or stricter and 269
  at platform level with training [results/overlap.json].

## 2. URL observations

These are patterns of these feeds, not of phishing in general.

- **[OBS]** `www.` appears in 2,813 of 4,257 clean rows (0.6608) vs 826 of 4,068
  phishing rows (0.203) [results/eda.json].
  - **[HYP] "the model mainly ranks on the scheme/`www.` prefix": weakened on
    training-like data, not on the external source.** Stripping both costs little on
    OOF (AUC 0.9346 vs 0.9421) [results/oof_headline.json] and on test (0.9645 vs
    0.9714). It costs a lot on external (0.8053 vs 0.9127) [results/evaluation.json].
- **[OBS]** Phishing clusters on shared hosting platforms. On the 650 rows where
  tenant- and platform-level keys differ (538 phishing), AUC is 0.9 if other tenants
  of the platform may be in training and 0.6092 if not [results/controls.json]. All 32
  `.app` rows are phishing, from only 5 platform groups [results/eda.json].
- **[HYP] contradicted: "HTTPS means legitimate".** 1,353 of 4,068 phishing rows use
  https (0.3326) vs 1,874 of 4,257 clean (0.4402) [results/eda.json].

## 3. Threshold and deployment

**Threshold:** 0.74956 is the OOF FPR <= 1% point at fold seed 0, fixed before any
test scoring (ADR 0003). OOF: TP 1,779, FP 42, TPR 0.4373, FPR 0.00987
[results/oof_headline.json].
- **Split sensitivity:** over fold seeds 0-4, recall at FPR <= 1% ranges 0.4373-0.5602
  and its threshold 0.6825-0.75171 [results/group_ablation.json]. Group-bootstrap
  intervals do not include this split variance. **[HYP]** The spread comes from which
  large platforms share a fold: blogspot.com alone gives 26 of the 42 seed-0 false
  positives [results/oof_headline.json].
- **Carry-over:** the model refit on all rows keeps the OOF threshold. Test FPR 0.0035
  and external 0 are inside the per-fold OOF range 0-0.0317
  [results/oof_headline.json]; the threshold was not re-tuned.

**Deployed precision** is derived from measured rates, not from `probability`:
precision = TPR·π / (TPR·π + FPR·(1-π)) with π = 0.001. FPR dominates, and it is the
rate the samples measure worst: each set has 1,429 clean URLs, so one false positive
moves FPR by 1/1,429. Exact joint-95% FPR intervals are 0.00095-0.0089 (test) and
0-0.0031 (external), implying precision 0.0615-0.4037 and >= 0.058
[results/evaluation.json]. The samples support "precision at 1:1000 is likely at least
about 0.06"; they cannot distinguish 0.06 from 0.4.

**Recommendation: prioritise review; do not block or warn users.** Even at the upper
estimates most alerts are false. At the external lower bound, about 6 in 100 alerts are
phishing, and only about one phishing URL in five from an unfamiliar source is caught.
The stricter 0.84127 point stays a scenario: test precision 0.225 (bootstrap
0.093-1.0) from 2 false positives, external TPR 0.0903 [results/evaluation.json].
**Evidence needed before blocking:** labelled deployment traffic with enough clean URLs
to bound FPR near 0.1%, false positives by hosting platform, a time-based split, and
recall on phishing from sources outside the training feeds.

## 4. Data handling

| Issue (raw 8,651 rows) | Rule | Rows dropped |
|---|---|---|
| 50 normalised keys with both labels | drop all | 107 (57 phish / 50 clean) |
| 219 normalised same-label copies | keep first in file | 219 (205 / 14) |
| 2 internal whitespace, 5 non-ASCII; all parse | keep | 0 |

- **Result:** 8,325 rows, 4,068 phishing [results/cleaning.json]. Dropping conflicts
  makes the data easier than real traffic.
- **Invalid inputs** (ADR 0005): a non-string or blank input returns `clean` with
  probability 0.0, meaning "not scored"; strings are cut at 4,000 characters. Neither
  evaluation set has such a URL [results/overlap.json]. A deployed system should route
  unscored inputs to their own queue, not into the clean stream.

## 5. Decision log (observation -> action -> effect)

1. Conflicts and copies -> drop/dedupe (ADR 0001) -> 8,325 rows. The representative
   choice barely matters: AUC 0.9421 / 0.9422 / 0.9417 [results/controls.json].
2. Tenant-level folds leaked platforms (AUC 0.9 vs 0.6092) -> platform-level groups
   (ADR 0002) -> recall at 1% FPR drops from 0.5951 to 0.4373 [results/controls.json].
3. `www.` gap -> stripped-input variant -> AUC 0.9346 vs 0.9421 on OOF. Not adopted,
   kept as a probe (ADR 0003); on external it exposes prefix dependence (section 2)
   [results/oof_headline.json].
4. The 0.1% FPR point rests on 4 false positives -> operate at 1% FPR, threshold
   0.74956 (ADR 0003); test FPR stayed inside the OOF fold range (section 3)
   [results/oof_headline.json].
5. Unsuccessful: duilawyeryork.com has 257 phishing rows -> excluded from training,
   paired over 5 fold seeds -> recall difference -0.0052 to 0.0111, all CIs include 0;
   baseline kept, no cap (ADR 0004) [results/group_ablation.json]. On test, removing
   its 91 rows from scoring moves TPR from 0.6123 to 0.5859 [results/evaluation.json].

## 6. Limitations and next steps

- **Configurations tried:** 1 hyperparameter configuration [results/baseline_oof.json].
  3 variants compared on OOF: baseline, stripped-input probe, duilawyeryork exclusion
  [results/group_ablation.json]. No variant was compared on test labels.
- **Scope:** the training feeds do not represent phishing in general (section 1).
- **Intervals:** exact intervals ignore grouping, the bootstrap breaks down at 0 false
  positives, and neither includes split variance. `probability` is uncalibrated and
  refers to the cleaned training mix (4,068 of 8,325 phishing).
- **Unfinished:** per-group cap (ADR 0004), effect of cleaning on metrics (ADR 0001),
  time-based split, host or reputation features, reproduction on a second OS.
- **Next:** labelled deployment traffic to measure FPR at the 0.1% scale; a model less
  dependent on URL-prefix conventions, judged on external-style data; calibration to
  deployment prevalence once such data exists.
