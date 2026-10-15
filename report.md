# URL phishing classifier: report

**Summary.** A char 3-5-gram TF-IDF + logistic regression on the raw URL (C=1, one
configuration) at threshold 0.74956 (OOF FPR <= 1%) catches 44% of phishing on unseen
platforms (OOF), 45% on test URLs without training overlap and 21% on the separate
external collection (61% on the full test set, inflated by overlap). At 1 phishing URL in
1,000, about 4% of alerts would be phishing (OOF: 0.018-0.209, the most false positives
of any set), so I recommend prioritising review, not blocking.

**OOF** = out-of-fold predictions on cleaned `train.csv`, folds grouped by platform
(ADR 0002). Test and external were scored once at thresholds fixed beforehand (ADR 0003);
two logged reruns only added fields (`results/evaluation_runs.jsonl`, git history of
`results/evaluation.json`). Brackets give each number's source file.

## 1. Result for the manager

**Metric:** recall at the fixed threshold with its measured FPR, and precision at 1:1000
derived from both. **I stand behind** OOF, external and the test rows without training
overlap, not the test headline [results/oof_headline.json, results/evaluation.json]:

| evaluation | AUC | TP / FP | recall (TPR) | FPR | precision on sample | precision at 1:1000, 95% interval |
|---|---|---|---|---|---|---|
| OOF (validation) | 0.9421 | 1,779 / 42 | 0.4373 | 0.00987 | not computed | 0.0425 (0.018-0.209) |
| test, all 2,858 | 0.9714 | 875 / 5 | 0.6123 | 0.0035 | 0.9943 | 0.149 (0.080-0.468) |
| test, no overlap (1,508) | 0.9472 | 249 / 4 | 0.4519 | 0.00418 | 0.9842 | 0.098 (0.049-0.315) |
| external, all 2,858 | 0.9127 | 304 / 0 | 0.2127 | 0 (<= 0.0031) | 1.0 | >= 0.058 (exact bound) |

Intervals: 95% bootstrap over platform groups at the fixed threshold, the headline
interval type chosen before any test label was read (ADR 0003). With 0 false positives
the bootstrap has zero width, so external uses the exact Clopper-Pearson bound (joint
95%). Precision on sample reflects the 1:1 class mix, not
deployment.

**Overlap** (`src/overlap.py`, ADR 0003): each evaluation URL gets the strictest of seven
levels at which it matches cleaned training [results/overlap.json]: **same page** = exact
string, or equal after the external-set normalisation (case, scheme, leading `www.`,
trailing slash), with or without path case; **same site** = same hostname plus first path
segment, or same hostname; **same group** = same tenant- or platform-level registered
domain (ADR 0002); otherwise **none**.
- **OOF:** platforms unseen in training; the only data that chose the threshold.
- **Test: same feeds as training, overlapping.** 1,350 of 2,858 URLs share at least a
  platform group with training; 145 are the same page after the external-set
  normalisation (0 exact). Same-site rows (813) reach TPR 0.7734 with 0 false positives,
  no-overlap rows 0.4519, close to OOF. **[OBS]** Overlap inflates the test headline.
- **External: a different source.** 2,589 of 2,858 URLs match training at no level, yet
  no-overlap TPR is 0.1224 vs 0.4519 on test, so overlap does not explain the gap.
  **[OBS]** The ranking transfers, the score scale does not: on its own ROC curve
  external reaches TPR 0.634 at FPR 0.1% (0.5595 without overlap), yet its phishing
  scores below 0.74956. Point estimates; at FPR 0.1% a set allows at most one false
  positive among 1,429 clean URLs (full table: ADR 0003).

**What the external normalisation misses:** it compares whole URLs, so the same page with
other query parameters, fragments or encoding, and sibling pages on one host or
platform, pass; external still shares 24 URLs with training at hostname level or
stricter and 269 at platform level or stricter [results/overlap.json].

**Supported:** the model ranks phishing above clean on both sources (AUC >= 0.91 in the
table, lower 95% bounds >= 0.89); the fixed threshold keeps its false-positive rate on a
new source but not its recall (external 0.2127 vs OOF 0.4373, intervals disjoint); at
1:1000 most alerts would be false on OOF and test without overlap (upper bounds 0.209 and
0.315). **Not supported:** that test numbers predict future traffic; any deployed
precision on real traffic, since every estimate assumes the curated clean URLs resemble
real clean traffic; any upper bound for external, which had no false positives.

## 2. URL observations

Patterns of these feeds, not of phishing in general [results/eda.json unless noted].
- **[OBS]** `www.` is in 0.6608 of clean rows (2,813) vs 0.203 of phishing (826).
  **[OBS]** The 12 most clean-leaning n-grams of the model are all fragments of
  `://www.` [results/baseline_oof.json]. **[HYP] "the model mainly ranks on the scheme/`www.` prefix": weakened on
  training-like data, not on external.** Stripping both costs little AUC on OOF (0.9346
  vs 0.9421) [results/oof_headline.json] and test (0.9645 vs 0.9714), a lot on external
  (0.8053 vs 0.9127) [results/evaluation.json].
- **[OBS]** Phishing clusters on shared hosting platforms: on the 650 rows whose tenant-
  and platform-level keys differ (538 phishing), AUC is 0.9 when other tenants of the
  platform may be in training and 0.6092 when not [results/controls.json]. All 32 `.app`
  rows are phishing, from only 5 platform groups.
- **[HYP] contradicted: "HTTPS means legitimate".** 0.3326 of phishing rows (1,353) use
  https vs 0.4402 of clean rows (1,874).
- **[OBS] OOF errors** [results/error_analysis_oof.json]: missed phishing looks like
  ordinary sites (`www.` 0.2905 vs 0.0905 for caught phishing); 26 of 42 false positives
  are blogspot.com. **[HYP]** Many misses are compromised or look-alike sites with no
  URL-level signal.

## 3. Threshold and deployment

**Threshold:** 0.74956, the OOF FPR <= 1% point, fixed before test scoring; not 0.1%,
because that OOF point rests on 4 false positives. Across fold seeds 0-4 the rule gives
recall 0.4373-0.5602 and thresholds 0.6825-0.75171 [results/group_ablation.json]; no
interval includes this split variance. Test FPR 0.0035 and external 0 lie inside the
per-fold OOF range 0-0.0317 [results/oof_headline.json], so it was not re-tuned.

**Deployed precision** = TPR·π / (TPR·π + FPR·(1-π)), π = 0.001, from measured rates,
not from `probability`. FPR dominates and is measured worst: with 1,429 clean URLs per
set, one false positive moves FPR by 1/1,429. Estimates: 0.0425 (OOF), 0.098 (test
without overlap), >= 0.058 (external); the exact cross-check for that test row, ignoring
grouping, is 0.0333-0.3524 [results/evaluation.json]. **The sample size supports** "most alerts are false" on OOF
and test, but cannot distinguish 0.04 from 0.4 or bound external from above.
**Assumptions:** prevalence 1:1000; measured rates hold in deployment, especially that
the curated 1,429 clean URLs per set resemble real clean traffic (weakest); the
threshold carries over to the refit model; rows (exact) or platform groups (bootstrap)
are independent.

**`probability`** is the score for the cleaned training population (4,068 of 8,325
phishing), not the chance of phishing at 1:1000, and is under-confident even there: OOF
rows scored 0.7-0.8 are 0.9265 phishing, rows scored 0.1-0.2 are 0.0558
[results/roc_calibration_oof.json]. It only ranks; no calibration was added.

**Recommendation: prioritise review; do not block or warn users.** About 4 in 100 alerts
would be phishing (OOF, 2-21 in 100), and one phishing URL in five from an unfamiliar
source is caught (0.12-0.32); a
stricter FPR <= 0.1% threshold stays a scenario (ADR 0003). **Evidence needed before
blocking:** labelled deployment traffic with enough clean URLs to bound FPR near 0.1%
and set that source's threshold; false positives by hosting platform; a time-based
split; recall on phishing from sources outside the training feeds.

## 4. Data handling

| Issue (raw 8,651 rows) | Rule | Rows dropped |
|---|---|---|
| 50 normalised keys with both labels | drop all | 107 (57 phish / 50 clean) |
| 219 normalised same-label copies | keep first in file | 219 (205 / 14) |
| 2 internal whitespace, 5 non-ASCII; all parse | keep | 0 |

- **Result:** 8,325 rows, 4,068 phishing [results/cleaning.json]. Dropping conflicts
  makes the data easier than real traffic; majority vote or "flagged once means
  phishing" are the alternatives (ADR 0001).
- **No additional data** was used, so no exclusion check was needed.
- **Invalid inputs** (ADR 0005): a non-string or blank input returns `clean` with
  probability 0.0, meaning "not scored"; strings are cut at 4,000 characters. No
  evaluation URL is affected [results/overlap.json]; deployment should queue unscored
  inputs separately.

## 5. Decision log (observation -> action -> effect)

1. Conflicts and copies -> drop/dedupe (ADR 0001) -> 8,325 rows; keeping the first, last
   or a random copy gives AUC 0.9421 / 0.9422 / 0.9417 [results/controls.json].
2. Tenant-level folds leaked platforms (AUC 0.9 vs 0.6092) -> platform-level groups
   (ADR 0002) -> recall at 1% FPR drops from 0.5951 to 0.4373 [results/controls.json].
3. `www.` gap -> stripped-input probe -> OOF AUC 0.9346 vs 0.9421; a diagnostic by
   design, never a candidate (ADR 0003); on external it exposes prefix dependence.
4. Only 4 false positives at the OOF 0.1% FPR point -> operate at 1% FPR (ADR 0003) ->
   test FPR stayed inside the OOF fold range.
5. Unsuccessful: duilawyeryork.com has 257 phishing rows -> excluded from training,
   paired over 5 fold seeds -> recall difference -0.0052 to 0.0111, all CIs include 0;
   no cap (ADR 0004) [results/group_ablation.json]. Removing its 91 test rows from
   scoring moves test TPR from 0.6123 to 0.5859 [results/evaluation.json].

## 6. Limitations and next steps

- **Configurations:** 1 hyperparameter configuration; 3 variants compared on OOF, none
  on test labels [results/baseline_oof.json, results/group_ablation.json]. Logistic
  regression was chosen, not compared with other models: the brief asks for a simple,
  explainable model, a regularised linear model suits sparse n-gram features, its
  per-n-gram weights can be read, and it outputs the required probability. The baseline
  was kept because the only candidate improvement (decision 5) had every CI including
  0; the probe was never a candidate.
- **Intervals** ignore split variance; exact ones also ignore grouping and the 24
  within-test duplicates [results/overlap.json]; the low-FPR ROC points have none.
- **Unfinished:** per-group cap (ADR 0004), effect of cleaning on metrics (ADR 0001),
  time-based split, host or reputation features, reproduction on a second OS.
- **Next:** labelled deployment traffic to measure FPR at the 0.1% scale; a model less
  dependent on URL-prefix conventions, judged on external-style data; calibration to
  deployment prevalence once such data exists.
