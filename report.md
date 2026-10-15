# URL phishing classifier: report

**Summary.** A char 3-5-gram TF-IDF + logistic regression on the raw URL (C=1, one
configuration), thresholded at 0.74956 (OOF FPR <= 1%), catches 61% of phishing on the
overlapping test set but 21% on the separate external collection, with FPR at or below
0.35% on both. At 1 phishing URL in 1,000, about 4% of alerts would be phishing, so I
recommend prioritising review, not blocking.

**OOF** = out-of-fold predictions on cleaned `train.csv`, folds grouped by platform
(ADR 0002). Test and external were scored once at thresholds fixed beforehand (ADR 0003);
two logged reruns only added fields (`results/evaluation_runs.jsonl`, git history of
`results/evaluation.json`). Brackets name each number's source file.

## 1. Result for the manager

**Metric:** recall at the fixed threshold with its measured FPR, and precision at 1:1000
derived from both. **Evaluations I stand behind:** OOF, external and the test rows with
no training overlap, not the test headline [results/oof_headline.json,
results/evaluation.json]:

| evaluation | AUC | TP / FP | recall (TPR) | FPR | precision on sample | precision at 1:1000, 95% interval |
|---|---|---|---|---|---|---|
| OOF (validation) | 0.9421 | 1,779 / 42 | 0.4373 | 0.00987 | not computed | 0.0425 (0.018-0.209) |
| test, all 2,858 | 0.9714 | 875 / 5 | 0.6123 | 0.0035 | 0.9943 | 0.149 (0.080-0.468) |
| test, no overlap (1,508) | 0.9472 | 249 / 4 | 0.4519 | 0.00418 | 0.9842 | 0.098 (0.049-0.315) |
| external, all 2,858 | 0.9127 | 304 / 0 | 0.2127 | 0 (<= 0.0031) | 1.0 | >= 0.058 (exact bound) |

Intervals: platform-group bootstrap at the fixed threshold, chosen as the headline
before the run (ADR 0003). It collapses at 0 false positives, so external uses the exact
Clopper-Pearson bound (joint 95%). Precision on sample reflects the sets' 1:1 class mix,
not deployment.

**Overlap** (`src/overlap.py`, ADR 0003): each evaluation URL gets the strictest of seven
levels at which it matches cleaned training, from exact string to shared platform group,
binned as same page, same site, same group or none [results/overlap.json].
- **OOF:** platforms unseen in training; the only data that chose the threshold.
- **Test: same feeds as training, and overlapping.** 1,350 of 2,858 URLs share at least
  a platform group with training; 145 are the same page after the external-set
  normalisation (0 exact). Same-site rows (813) reach TPR 0.7734 with 0 false positives,
  no-overlap rows 0.4519, close to OOF. **[OBS]** Overlap makes the test headline optimistic.
- **External: a different source.** 2,589 of 2,858 URLs match training at no level, yet
  no-overlap TPR is 0.1224 against 0.4519 on test, so overlap does not explain the gap.
  **[OBS]** The ranking transfers but the score scale does not: on its own ROC curve
  external reaches TPR 0.634 at FPR 0.1% (0.5595 on no-overlap rows) and partial AUC
  0.827 vs 0.7927 for test, yet its phishing scores below 0.74956. These are point
  estimates; at FPR 0.1% a set allows at most one false positive among 1,429 clean URLs
  (full table: ADR 0003).

**What the external normalisation misses:** it compares whole URLs, so the same page with
other query parameters, fragments or encoding, and sibling pages on one host or
platform, pass. External still shares 24 URLs with training at hostname level or
stricter and 269 at platform level [results/overlap.json].

**Supported:** the model ranks phishing above clean on both sources (AUC >= 0.91); the
fixed threshold does not transfer; at 1:1000 most alerts would be false on OOF and test
(upper bounds 0.209 and 0.468). **Not supported:** that test numbers predict future
traffic; deployed precision on unseen sources above the OOF estimate (0.0425); any upper
bound for external, which had no false positives.

## 2. URL observations

Patterns of these feeds, not of phishing in general [results/eda.json unless noted].
- **[OBS]** `www.` appears in 0.6608 of clean rows (2,813) vs 0.203 of phishing (826).
  **[HYP] "the model mainly ranks on the scheme/`www.` prefix": weakened on
  training-like data, not on external.** Stripping both costs little AUC on OOF (0.9346
  vs 0.9421) [results/oof_headline.json] and test (0.9645 vs 0.9714), but a lot on
  external (0.8053 vs 0.9127) [results/evaluation.json].
- **[OBS]** Phishing clusters on shared hosting platforms. On the 650 rows where tenant-
  and platform-level keys differ (538 phishing), AUC is 0.9 if other tenants of the
  platform may be in training and 0.6092 if not [results/controls.json]. All 32 `.app`
  rows are phishing, from only 5 platform groups.
- **[HYP] contradicted: "HTTPS means legitimate".** 0.3326 of phishing rows (1,353) use
  https vs 0.4402 of clean rows (1,874).
- **[OBS] OOF errors** [results/error_analysis_oof.json]: missed phishing (2,289) looks
  like ordinary sites (`www.` 0.2905 vs 0.0905 for caught phishing, median length 46 vs
  69), often on trusted platforms (blogspot.com 108, appspot.com 93, google.com 83). Of
  42 false positives, 26 are blogspot.com; none has `www.`. **[HYP]** Many misses are
  compromised or look-alike sites with no URL-level signal.

## 3. Threshold and deployment

**Threshold:** 0.74956, the OOF point with FPR <= 1%, fixed before test scoring. Not
0.1%: that OOF point rests on 4 false positives. Over fold seeds 0-4 the rule gives
recall 0.4373-0.5602 and thresholds 0.6825-0.75171 [results/group_ablation.json]; no
interval includes this split variance. Test FPR 0.0035 and external 0 lie inside the
per-fold OOF range 0-0.0317 [results/oof_headline.json]; the threshold was not re-tuned.

**Deployed precision** = TPR·π / (TPR·π + FPR·(1-π)), π = 0.001, from measured rates,
not from `probability`. FPR dominates and is measured worst: each evaluation set has
1,429 clean URLs, so one false positive moves FPR by 1/1,429. Estimates: 0.0425 (OOF),
0.149 (test), >= 0.058 (external); the exact cross-check for test, ignoring grouping, is
0.0615-0.4037 [results/evaluation.json]. **The sample size allows** "most alerts are
false" on OOF and test, but cannot distinguish 0.04 from 0.4 or bound external from above.

**Assumptions:** prevalence 1:1000; measured TPR and FPR hold for deployment, in
particular the curated 1,429 clean URLs per set resemble real clean traffic (weakest);
the threshold carries over to the model refit on all rows; exact intervals assume
independent rows, the bootstrap independent platform groups.

**`probability`** is the score for the cleaned training population (4,068 of 8,325
phishing), not the chance of phishing at 1:1000. Even on that mix it is
under-confident: OOF rows scored 0.7-0.8 are 0.9265 phishing, rows scored 0.1-0.2 are
0.0558 [results/roc_calibration_oof.json]. It only ranks; no calibration was added.

**Recommendation: prioritise review; do not block or warn users.** About 4 in 100 alerts
would be phishing, and one phishing URL in five from an unfamiliar source is caught. A
stricter 0.84127 threshold stays a scenario: test precision 0.225 (0.093-1.0) from 2
false positives, external TPR 0.0903 [results/evaluation.json]. **Evidence needed before
blocking:** labelled deployment traffic with enough clean URLs to bound FPR near 0.1% and
set the threshold for that source; false positives by hosting platform; a time-based
split; recall on phishing from sources outside the training feeds.

## 4. Data handling

| Issue (raw 8,651 rows) | Rule | Rows dropped |
|---|---|---|
| 50 normalised keys with both labels | drop all | 107 (57 phish / 50 clean) |
| 219 normalised same-label copies | keep first in file | 219 (205 / 14) |
| 2 internal whitespace, 5 non-ASCII; all parse | keep | 0 |

- **Result:** 8,325 rows, 4,068 phishing [results/cleaning.json]. Dropping conflicts
  makes the data easier than real traffic; majority vote and "flagged once means
  phishing" are the alternatives (ADR 0001).
- **No additional data** was used, so no exclusion check was needed.
- **Invalid inputs** (ADR 0005): a non-string or blank input returns `clean` with
  probability 0.0, meaning "not scored"; strings are cut at 4,000 characters. No
  evaluation URL is affected [results/overlap.json]. Deployment should route unscored
  inputs to their own queue.

## 5. Decision log (observation -> action -> effect)

1. Conflicts and copies -> drop/dedupe (ADR 0001) -> 8,325 rows; keeping the first,
   last or a random copy gives AUC 0.9421 / 0.9422 / 0.9417 [results/controls.json].
2. Tenant-level folds leaked platforms (AUC 0.9 vs 0.6092) -> platform-level groups
   (ADR 0002) -> recall at 1% FPR drops from 0.5951 to 0.4373 [results/controls.json].
3. `www.` gap -> stripped-input probe -> OOF AUC 0.9346 vs 0.9421; a diagnostic by
   design, never a candidate (ADR 0003); on external it exposes prefix dependence.
4. The 0.1% FPR point rests on 4 false positives -> operate at 1% FPR (ADR 0003) ->
   test FPR stayed inside the OOF fold range.
5. Unsuccessful: duilawyeryork.com has 257 phishing rows -> excluded from training,
   paired over 5 fold seeds -> recall difference -0.0052 to 0.0111, all CIs include 0;
   no cap (ADR 0004) [results/group_ablation.json]. Removing its 91 test rows from
   scoring moves test TPR from 0.6123 to 0.5859 [results/evaluation.json].

## 6. Limitations and next steps

- **Configurations:** 1 hyperparameter configuration; 3 variants compared on OOF, none
  on test labels [results/baseline_oof.json, results/group_ablation.json]. The baseline
  was kept because the only candidate improvement (decision 5) had every CI including
  0; the probe was never a candidate.
- **Intervals** ignore split variance; exact ones also ignore grouping and the 24
  within-test duplicates [results/overlap.json]; the low-FPR ROC points have none.
- **Unfinished:** per-group cap (ADR 0004), effect of cleaning on metrics (ADR 0001),
  time-based split, host or reputation features, reproduction on a second OS.
- **Next:** labelled deployment traffic to measure FPR at the 0.1% scale; a model less
  dependent on URL-prefix conventions, judged on external-style data; calibration to
  deployment prevalence once such data exists.
