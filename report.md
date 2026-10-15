# URL phishing classifier: report

**Summary.** A char n-gram logistic regression on the raw URL, thresholded at 0.74956
(OOF FPR <= 1%), catches 61% of phishing on the overlapping test set but 21% on the
separate external collection, with FPR at or below 0.35% on both. External is ranked
well at low FPR; it is the threshold that does not transfer between sources. At 1
phishing URL in 1,000, about 4% of alerts would be phishing at the OOF operating point
(0.0425, 95% 0.018-0.209); test gives 0.149, and external, with 0 false positives, only
bounds it from below (>= 0.058). I therefore recommend using it to prioritise review,
not to block, until it is validated on labelled deployment traffic.

**OOF** = out-of-fold predictions on cleaned `train.csv`: every training row scored by
a model that did not see its platform group (5-fold grouped CV, ADR 0002). Test and
external numbers come from one evaluation at thresholds fixed beforehand (ADR 0003); two
later logged runs only added the sample-precision column and the low-FPR ROC
diagnostic; every other value is identical (`results/evaluation_runs.jsonl`). Brackets name the file each number comes from.

## 1. Result for the manager

**Model:** the raw URL string is split into overlapping 3-5 character pieces
(`pay`, `ypal`, `.tk/`), case kept; each piece is weighted by TF-IDF, and a logistic
regression (C=1) learns one weight per piece. A URL's score is the weighted sum of its pieces'
TF-IDF values, passed through a sigmoid. One configuration was used. The threshold is 0.74956,
chosen on OOF for FPR <= 1%. I stand behind the **external set** and the **test
rows with no training overlap**, not the test headline [results/evaluation.json]:

| evaluation | AUC | TP / FP | recall (TPR) | FPR | precision on sample | precision at 1:1000, 95% interval |
|---|---|---|---|---|---|---|
| OOF (validation) | 0.9421 | 1,779 / 42 | 0.4373 | 0.00987 | not computed | 0.0425 (0.018-0.209) |
| test, all 2,858 | 0.9714 | 875 / 5 | 0.6123 | 0.0035 | 0.9943 | 0.149 (0.080-0.468) |
| test, no overlap (1,508) | 0.9472 | 249 / 4 | 0.4519 | 0.00418 | 0.9842 | 0.098 (0.049-0.315) |
| external, all 2,858 | 0.9127 | 304 / 0 | 0.2127 | 0 (<= 0.0031) | 1.0 | >= 0.058 (exact bound) |

OOF row: `results/oof_headline.json`. "Precision on sample" is TP / (TP + FP) on the
evaluation sets' own roughly 1:1 class mix; it says nothing about deployment, where
phishing is far rarer. OOF and test intervals are the platform-group bootstrap at the
fixed threshold, the headline interval chosen before the run (ADR 0003); with 0
false positives that collapses, so external uses the exact Clopper-Pearson bound
(joint 95%, alpha 0.025 per rate).

**Supported:** the model ranks phishing above clean on both sources (AUC >= 0.91),
including at low FPR (table below); the fixed threshold does not transfer (about one
phishing URL in five caught on the separate collection); at 1:1000 most alerts would
be false on OOF and test (upper bounds 0.209 and 0.468); external had no false
positives, so its precision has no informative upper bound. **Not supported:** that
test numbers predict future traffic, or that deployed precision on unseen sources
would exceed the OOF estimate (0.0425).

**Overlap definition** (`src/overlap.py`). Each evaluation URL gets the strictest of
seven levels at which it matches any cleaned training URL; each level implies the
looser ones:
1. exact string;
2. normalised with path case kept;
3. normalised by the external-set rule (lower-case, no scheme, no leading `www.`, no
   trailing slash);
4. hostname plus first path segment (near-duplicate pages);
5. hostname (no `www.`, userinfo or port);
6. tenant group (eTLD+1, hosting-platform suffixes count, so each tenant is separate);
7. platform group (eTLD+1, ICANN suffixes only, the CV grouping key, ADR 0002).

Metrics are split into four bins: same page (1-3), same site (4-5), same group (6-7)
and none. Looser levels matter because a shared host or platform lets the model
recognise a site rather than judge a URL (section 2).

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
  **Observation:** the ranking transfers but the score scale does not. On its own ROC
  curve, external reaches TPR 0.634 at FPR 0.1% and partial AUC 0.827, higher than test
  (0.2302, 0.7927), so its phishing URLs are ranked well but score below 0.74956.
  **Hypothesis:** the shift comes from source-specific conventions such as scheme and
  `www.`: without them the probe's external partial AUC falls to 0.6172. Not tested
  further.

**Low-FPR ROC (diagnostic, each set's own ROC curve; no threshold follows from it):**

| TPR at FPR | 0.1% | 0.5% | 1% | 5% | partial AUC (FPR <= 1%) |
|---|---|---|---|---|---|
| OOF | 0.2085 | 0.3545 | 0.4373 | 0.7141 | 0.6674 |
| test | 0.2302 | 0.6697 | 0.7341 | 0.8572 | 0.7927 |
| test, no overlap | 0.0871 | 0.5064 | 0.5608 | 0.7314 | 0.7194 |
| external | 0.634 | 0.6648 | 0.6704 | 0.7502 | 0.827 |
| external, no overlap | 0.5595 | 0.5952 | 0.602 | 0.6973 | 0.7913 |

OOF row: `results/roc_calibration_oof.json`; others `results/evaluation.json`. Point
estimates without intervals; at FPR 0.1% each evaluation set allows at most one false
positive among 1,429 clean URLs. Partial AUC is standardised (0.5 = random, 1 = perfect).
- **What the external normalisation misses:** it compares whole URLs, so the same page
  with other query parameters, fragments or encoding, and sibling pages on one host or
  platform, pass. External still shares 24 URLs at hostname level or stricter and 269
  at platform level with training [results/overlap.json].
- **Duplication across files** (URLs only): no normalised URL is in all three files.
  Against the raw training file, 146 test and 0 external URLs repeat a page after
  normalisation, and 0 exactly. Within test, 24 rows repeat another test row after
  normalisation (external: 0) [results/overlap.json].

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
- **[OBS] Errors at the operating threshold (OOF)** [results/error_analysis_oof.json].
  Missed phishing (2,289) looks like ordinary sites: `www.` 0.2905 vs 0.0905 for caught
  phishing, bare host 0.2473 vs 0.104, median length 46 vs 69. Phishing on trusted
  platforms is often missed (blogspot.com 108, appspot.com 93, google.com 83,
  sharepoint.com 48). Of 42 false positives, 26 are blogspot.com and 8 are legitimate
  microsoftonline.com/apple.com login URLs; none has `www.`. **[HYP]** Many misses are
  compromised or look-alike sites with no URL-level signal; this is a limit of URL-only
  input, not of tuning.

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
[results/evaluation.json]. For test this exact range ignores grouping and is a
cross-check; section 1 gives the headline bootstrap interval, 0.080-0.468 (ADR 0003).
The exact lower bounds on both sets are about 0.06, but the
OOF point estimate, at the FPR the threshold was chosen for, is 0.0425
[results/oof_headline.json]; the samples cannot distinguish 0.04 from 0.4.

**Assumptions behind these figures:**
- 1 in 1,000 incoming URLs is phishing.
- TPR and FPR measured on these samples hold for deployment traffic. In particular, the
  1,429 clean URLs per set look like real clean traffic; they are a curated, balanced
  sample, so this is the weakest assumption.
- The threshold transfers unchanged from OOF to the refit model (checked above).
- Exact intervals assume independent rows; the bootstrap assumes independent platform
  groups.

**`probability`** is the logistic-regression score for the cleaned training population
(4,068 of 8,325 phishing, about 1:1). It is not the chance that a URL is phishing at
1:1000: even if it were calibrated for the training mix, the same score would mean a
much lower chance at that prevalence. Even on the training mix it is under-confident:
OOF rows scored 0.7-0.8 (mean 0.7549) are 0.9265 phishing, rows scored 0.1-0.2 (mean
0.1453) are 0.0558 phishing; Brier score 0.1079 vs 0.2499 for a constant base-rate
guess [results/roc_calibration_oof.json]. It is used only to rank and to apply the
threshold; no calibration step was added (ADR 0003).

**Recommendation: prioritise review; do not block or warn users.** On OOF and test,
even the upper estimates leave most alerts false; external, with 0 false positives, is
bounded only from below. At the OOF point estimate about 4 in 100 alerts are phishing,
and only about one phishing URL in five from an unfamiliar source is caught.
The stricter 0.84127 point stays a scenario: test precision 0.225 (bootstrap
0.093-1.0) from 2 false positives, external TPR 0.0903 [results/evaluation.json].
**Evidence needed before blocking:** labelled deployment traffic with enough clean URLs
to bound FPR near 0.1% and to set the threshold for that source (the external ROC shows
the threshold, not the ranking, fails to transfer), false positives by hosting
platform, a time-based split, and recall on phishing from sources outside the training
feeds.

## 4. Data handling

| Issue (raw 8,651 rows) | Rule | Rows dropped |
|---|---|---|
| 50 normalised keys with both labels | drop all | 107 (57 phish / 50 clean) |
| 219 normalised same-label copies | keep first in file | 219 (205 / 14) |
| 2 internal whitespace, 5 non-ASCII; all parse | keep | 0 |

- **Result:** 8,325 rows, 4,068 phishing [results/cleaning.json]. Dropping conflicts
  makes the data easier than real traffic.
- **Alternatives:** majority vote or "flagged once means phishing" for conflicts (both
  keep labels the data shows to be unreliable); keeping the last or a random copy
  instead of the first (AUC 0.9422 / 0.9417 vs 0.9421, so it does not matter)
  [results/controls.json] (ADR 0001).
- **No additional data** was used, so no exclusion check against the evaluation sets
  was needed.
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
  positives, and neither includes split variance; the low-FPR ROC points have no intervals.
  The 24 within-test duplicates make the exact intervals slightly narrow (the bootstrap
  resamples them together); whether duplicates share a label was not checked, as that
  needs test labels.
  `probability` is under-confident and not calibrated (section 3).
- **Unfinished:** per-group cap (ADR 0004), effect of cleaning on metrics (ADR 0001),
  time-based split, host or reputation features, reproduction on a second OS.
- **Next:** labelled deployment traffic to measure FPR at the 0.1% scale; a model less
  dependent on URL-prefix conventions, judged on external-style data; calibration to
  deployment prevalence once such data exists.
