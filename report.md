# URL phishing classifier: report (DRAFT)

Numbers are out-of-fold (OOF) on cleaned `train.csv`, platform-level grouped
5-fold CV (ADR 0002), unless stated.

## 1. Result for the manager

TODO(me, <=120 words): metric and evaluation I stand behind, claims supported;
test and external recall / FPR / precision at 1:1000 / ROC-AUC at the threshold.

## 2. URL observations

Patterns of these feeds, not of phishing in general.

- **[OBS]** `www.` appears in 2,813 of 4,257 clean rows (0.6608) vs 826 of 4,068
  phishing rows (0.203) [results/eda.json]. **[HYP] weakened:** "the model mainly
  ranks on the scheme/`www.` prefix". Stripping both gives AUC 0.9346 vs 0.9421,
  recall at FPR <= 1% 0.4462 vs 0.4373 [results/oof_headline.json].
- **[OBS]** Phishing clusters on shared hosting platforms: on the 650 rows where
  tenant- and platform-level keys differ (538 phishing), AUC is 0.9 if other
  tenants of the platform may be in training, 0.6092 if not
  [results/controls.json]. All 32 `.app` rows are phishing, from only 5 platform
  groups [results/eda.json].
- **[HYP] contradicted: "HTTPS means legitimate".** 1,353 of 4,068 phishing rows
  use https (0.3326) vs 1,874 of 4,257 clean (0.4402) [results/eda.json].

## 3. Threshold and deployment

TODO(me, <=150 words): threshold (ADR 0003), assumptions, precision at 1:1000
from measured test TPR/FPR with intervals, what the sample size allows,
deployment mode, evidence needed.

## 4. Data handling

| Issue (raw 8,651 rows) | Rule | Rows dropped |
|---|---|---|
| 50 normalised keys with both labels | drop all | 107 (57 phish / 50 clean) |
| 219 normalised same-label copies | keep first in file | 219 (205 / 14) |
| 2 internal whitespace, 5 non-ASCII; all parse | keep | 0 |

Result: 8,325 rows, 4,068 phishing [results/cleaning.json]. Dropping conflicts
makes the data easier than real traffic.

## 5. Decision log (observation -> action -> effect)

1. Conflicts and copies -> drop/dedupe (ADR 0001) -> 8,325 rows; representative choice: AUC 0.9421 / 0.9422 / 0.9417 [results/controls.json].
2. Tenant-level folds leaked platforms (AUC 0.9 vs 0.6092) -> platform-level groups (ADR 0002) -> recall at 1% FPR 0.5951 to 0.4373 [results/controls.json].
3. `www.` gap -> stripped-input variant -> AUC 0.9346 vs 0.9421; not adopted, kept as probe (ADR 0003) [results/oof_headline.json].
4. 0.1% FPR rests on 4 false positives -> operate at 1% FPR, threshold 0.74956, precision at 1:1000 0.0425 (ADR 0003) [results/oof_headline.json].

## 6. Limitations and next steps

TODO(me, <=100 words): limitations, unfinished work, next steps.
