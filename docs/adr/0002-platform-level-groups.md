# 0002: Group CV folds at platform level

## Status
Accepted. Supersedes the tenant-level group key adopted in commit 061f92a
(no prior ADR; ADR 0001 covers cleaning).

## Context
Two group keys, both eTLD+1 via tldextract's bundled suffix list
(`src/groups.py`):

- **tenant-level**: PSL private suffixes count, so `a.netlify.app` and
  `b.netlify.app` are separate groups (5,599 groups after cleaning).
- **platform-level**: only ICANN suffixes count, so all tenants of a hosting
  platform share one group (5,130 groups).

The keys differ on 650 of 8,325 cleaned rows, 538 of them phishing.
Same baseline (char 3-5-gram TF-IDF + logistic regression, raw URL), 5-fold
StratifiedGroupKFold, seed 0 (`results/controls.json`):

| key | OOF AUC | fold AUC SD | recall at FPR <= 1% | AUC on the 650 differing rows |
|---|---|---|---|---|
| tenant-level | 0.9491 | 0.0105 | 0.5951 | 0.900 |
| platform-level | 0.9421 | 0.0236 | 0.4373 | 0.609 |

With tenant-level folds the model sees other tenants of the same platform
during training; on those rows it ranks well only in that case.

## Decision
Use the platform-level key (`src.groups.cv_group_key`) for CV folds, threshold
choice and the bootstrap resampling unit. The tenant-level key stays available
as a control.

## Alternatives
- Keep tenant-level and report the control as a limitation: about 16 points
  higher recall at 1% FPR, which the control shows depends on seeing other
  tenants of the same platform.
- Hostname-level groups: finer than tenant-level, so the leak would be larger.

## Consequences
- Largest platform-level groups after cleaning: blogspot.com 283 rows (3.40%,
  192 phishing), duilawyeryork.com 257 (3.09%), appspot.com 196 (2.35%); the
  10% rule is not triggered (`results/cleaning.json`).
- Headline OOF numbers drop (`results/baseline_oof.json`,
  `results/oof_headline.json`): AUC 0.9491 -> 0.9421, recall at FPR <= 1%
  0.5951 -> 0.4373, and the fold AUC SD roughly doubles.
- The rule-2 representative choice still makes no material difference under
  platform-level folds (AUC 0.9421 / 0.9422 / 0.9417 for first / last / random).
- Estimates are now conservative for platforms present in training; they may be
  pessimistic if deployment traffic contains many tenants of known platforms.
