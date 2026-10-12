# 0001: Clean training data by normalised URL

## Status
Accepted

## Context
`data/train.csv` merges several feeds (8,651 rows; 4,330 phishing, 4,321 clean).
Counts from `results/cleaning.json` (`raw_quality`):

- 0 blank URLs, 0 labels outside {0, 1}, 0 URLs without a registered domain or IP.
- 2 rows contain internal whitespace and 5 contain non-ASCII characters. All parse.
- Exact URL string: 30 redundant same-label copies; 50 URLs carry both labels (100 rows).
- Normalised URL (the rule used to build `external_test.csv`: lower-case, strip
  scheme, leading `www.` and trailing slashes): 219 redundant same-label copies;
  50 keys carry both labels (107 rows). Most extra matches differ only by a
  trailing slash or `http`/`https`.
- Conflicts are mostly bare root domains (e.g. a site listed as both clean and
  phishing), which may be compromised sites or feeds observed at different times.
  The data cannot tell which.

## Decision
Using the normalised URL only as a matching key, in this order:

1. Drop every row whose key carries both labels: 107 rows (50 keys).
2. Among the remaining rows, keep the first row per key in file order: 219 rows dropped.

Cleaned rows keep their original URL string, case included. Malformed-looking
rows are kept unchanged: they parse and are inputs the predictor must handle.
No cleaned file is written; `src.clean.load_clean_train()` regenerates it.

## Alternatives
- Exact-string dedupe only (30 copies): keeps trailing-slash and scheme variants
  of the same page as separate rows.
- Normalise but keep path case: drops 217 instead of 219 rows, so only 2 rows
  merge purely by path case. Not worth diverging from the external-set rule.
- Majority vote or "flagged once means phishing" for conflicts: both keep labels
  the data itself shows to be unreliable.
- Last-in-file or a random representative instead of the first: tested, see
  Consequences.

## Consequences
- 8,651 → 8,325 rows; 4,068 phishing and 4,257 clean. Rule 1 dropped 57
  phishing / 50 clean rows; rule 2 dropped 205 phishing / 14 clean rows, so
  normalized duplicates sit almost entirely in the phishing feeds.
- Dropping conflicts removes exactly the ambiguous cases (e.g. compromised
  legitimate sites). The cleaned data is therefore easier than real traffic, and
  validation metrics on it are optimistic for that slice.
- Grouped CV already puts duplicates in the same fold, so dedupe changes row
  weighting, not leakage. After cleaning the largest platform-level group
  (ADR 0002) is blogspot.com with 3.40% of rows, under the 10% threshold for
  revisiting the group key.
- The file order shows no structure (label runs test z = 0.89; per-decile
  phishing share 47.9-52.8%; `results/eda.json`), so "first in file" is not
  tied to one feed.
- Representative choice (`results/controls.json`, platform-level folds held
  fixed per group): keeping the last or a random copy changes 166 and 94 kept
  URL strings, OOF AUC 0.9421 / 0.9422 / 0.9417 and recall at FPR <= 1%
  0.4373 / 0.4373 / 0.4334 for first / last / random; the bare-host
  no-slash share of phishing stays 29.2% / 29.1% / 29.3%. The choice does not
  matter materially.
- Effect of cleaning itself (cleaned vs raw) on OOF metrics: not measured.
