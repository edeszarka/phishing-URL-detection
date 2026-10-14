# 0005: Invalid-input policy for predict()

## Status
Accepted.

## Context
`predict()` must accept any raw string, never crash, and return exactly
`{"label", "probability"}` with a finite probability in [0, 1]. The training data
has no blank URLs, and every row parses (ADR 0001). The longest training URL is 1,641
characters, raw and cleaned. So the model has never seen empty input or very
long input, and whatever it outputs for either is not evidence of anything.

`src.features.prepare_url` used to convert any input with `str()`. That turns
`12345` or `None` into text the model scores as if it were a URL.

## Decision
1. **Not scored**: a non-`str` input (None, numbers, bytes) or a string that is
   empty after `strip()` returns `{"label": "clean", "probability": 0.0}`.
   This is fail-open. The 0.0 means "not scored", not "certainly clean".
2. **Length cap**: every other string is cut to 4,000 characters
   (`MAX_URL_CHARS`, `src/features.py`) and scored by the model. The cap only
   limits char n-gram vectorisation cost on pathological strings. It is not
   tied to the training length distribution. Cutting drops the end of the path
   and the query string, along with any signal they carry.
3. **Malformed but non-empty strings are scored.** Odd hosts, whitespace,
   non-ASCII characters and a missing scheme are all scored, consistent with
   ADR 0001, which kept such rows as inputs the predictor must handle.
4. **Never raises**: an outer try/except, and a check for non-finite
   probabilities, both return the not-scored answer from (1).
5. `score_urls()` (batch) and `predict()` share this one path.

## Alternatives
- Convert everything to `str` and score it: numbers and `None` become fake URLs
  with arbitrary scores.
- Fail-closed (`phishing`, 1.0): floods the review queue that ADR 0003 recommends
  with inputs that are not URLs at all.
- A cap based on the training maximum: ties a safety limit to one sample's
  length distribution and would cut long but valid URLs.

## Consequences
- The two-key contract cannot express "not scored". A deployed system should
  send unscored inputs to their own bucket, not into the clean stream.
- Raising the cap from 2,048 to 4,000 affects no training row (longest 1,641),
  so the frozen model and the ADR 0003 thresholds are unchanged.
- Measured effect: `tests/test_predict.py` covers the contract and edge-case
  inputs. Neither evaluation set has an unscored or truncated URL
  (`results/overlap.json`: 0 and 0 in test and external), so the policy changes no
  reported metric.
