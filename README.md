# phishing-URL-detection

URL-only phishing classifier: char 3-5-gram TF-IDF + logistic regression, trained on
cleaned `train.csv` and evaluated once on `test.csv` and `external_test.csv`.
See [report.md](report.md) for results and [docs/adr/](docs/adr/) for decisions.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python is pinned to 3.12
(`.python-version`, `requires-python = ">=3.12,<3.13"`); `uv.lock` pins every
dependency. Runs on CPU.

```bash
uv sync
```

Pinned versions used for the committed model (also in `results/final_model.json`):
Python 3.12.13, scikit-learn 1.9.1, numpy 2.5.4, scipy 1.18.1, pandas 3.0.6,
joblib 1.6.0, tldextract 5.4.0 (bundled suffix list, no network).

**The supplied data is not included.** Put `train.csv`, `test.csv` and
`external_test.csv` in `data/` (ignored by git). Scripts regenerate any derived
file (cleaned data, OOF predictions, overlap levels) in `data/`.

## Inference

The trained model is committed (`models/model.joblib`), so inference needs no data
and no training:

```python
from predict import predict

predict("https://www.PayPal-secure.login.tk/webscr?cmd=x")
# {"label": "clean" | "phishing", "probability": <float in [0, 1]>}
```

- Threshold 0.74956 (OOF FPR <= 1%, ADR 0003); `probability >= threshold` means
  `"phishing"`.
- `probability` is the model score for the cleaned training population (48.9%
  phishing), not calibrated to deployment prevalence.
- Invalid inputs (ADR 0005): a non-`str` input or a blank string is **not scored**
  and returns `{"label": "clean", "probability": 0.0}`. Here 0.0 means "not scored",
  not "safe". Strings longer than 4,000 characters are cut. `predict` never raises.
  If library versions differ from those the model was saved with, a warning is
  printed.
- `predict.score_urls(list_of_urls)` scores a batch through the same path.

## Reproduce

Run from the repo root, in this order. Each script writes its numbers to `results/`.

```bash
# data inspection and cleaning (ADR 0001, 0002)
uv run python -m src.groups              # results/groups_raw.json
uv run python -m src.clean               # results/cleaning.json
uv run python -m src.inspect_data        # results/eda.json

# grouped out-of-fold evaluation and experiments (ADR 0002-0004)
uv run python -m src.train               # results/baseline_oof.json, data/oof_predictions.csv
uv run python -m src.oof_metrics         # results/oof_headline.json
uv run python -m src.controls            # results/controls.json
uv run python -m src.group_ablation      # results/group_ablation.json

# final model (overwrites models/*.joblib, results/final_model.json)
uv run python -m src.train_final

# evaluation
uv run python -m src.overlap             # results/overlap.json (evaluation URLs only)
uv run python -m src.evaluate --dry-run  # must report MATCH against oof_headline.json
uv run python -m src.evaluate --rerun-reason "reproduction"   # results/evaluation.json
uv run python -m src.write_predictions   # predictions.csv

# checks
uv run ruff check . && uv run pytest
```

- `src.evaluate` is the only script that reads evaluation labels. It refuses to
  overwrite `results/evaluation.json` without `--rerun-reason`. Every run is
  appended to `results/evaluation_runs.jsonl`; the committed result is the single
  first run.
- Seeds are fixed (0). On the development machine (Windows 11), rerunning
  `src.train_final` reproduces the committed model files byte for byte. Their
  SHA-256 hashes are in `results/final_model.json`. This has not been checked
  on another OS.

## Layout

| path | content |
|---|---|
| `predict.py` | single-URL and batch inference |
| `src/` | cleaning, grouping, training, evaluation scripts |
| `models/` | committed model (`model.joblib`) and shortcut probe (`probe.joblib`) |
| `results/` | every reported number, written by scripts |
| `predictions.csv` | `url,probability` for each row of `data/test.csv` |
| `docs/adr/` | decision records 0001-0005 |
| `tests/` | pytest suite (no supplied data needed) |
