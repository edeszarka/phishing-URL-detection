"""Write predictions.csv (url,probability), one row per URL in data/test.csv, in file order.

Run as `uv run python -m src.write_predictions`. Reads ONLY the url column of
data/test.csv and scores it with predict.score_urls, the same path as predict().
"""

from pathlib import Path

import pandas as pd

from predict import score_urls
from src.overlap import load_eval_urls

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "predictions.csv"


def predictions_frame(urls: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"url": urls, "probability": score_urls(urls.tolist())})


def main() -> None:
    urls = load_eval_urls("test")
    frame = predictions_frame(urls)
    if len(frame) != len(urls) or not frame["url"].equals(urls):
        raise SystemExit("predictions do not line up one-to-one with data/test.csv")
    frame.to_csv(OUT, index=False)
    print(f"wrote {OUT.name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
