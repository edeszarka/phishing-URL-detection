import pandas as pd

from predict import predict
from src.write_predictions import predictions_frame


def test_predictions_frame_keeps_order_and_matches_predict():
    urls = pd.Series(["http://b.com/login", "a.org", "", "http://192.168.0.1/x"])
    frame = predictions_frame(urls)
    assert list(frame.columns) == ["url", "probability"]
    assert frame["url"].equals(urls)
    assert frame["probability"].tolist() == [predict(u)["probability"] for u in urls]
