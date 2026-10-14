"""The single featurize path shared by src.train and predict.py.

All functions are module level so a pickled pipeline references src.features,
never __main__.
"""

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

SEED = 0
MAX_URL_CHARS = 4000  # guards n-gram cost on pathological input; longest train URL is 1641
_SCHEME_AND_WWW = re.compile(r"^(?:[a-z][a-z0-9+.\-]*://)?(?:www\.)?", re.IGNORECASE)


def prepare_url(raw: object) -> str:
    """Coerce any input to the text the model sees: str, stripped, capped."""
    text = "" if raw is None else str(raw)
    return text.strip()[:MAX_URL_CHARS]


def prepare_url_without_scheme_and_www(raw: object) -> str:
    """prepare_url, then drop a leading scheme and a leading 'www.' (ablation)."""
    return _SCHEME_AND_WWW.sub("", prepare_url(raw), count=1)


def build_pipeline(strip_scheme_and_www: bool = False) -> Pipeline:
    """Char 3-5-gram TF-IDF + logistic regression; fit on training folds only.

    lowercase=False: URL paths are case-sensitive and case patterns carry signal.
    """
    preprocessor = prepare_url_without_scheme_and_www if strip_scheme_and_www else prepare_url
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        sublinear_tf=True,
        lowercase=False,
        preprocessor=preprocessor,
    )
    classifier = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])
