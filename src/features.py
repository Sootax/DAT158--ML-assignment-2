import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

SUSPICIOUS_WORDS = ("login", "signin", "verify", "secure", "account", "update", "bank", "confirm", "password", "wallet", "support", "billing")

LEXICAL_FEATURE_NAMES = [
    "url_len", "host_len", "path_len", "query_len", "n_dots", "n_hyphens", "n_underscores", "n_digits", "digit_ratio", "n_special", "has_at",
    "n_subdomains", "host_is_ip", "has_port", "is_https", "path_depth", "host_entropy", "n_suspicious_words"
]


def _entropy(text: str) -> float:
    if not text:
        return 0.0

    counts = Counter(text)
    total = len(text)

    return -sum(c / total * math.log2(c / total) for c in counts.values())


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))

        return True

    except ValueError:
        return False


def extract_url_features(url: str) -> list[float]:
    url = (url or "").strip()

    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        host = (parsed.hostname or "").lower()
        port = parsed.port

    except ValueError:
        parsed, host, port = urlparse(""), "", None

    lowered = url.lower()
    labels = host.split(".") if host else []

    n_digits = sum(ch.isdigit() for ch in url)

    return [
        len(url),
        len(host),
        len(parsed.path),
        len(parsed.query),
        url.count("."),
        url.count("-"),
        url.count("_"),
        n_digits,
        n_digits / max(len(url), 1),
        len(re.findall(r"[%=&?@~#]", url)),
        float("@" in url),
        max(len(labels) - 2, 0),
        float(_is_ip(host)),
        float(port is not None),
        float(parsed.scheme == "https"),
        len([p for p in parsed.path.split("/") if p]),
        _entropy(host),
        sum(word in lowered for word in SUSPICIOUS_WORDS)
    ]


def lexical_matrix(urls) -> np.ndarray:
    return np.array([extract_url_features(u) for u in urls], dtype=float)


def normalize_url(url: str) -> str:
    u = (url or "").strip().lower()
    u = re.sub(r"^[a-z][a-z0-9+.-]*://", "", u)
    u = re.sub(r"^www\.", "", u)

    return u.rstrip("/")


def normalize_urls(urls) -> list[str]:
    return [normalize_url(u) for u in urls]


def build_pipeline(classifier=None, max_char_features: int = 300_000, normalize: bool = False, word_tokens: bool = False, char_ngrams: bool = True, ngram_range: tuple[int, int] = (3, 5)) -> Pipeline:
    lexical = Pipeline([
        ("extract", FunctionTransformer(lexical_matrix)),
        ("scale", StandardScaler())
    ])
    branches = [("lexical", lexical)]

    if char_ngrams:
        branches.append(("chars", TfidfVectorizer(analyzer="char", ngram_range=ngram_range, lowercase=True, max_features=max_char_features, sublinear_tf=True)))

    if word_tokens:
        branches.append(("words", TfidfVectorizer(analyzer="word", token_pattern=r"[a-z0-9]+", lowercase=True, min_df=2, max_features=200_000, sublinear_tf=True)))

    steps = []

    if normalize:
        steps.append(("normalize", FunctionTransformer(normalize_urls)))

    steps += [
        ("features", FeatureUnion(branches)),
        ("clf", classifier or LogisticRegression(C=1.0, max_iter=1000))
    ]

    return Pipeline(steps)
