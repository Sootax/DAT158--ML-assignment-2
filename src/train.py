import argparse
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.diagnostics import brand_recall, invariance, latency_ms
from src.evaluate import evaluate, threshold_for_recall
from src.features import build_pipeline, extract_url_features, LEXICAL_FEATURE_NAMES
from src.tracking import log_run


def load(path: str, sample: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["y"] = (df["label"] == "phish").astype(int)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["url", "date"]).drop_duplicates(subset="url")

    if sample:
        parts = [g.sample(min(sample, len(g)), random_state=seed) for _, g in df.groupby("split")]
        df = pd.concat(parts)

    return df


def temporal_validation_split(train: pd.DataFrame, frac: float = 0.2):
    train = train.sort_values("date")
    cut = int(len(train) * (1 - frac))

    return train.iloc[:cut], train.iloc[cut:]


def rule_baseline_scores(urls) -> list[float]:
    idx = {name: i for i, name in enumerate(LEXICAL_FEATURE_NAMES)}
    scores = []

    for url in urls:
        f = extract_url_features(url)
        scores.append(f[idx["host_is_ip"]] * 2 + f[idx["has_at"]] + min(f[idx["n_suspicious_words"]], 3) + (f[idx["url_len"]] > 75) + (f[idx["n_subdomains"]] > 2))

    return scores


def make_classifier(name: str, C: float, seed: int):
    if name == "logreg":
        return LogisticRegression(C=C, max_iter=1000)

    if name == "svc":
        return CalibratedClassifierCV(LinearSVC(C=C, dual="auto"), cv=3)

    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, early_stopping=True, random_state=seed)

    raise ValueError(f"unknown model {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/phreshphish_urls.parquet")
    parser.add_argument("--sample", type=int, default=None, help="rows per split (speed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true", help="also evaluate on test")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--name", default="run", help="short label shown in plots")
    parser.add_argument("--note", default="", help="WHAT you changed and WHY (goes in the report)")
    parser.add_argument("--C", type=float, default=1.0, help="inverse regularisation strength")
    parser.add_argument("--max-char-features", type=int, default=300_000)
    parser.add_argument("--model", choices=["logreg", "svc", "hgb"], default="logreg")
    parser.add_argument("--normalize", action="store_true", help="strip scheme, www., trailing / and case")
    parser.add_argument("--word-tokens", action="store_true", help="add a word-level TF-IDF branch")
    parser.add_argument("--no-chars", action="store_true", help="drop the character n-grams (lexical only)")
    parser.add_argument("--ngram-min", type=int, default=3)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--parent", default=None, help="name of the run this one is compared to")
    args = parser.parse_args()

    if not args.note:
        print("Tip: use --note to record what you changed in this run.\n")

    df = load(args.data, args.sample, args.seed)
    train_all = df[df["split"] == "train"]
    train, val = temporal_validation_split(train_all)

    print(f"train {len(train):,} | val {len(val):,} | phishing share in val {val['y'].mean():.1%}")
    print(f"train dates {train['date'].min().date()}..{train['date'].max().date()}, val dates {val['date'].min().date()}..{val['date'].max().date()}")

    results, arrays = {}, {"y_val": val["y"].to_numpy()}

    dummy = DummyClassifier(strategy="prior").fit(train["url"], train["y"])
    results["majority"] = evaluate(val["y"], dummy.predict_proba(val["url"])[:, 1], 0.5)

    rule = rule_baseline_scores(val["url"])
    results["rule"] = evaluate(val["y"], rule, threshold_for_recall(val["y"], rule, 0.9))
    arrays["rule_val"] = rule

    model = build_pipeline(make_classifier(args.model, args.C, args.seed), max_char_features=args.max_char_features, normalize=args.normalize, word_tokens=args.word_tokens,
                           char_ngrams=not args.no_chars, ngram_range=(args.ngram_min, args.ngram_max))
    start = time.perf_counter()
    model.fit(train["url"], train["y"])
    fit_seconds = time.perf_counter() - start
    val_scores = model.predict_proba(val["url"])[:, 1]
    threshold = threshold_for_recall(val["y"], val_scores, 0.9)
    results["model_val"] = evaluate(val["y"], val_scores, threshold)
    arrays["model_val"] = val_scores

    results["robustness"] = invariance(model, val["url"], threshold)
    results["latency_ms_per_url"] = latency_ms(model, val["url"])
    results["fit_seconds"] = fit_seconds
    results["worst_brands"] = brand_recall(val, val_scores, threshold)

    if args.test:
        test = df[df["split"] == "test"]
        test_scores = model.predict_proba(test["url"])[:, 1]
        results["model_test"] = evaluate(test["y"], test_scores, threshold)
        arrays["y_test"], arrays["model_test"] = test["y"].to_numpy(), test_scores

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    joblib.dump(model, out / "model.joblib")

    run_id = log_run(args.out, args.name, args.note, params={"model": args.model, "C": args.C, "max_char_features": args.max_char_features, "normalize": args.normalize, "word_tokens": args.word_tokens,
        "char_ngrams": not args.no_chars, "ngram_range": [args.ngram_min, args.ngram_max], "parent": args.parent, "sample": args.sample, "seed": args.seed, "target_recall": 0.9},
        results=results,
        data_info={"n_train": len(train), "n_val": len(val), "val_phish_share": float(val["y"].mean()), "train_dates": [str(train["date"].min().date()), str(train["date"].max().date())],
                   "val_dates": [str(val["date"].min().date()), str(val["date"].max().date())]}, arrays=arrays)

    metric_blocks = {k: r for k, r in results.items() if isinstance(r, dict) and "confusion" in r}
    summary = pd.DataFrame({k: {m: v for m, v in r.items() if m != "confusion"} for k, r in metric_blocks.items()}).round(3)

    print("\n", summary.to_string())
    print(f"\nrobustness {results['robustness']}")
    print(f"latency {results['latency_ms_per_url']:.1f} ms/url, fit {fit_seconds:.0f} s")
    print(f"\nLogged run: {run_id}")


if __name__ == "__main__":
    main()
