import time

import numpy as np
import pandas as pd

from src.features import normalize_url


def invariance(model, urls, threshold: float, n: int = 1000, seed: int = 0) -> dict:
    sample = pd.Series(list(urls)).sample(min(n, len(urls)), random_state=seed)
    bare = [normalize_url(u) for u in sample]
    dressed = [f"https://www.{b}/" for b in bare]
    p_bare = model.predict_proba(bare)[:, 1]
    p_dressed = model.predict_proba(dressed)[:, 1]

    return {
        "n": len(bare),
        "mean_abs_prob_change": float(np.mean(np.abs(p_dressed - p_bare))),
        "decision_flip_rate": float(np.mean((p_bare >= threshold) != (p_dressed >= threshold)))
    }


def latency_ms(model, urls, n: int = 200) -> float:
    times = []

    for u in list(urls)[:n]:
        start = time.perf_counter()
        model.predict_proba([u])
        times.append((time.perf_counter() - start) * 1000)

    return float(np.median(times))


def brand_recall(val: pd.DataFrame, scores, threshold: float, min_n: int = 200) -> list[dict]:
    phish = val[val["y"] == 1].assign(caught=lambda d: np.asarray(scores)[val["y"].to_numpy() == 1] >= threshold)
    by = phish.groupby("target")["caught"].agg(n="size", recall="mean")
    by = by[by["n"] >= min_n].sort_values("recall").head(10)

    return [{"brand": b, "n": int(r.n), "recall": float(r.recall)} for b, r in by.iterrows()]
