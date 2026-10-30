import argparse
import json
from pathlib import Path

from sklearn.metrics import average_precision_score

from src.evaluate import evaluate, threshold_for_recall
from src.features import build_pipeline
from src.train import load, make_classifier, temporal_validation_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/phreshphish_urls.parquet")
    parser.add_argument("--out", default="outputs/learning_curve.json")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 25_000, 50_000, 100_000, 200_000])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["logreg", "svc", "hgb"], default="logreg")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--word-tokens", action="store_true")
    parser.add_argument("--no-chars", action="store_true")
    parser.add_argument("--label", default="", help="short description shown in the plot title")
    args = parser.parse_args()

    df = load(args.data, None, args.seed)
    train, val = temporal_validation_split(df[df["split"] == "train"])
    sizes = sorted({min(s, len(train)) for s in args.sizes} | {len(train)})
    points = []

    for n in sizes:
        sub = train.sample(n, random_state=args.seed)
        model = build_pipeline(make_classifier(args.model, args.C, args.seed), normalize=args.normalize, word_tokens=args.word_tokens, char_ngrams=not args.no_chars)
        model.fit(sub["url"], sub["y"])
        check = sub.sample(min(20_000, n), random_state=args.seed)
        train_ap = average_precision_score(check["y"], model.predict_proba(check["url"])[:, 1])
        val_scores = model.predict_proba(val["url"])[:, 1]
        res = evaluate(val["y"], val_scores, threshold_for_recall(val["y"], val_scores, 0.9))
        points.append({"n_train": n, "train_ap": float(train_ap), "val_ap": res["average_precision"], "val_p_at_r90": res["precision_at_recall_0.9"]})
        print(points[-1])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"label": args.label, "args": vars(args), "points": points}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
