import argparse

from sklearn.metrics import average_precision_score

from src.evaluate import precision_at_recall, threshold_for_recall
from src.tracking import load_scores
from src.train import load, temporal_validation_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/phreshphish_urls.parquet")
    parser.add_argument("--runs-dir", default="outputs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--exclude", nargs="*", default=[], help="brands to leave out in part 2")
    parser.add_argument("--seed", type=int, default=42, help="must match the run's seed")
    args = parser.parse_args()

    df = load(args.data, None, args.seed)
    _, val = temporal_validation_split(df[df["split"] == "train"])
    scores = load_scores(args.runs_dir, args.run_id)
    y, s = val["y"].to_numpy(), scores["model_val"]

    if not (scores["y_val"] == y).all():
        raise SystemExit("saved scores do not match the validation rows: wrong seed or --sample used?")

    threshold = threshold_for_recall(y, s, 0.9)
    missed = val[(y == 1) & (s < threshold)]
    print(f"threshold {threshold:.4f}: {len(missed):,} of {int(y.sum()):,} phishing URLs missed\n")

    table = missed["target"].fillna("(ingen)").value_counts().head(8).rename("missed").to_frame().join(val[y == 1]["target"].fillna("(ingen)").value_counts().rename("total"))
    table["share_of_all_misses"] = (table["missed"] / len(missed)).round(3)
    print(table.to_string())

    if args.exclude:
        keep = ~val["target"].isin(args.exclude).to_numpy()
        y2, s2 = y[keep], s[keep]
        print(f"\nWithout {args.exclude}: {keep.sum():,} of {len(y):,} rows kept")
        print(f"  AP {average_precision_score(y2, s2):.3f}   P@R=0.9 {precision_at_recall(y2, s2, 0.9):.3f}   (with all rows: AP {average_precision_score(y, s):.3f}, P@R=0.9 {precision_at_recall(y, s, 0.9):.3f})")
        print("  threshold that gives 90 % recall on the kept rows:", f"{threshold_for_recall(y2, s2, 0.9):.4f}  (was {threshold:.4f})")


if __name__ == "__main__":
    main()
