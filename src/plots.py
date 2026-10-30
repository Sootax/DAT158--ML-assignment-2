import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve

from src.evaluate import precision_at_base_rate
from src.features import LEXICAL_FEATURE_NAMES
from src.tracking import load_runs, load_scores

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, GRAY = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#b5b3aa"
SERIES = [BLUE, ORANGE, AQUA]
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
KEY = "model_val"


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 7
    })


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_dir / name)


def label(run: dict, i: int | None = None) -> str:
    return f"{i}. {run['name']}" if i is not None else run["name"]


def write_runs_table(runs: list[dict], path: Path) -> None:
    rows = ["| # | Run | Forelder | Endring | AP (val) | Δ AP | P@R=0,9 (val) | Δ P@R | Flip | ms/URL | AP (test) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    seen: list[tuple[str, float, float]] = []

    for i, r in enumerate(runs, 1):
        v = r["results"].get(KEY, {})
        ap, par = v.get("average_precision"), v.get("precision_at_recall_0.9")
        parent_name = (r.get("params") or {}).get("parent")
        ref = next((e for e in reversed(seen) if e[0] == parent_name), None) if parent_name else None
        ref = ref or (seen[-1] if seen else None)
        d_ap = f"{ap - ref[1]:+.3f}" if ref else "–"
        d_par = f"{par - ref[2]:+.3f}" if ref else "–"
        test_ap = r["results"].get("model_test", {}).get("average_precision")
        flip = r["results"].get("robustness", {}).get("decision_flip_rate")
        ms = r["results"].get("latency_ms_per_url")
        rows.append(f"| {i} | {r['name']} | {parent_name or ('–' if i == 1 else 'forrige')} | {r['note'] or '(ingen notat)'} | {ap:.3f} | {d_ap} | {par:.3f} | {d_par} | "
                    f"{'–' if flip is None else f'{flip:.1%}'} | {'–' if ms is None else f'{ms:.1f}'} | {'–' if test_ap is None else f'{test_ap:.3f}'} |")
        seen.append((r["name"], ap, par))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("wrote", path)


def plot_progress(runs: list[dict], out_dir: Path) -> None:
    x = np.arange(1, len(runs) + 1)
    ap = [r["results"][KEY]["average_precision"] for r in runs]
    par = [r["results"][KEY]["precision_at_recall_0.9"] for r in runs]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))

    for vals, color, name in [(ap, BLUE, "Average precision"), (par, ORANGE, "Presisjon ved 90 % gjenkalling")]:
        ax.plot(x, vals, color=color, marker="o", label=name, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{vals[-1]:.2f}", (x[-1], vals[-1]), textcoords="offset points", xytext=(8, 0), va="center", color=INK, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([label(r, i) for i, r in enumerate(runs, 1)], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.6, len(runs) + 0.9)
    ax.set_ylabel("Verdi (valideringssett)")
    ax.set_title("Utvikling over kjøringer")
    ax.legend(loc="lower right")
    save(fig, out_dir, "progress.png")


def plot_pr_curves(runs: list[dict], out_dir: Path, scores_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    base = None

    for i, r in enumerate(runs):
        s = load_scores(scores_dir, r["run_id"])
        y = s["y_val"]
        base = y.mean()
        p, rec, _ = precision_recall_curve(y, s["model_val"])
        ax.plot(rec, p, color=SERIES[i], label=label(r), linewidth=2)

    ax.axhline(base, color=GRAY, linestyle="--", linewidth=1.2)
    ax.annotate(f"Tilfeldig gjetting ({base:.0%})", (0.02, base), textcoords="offset points", xytext=(0, 5), color=INK2, fontsize=8)
    ax.axvline(0.9, color=MUTED, linestyle=":", linewidth=1.2)
    ax.annotate("90 % gjenkalling", (0.9, 0.02), textcoords="offset points", xytext=(-4, 0), ha="right", color=INK2, fontsize=8)
    ax.set_xlabel("Gjenkalling (recall)")
    ax.set_ylabel("Presisjon")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Presisjon–gjenkalling (valideringssett)")
    ax.legend(loc="lower left")
    save(fig, out_dir, "pr_curves.png")


def plot_confusion(run: dict, out_dir: Path, key: str = KEY) -> None:
    c = run["results"][key]["confusion"]
    m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    share = m / m.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(share, cmap=SEQUENTIAL, vmin=0, vmax=1)
    ax.grid(False)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]:,}\n{share[i, j]:.0%}".replace(",", " "), ha="center", va="center", color="white" if share[i, j] > 0.45 else INK, fontsize=10)

    ax.set_xticks([0, 1], ["Legitim", "Phishing"])
    ax.set_yticks([0, 1], ["Legitim", "Phishing"])
    ax.set_xlabel("Predikert")
    ax.set_ylabel("Faktisk")

    for s in ax.spines.values():
        s.set_visible(False)

    ax.set_title(f"Forvekslingsmatrise: {run['name']}")
    save(fig, out_dir, "confusion.png")


def plot_base_rate(runs: list[dict], out_dir: Path) -> None:
    rates = np.logspace(-3, np.log10(0.5), 200)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))

    for i, r in enumerate(runs):
        c = r["results"][KEY]["confusion"]
        rec, fpr = c["tp"] / (c["tp"] + c["fn"]), c["fp"] / (c["fp"] + c["tn"])
        ax.plot(rates, [precision_at_base_rate(rec, fpr, b) for b in rates], color=SERIES[i], label=label(r))

    for b in (0.01, 0.05):
        ax.axvline(b, color=MUTED, linestyle=":", linewidth=1.2)
        ax.annotate(f"{b:.0%}", (b, 1.0), textcoords="offset points", xytext=(3, -2), va="top", color=INK2, fontsize=8)

    ax.set_xscale("log")
    ax.set_xticks([0.001, 0.01, 0.1, 0.5], ["0,1 %", "1 %", "10 %", "50 %"])
    ax.minorticks_off()
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Andel phishing i trafikken (logaritmisk akse)")
    ax.set_ylabel("Forventet presisjon")
    ax.set_title("Presisjon faller når phishing er sjeldent")

    if len(runs) > 1:
        ax.legend(loc="upper right")

    save(fig, out_dir, "base_rate.png")


def plot_class_share_by_month(data_path: str, out_dir: Path) -> None:
    df = pd.read_parquet(data_path)
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    df["phish"] = (df["label"] == "phish").astype(int)
    splits = [s for s in ("train", "test") if (df["split"] == s).any()]
    stats = {s: df[df["split"] == s].groupby("month").agg(share=("phish", "mean"), n=("phish", "size")) for s in splits}
    widths = [len(stats[s]) for s in splits]
    fig, axes = plt.subplots(2, len(splits), figsize=(6.8, 4.2), sharey="row", squeeze=False, gridspec_kw={"width_ratios": widths, "height_ratios": [3, 1.3]})

    for col, s in enumerate(splits):
        st = stats[s]
        top, bottom = axes[0, col], axes[1, col]
        top.bar(st.index, st["share"], color=BLUE, width=0.7)
        top.set_ylim(0, 1)
        top.set_title({"train": "Trening", "test": "Test"}[s])
        top.set_xticklabels([])
        bottom.bar(st.index, st["n"], color=GRAY, width=0.7)
        bottom.tick_params(axis="x", rotation=60, labelsize=7)

        for lab in bottom.get_xticklabels():
            lab.set_ha("right")

    axes[0, 0].set_ylabel("Andel phishing")
    axes[0, 0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[1, 0].set_ylabel("Antall URL-er")
    axes[1, 0].yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    fig.suptitle("Klassefordelingen varierer sterkt fra måned til måned", x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    save(fig, out_dir, "class_share_by_month.png")


def plot_top_features(model_path: str, out_dir: Path, k: int = 12) -> None:
    model = joblib.load(model_path)
    clf = model.named_steps["clf"]

    if not hasattr(clf, "coef_"):
        print("skip top features: classifier has no coef_")

        return

    union = model.named_steps["features"]
    names = list(LEXICAL_FEATURE_NAMES)

    for branch, transformer in union.transformer_list[1:]:
        prefix = "ord: " if branch == "words" else ""
        names += [f'{prefix}"{t}"' for t in transformer.get_feature_names_out()]

    coef = clf.coef_[0]
    order = np.argsort(coef)
    picks = list(order[:k]) + list(order[-k:])
    vals = coef[picks]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    colors = [ORANGE if v < 0 else BLUE for v in vals]
    ax.barh(range(len(picks)), vals, color=colors, height=0.7)
    ax.set_yticks(range(len(picks)), [names[i] for i in picks], fontsize=8)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Koeffisient (logistisk regresjon)")
    ax.set_title("Egenskaper som veier tyngst")
    ax.text(0.98, 0.02, "blå: peker mot phishing", transform=ax.transAxes, ha="right", color=BLUE, fontsize=8)
    ax.text(0.02, 0.98, "oransje: peker mot legitim", transform=ax.transAxes, ha="left", va="top", color=ORANGE, fontsize=8)
    save(fig, out_dir, "top_features.png")


def plot_learning_curve(path: Path, out_dir: Path) -> None:
    data = json.loads(path.read_text())
    pts = data["points"]
    n = [p["n_train"] for p in pts]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    for key, color, name in [("train_ap", BLUE, "Treningssett"), ("val_ap", ORANGE, "Valideringssett")]:
        vals = [p[key] for p in pts]
        ax.plot(n, vals, color=color, marker="o", markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{name} {vals[-1]:.2f}", (n[-1], vals[-1]), textcoords="offset points", xytext=(-6, 8 if key == "train_ap" else -14), ha="right", color=INK, fontsize=9)

    ax.set_xscale("log")
    ax.set_xticks(n, [f"{v // 1000}k" for v in n])
    ax.minorticks_off()
    ax.set_xlabel("Antall treningseksempler")
    ax.set_ylabel("Average precision")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Læringskurve" + (f": {data['label']}" if data.get("label") else ""))
    save(fig, out_dir, "learning_curve.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="outputs")
    parser.add_argument("--out", default="reports/figures")
    parser.add_argument("--data", default="data/phreshphish_urls.parquet")
    parser.add_argument("--compare", nargs="+", help="run ids for the comparison charts (max 3)")
    parser.add_argument("--eda", action="store_true")
    parser.add_argument("--features", action="store_true")
    parser.add_argument("--learning-curve", action="store_true", help="figure from outputs/learning_curve.json")
    args = parser.parse_args()

    setup_style()
    out = Path(args.out)
    runs = [r for r in load_runs(args.runs_dir) if KEY in r["results"]]

    if args.learning_curve:
        plot_learning_curve(Path(args.runs_dir) / "learning_curve.json", out)

    if args.eda:
        plot_class_share_by_month(args.data, out)

    if not runs:
        print("No runs logged yet. Run python -m src.train first.")

        return

    write_runs_table(runs, out.parent / "runs_table.md")
    plot_progress(runs, out)
    chosen = ([r for r in runs if r["run_id"] in args.compare] if args.compare else runs[-3:])[:3]
    plot_pr_curves(chosen, out, args.runs_dir)
    plot_base_rate(chosen, out)
    plot_confusion(chosen[-1], out)

    if args.features:
        plot_top_features(Path(args.runs_dir) / "model.joblib", out)


if __name__ == "__main__":
    main()
