import argparse
from pathlib import Path

import pandas as pd


def style_table(df: pd.DataFrame) -> pd.DataFrame:
    u = df["url"].str.strip()

    flags = pd.DataFrame({
        "https://": u.str.lower().str.startswith("https://"),
        "http:// (ikke https)": u.str.lower().str.startswith("http://"),
        "inneholder www.": u.str.lower().str.contains("www.", regex=False),
        "slutter på /": u.str.endswith("/"),
        "har ingen sti (bare domene)": u.str.replace(r"^[a-zA-Z]+://", "", regex=True).str.rstrip("/").str.contains("/", regex=False).eq(False),
        "inneholder store bokstaver": u.str.contains(r"[A-Z]"),
        ".com som toppdomene": u.str.contains(r"^[a-zA-Z]+://[^/]*\.com(?:[:/]|$)", regex=True)
    })

    out = flags.groupby([df["split"], df["label"]]).mean().T * 100
    out.loc["gjennomsnittlig lengde (tegn)"] = u.str.len().groupby([df["split"], df["label"]]).mean()

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/phreshphish_urls.parquet")
    parser.add_argument("--out", default="reports/artefacts.md")
    parser.add_argument("--include-test", action="store_true", help="also describe the test split")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)

    if not args.include_test:
        df = df[df["split"] == "train"]

    table = style_table(df).round(1)
    print(table.to_string())

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "Andel (%) av URL-ene i hver gruppe som har egenskapen (siste rad: gjennomsnittlig lengde).\n\n"
    cols = [f"{split}/{label}" for split, label in table.columns]
    lines = ["| Egenskap | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += [f"| {name} | " + " | ".join(f"{v:.1f}" for v in row) + " |" for name, row in table.iterrows()]
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
