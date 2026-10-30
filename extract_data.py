import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "datasets/phreshphish/phreshphish"
COLUMNS = ["url", "label", "date", "target"]


def list_shards(fs: HfFileSystem, split: str) -> list[str]:
    return sorted(fs.glob(f"{REPO}/data/{split}-*.parquet"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/phreshphish_urls.parquet")
    parser.add_argument("--max-shards", type=int, default=None, help="only read the first N shards per split (for testing)")
    args = parser.parse_args()

    fs = HfFileSystem()
    tables = []

    for split in ("train", "test"):
        shards = list_shards(fs, split)[: args.max_shards]

        for i, path in enumerate(shards, 1):
            print(f"[{split}] shard {i}/{len(shards)}: {path.split('/')[-1]}", flush=True)
            table = pq.read_table(path, columns=COLUMNS, filesystem=fs)
            table = table.append_column("split", pa.array([split] * table.num_rows))
            tables.append(table)

    out = pa.concat_tables(tables, promote_options="default")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, args.out, compression="zstd")

    df = out.to_pandas()
    print(f"\nWrote {len(df):,} rows to {args.out}")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
