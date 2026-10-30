import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import sklearn


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5)

        return out.stdout.strip() if out.returncode == 0 else None

    except Exception:
        return None


def git_state() -> str | None:
    commit = _git("rev-parse", "--short", "HEAD")

    if not commit:
        return None

    return commit + ("+dirty" if _git("status", "--porcelain") else "")


def features_hash() -> str:
    source = Path(__file__).with_name("features.py").read_bytes()

    return hashlib.sha1(source).hexdigest()[:8]


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]

    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    return obj


def log_run(out_dir: str, name: str, note: str, params: dict, results: dict, data_info: dict, arrays: dict) -> str:
    out = Path(out_dir)
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-") or "run"
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{slug}"
    run_dir = out / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(run_dir / "scores.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    results = _clean(results)
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False))

    record = {
        "run_id": run_id,
        "time": datetime.now().isoformat(timespec="seconds"),
        "name": name,
        "note": note,
        "params": params,
        "data": data_info,
        "features_hash": features_hash(),
        "git": git_state(),
        "sklearn": sklearn.__version__,
        "results": results,
    }

    with open(out / "experiments.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(_clean(record), allow_nan=False) + "\n")

    return run_id


def load_runs(out_dir: str) -> list[dict]:
    path = Path(out_dir) / "experiments.jsonl"

    if not path.exists():
        return []

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_scores(out_dir: str, run_id: str) -> dict:
    with np.load(Path(out_dir) / "runs" / run_id / "scores.npz") as data:
        return {k: data[k] for k in data.files}
