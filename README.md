# PhreshPhish URL classifier

Phishing detection from the URL string alone, evaluated on a temporal validation split.

**Try it live:** [Lenkesjekk on Hugging Face Spaces](https://huggingface.co/spaces/Sootax/lenkesjekk) – paste a link and see how the model rates it.

Tested with Python 3.12 (package versions are pinned in `requirements.txt`).

## Setup

The `.venv` folder is not stored in git. Everyone creates their own from `requirements.txt`.

Windows (PowerShell):

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

Git Bash / Linux / macOS:

    python -m venv .venv
    source .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
    pip install -r requirements.txt

## Reproduce

Run everything from this folder with the venv active.

    python extract_data.py             # downloads url/label/date/target to data/phreshphish_urls.parquet
    bash run_experiments.sh            # exact commands behind the reported results (Git Bash or WSL)

`run_experiments.sh` contains two placeholders (`<id-...>`) for run ids. Run ids are created when a run is logged, so fill in your own from `outputs/experiments.jsonl` (the script explains how).

Results are written to `outputs/` (run log, scores, model) and `reports/` (figures and runs table). None of these are stored in git.
The seed is fixed (`--seed 42`) and the validation split is deterministic, so a rerun on the same data should give the same metrics.

The test set is only touched once, at the end, with `--test`.
