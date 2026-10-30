# Reproduces the experiments (full data, validation set only).
#
# Before you start:
#   1. Create and activate the venv, install requirements.txt (see README.md).
#   2. Run from the project root: python extract_data.py   (creates data/phreshphish_urls.parquet)
#   3. Use Git Bash or WSL. In Windows cmd/PowerShell the for-loop and comments must be rewritten.
#
# Each training run takes about 1.5-4 minutes and is appended to outputs/experiments.jsonl.
# Do NOT add --test to any of these commands. The test set is used once, at the very end.
# The --note texts are stored in the run log and are shown in the runs table, so keep them as they are.


# Step 1: describe how benign and phishing URLs are written (no training)
python -m src.artefacts

# Step 2: baseline and feature changes, each compared to its --parent
python -m src.train --name baseline --note "Utgangspunkt: logistisk regresjon (C=1), 18 leksikalske egenskaper + tegn-n-grammer 3-5"
python -m src.train --name normaliser --parent baseline --normalize --note "Normaliserer URL-en (fjerner skjema, www., avsluttende / og store bokstaver) før egenskapene beregnes"
python -m src.train --name ord-tokens --parent normaliser --normalize --word-tokens --note "Legger til TF-IDF på hele ord (delt ved tegn som ./-_?=&)"

# Step 3: regularisation sweep on the best feature set
for C in 0.3 3 10 30 100; do
python -m src.train --name "C-$C" --parent ord-tokens --normalize --word-tokens --C $C --note "Regularisering: C=$C (var 1) på normalisert URL + ord-tokens"
done

# Step 4: other models and n-gram lengths
python -m src.train --name linearsvc --parent ord-tokens --normalize --word-tokens --model svc --note "LinearSVC (C=1) med sigmoid-kalibrering (3-fold) i stedet for logistisk regresjon, som i artikkelen"
python -m src.train --name hgb-leksikalsk --parent normaliser --normalize --no-chars --model hgb --note "Gradient boosting på bare de 18 leksikalske egenskapene (ingen n-grammer)"
python -m src.train --name ngram-4-6 --parent ord-tokens --normalize --word-tokens --ngram-min 4 --ngram-max 6 --note "Lengre tegn-n-grammer: 4-6 i stedet for 3-5 (C=1)"

# Step 5: learning curve (writes outputs/learning_curve.json)
python -m src.learning_curve --normalize --word-tokens --C 10 --label "normalisert URL, ord-tokens, C=10"


# The next two steps need run ids. Run ids are generated when a run is logged (date-time + name).
# List yours with:  grep -o '"run_id": "[^"]*"' outputs/experiments.jsonl
# Replace <id-baseline>, <id-normaliser> and <id-C-10> below with the ids of those runs.

# Step 6: error analysis, which phishing campaigns (brands) make up the missed URLs
python -m src.campaigns --run-id <id-C-10> --exclude robinhood qmp

# Step 7: report figures (writes reports/figures/ and reports/runs_table.md)
python -m src.plots --eda --learning-curve --compare <id-baseline> <id-normaliser> <id-C-10>


# Step 8: final model (normalize + word tokens + C=10), the one exported to web/
# It is trained into its own folder so it does not add a row to the main run log.
python -m src.train --out outputs_final --name valgt --normalize --word-tokens --C 10

# Step 9: top-features figure, which needs the final model from step 8
python -m src.plots --runs-dir outputs_final --out reports_final/figures --features
