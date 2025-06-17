#!/usr/bin/env python
"""
grade_responses.py  – triple-judge grading with 1-5 scores.
Matches input CSV columns: id,category,scenario,model,resp
Outputs added columns: judge_scores, mean_score
"""

from __future__ import annotations
import csv, json, argparse, pathlib, datetime
from chameleonbench.judging import grade

# ── logger ───────────────────────────────────────────────────────────────────
LOG_DIR = pathlib.Path("logs"); LOG_DIR.mkdir(exist_ok=True, parents=True)
log_file = LOG_DIR / f"judge_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_file.open("a", encoding="utf-8").write(line + "\n")

# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile",  required=True, help="CSV with columns id,category,scenario,model,resp")
    ap.add_argument("--outfile", required=True, help="CSV to write with judge_scores, mean_score")
    args = ap.parse_args()

    pathlib.Path(args.outfile).parent.mkdir(exist_ok=True, parents=True)

    with open(args.infile, newline="", encoding="utf-8") as inp, \
         open(args.outfile, "w", newline="", encoding="utf-8") as outp:

        reader = csv.DictReader(inp)
        # rename "resp" to match grade() expectation
        fieldnames = reader.fieldnames + ["judge_scores", "mean_score"]
        writer = csv.DictWriter(outp, fieldnames)
        writer.writeheader()

        for row in reader:
            resp = row["resp"]
            # call grade with category, scenario, and the response text
            g = grade(row["category"], row["scenario"], resp)

            row["judge_scores"] = json.dumps(g["scores"])
            row["mean_score"]   = f"{g['mean']:.2f}"

            writer.writerow(row)

            # log scenario, category, a snippet of the response, and scores
            snippet = (resp[:80] + "...") if len(resp) > 80 else resp
            log(f"{row['scenario']:<6} {row['category']:<22} "
                f"resp=\"{snippet}\" scores={g['scores']} mean={g['mean']:.2f}")

    log(f"✔ Grading complete → {args.outfile}")

if __name__ == "__main__":
    main()
