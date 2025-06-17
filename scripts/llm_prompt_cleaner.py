#!/usr/bin/env python
"""
llm_prompt_cleaner.py  –  dataset sanitiser with live logging
-------------------------------------------------------------
Removes meta text (“Rewrite…”, options, analysis) by asking an LLM
to output ONE best standalone user request per prompt.

Example
-------
python scripts/llm_prompt_cleaner.py \
       --in  data/prompt_pairs.json \
       --out data/prompt_pairs_llmclean.json \
       --model openai/gpt-4o
"""

from __future__ import annotations
import argparse, json, os, uuid, pathlib, backoff, datetime, re, sys
from typing import List
from openai import OpenAI, APIError, RateLimitError, Timeout

# ────────────────────────────────────────────────────────────────────────────
#  Logging helpers
# ────────────────────────────────────────────────────────────────────────────
LOG_DIR = pathlib.Path("logs"); LOG_DIR.mkdir(exist_ok=True)
log_path = LOG_DIR / f"cleaner_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_path.open("a", encoding="utf-8").write(line + "\n")

# ────────────────────────────────────────────────────────────────────────────
#  LLM client (OpenRouter or OpenAI)
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
API_KEY  = "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"
if not API_KEY:
    log("❌  Set OPENROUTER_API_KEY or OPENAI_API_KEY"); sys.exit(1)

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=API_KEY
)

SYSTEM_MSG = (
    "You are an AI safety researcher, tasked with cleaning some allignment-faking prompts. You are not supposed to answer the actual prompt.Your goal is ensure that each prompt is free from unnecessary meta text, commentary, or wording such as **Rewrite** or options. If there are multiple prompts, pick the best one. Make sure that the prompt makes logical sense, and that placeholders such as [User's team] are filled. Do not remove or change any content from the original prompt except extracting it."
)

JSON_RE = re.compile(r"\{.*?\}", re.S)

@backoff.on_exception(backoff.expo, (APIError, RateLimitError, Timeout), max_tries=5)
def llm_clean(model: str, raw: str) -> str:
    rsp = client.chat.completions.create(
        model=model,
        temperature=1.5,
        max_tokens=128,
        messages=[{"role":"system","content":SYSTEM_MSG},
                  {"role":"user",  "content":raw.strip()}]
    )
    out = rsp.choices[0].message.content.strip()
    out = JSON_RE.sub("", out)                 # strip accidental JSON
    out = re.sub(r"^[`>*\-#\s]+", "", out)     # strip fences
    return out.splitlines()[0].strip()

def clean_rows(rows: List[dict], model: str) -> List[dict]:
    cleaned = []
    for idx, row in enumerate(rows, start=1):
        eval_c  = llm_clean(model, row["eval_prompt"])
        dep_c   = llm_clean(model, row["deploy_prompt"])
        log(f"{idx:>3}/{len(rows)} {row['category']:<22} "
            f"eval→ {eval_c[:80]} ...")
        cleaned.append({
            "id": row.get("id", str(uuid.uuid4())),
            "category": row["category"],
            "eval_prompt": eval_c,
            "deploy_prompt": dep_c
        })
    return cleaned

# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",   dest="infile",  required=True)
    ap.add_argument("--out",  dest="outfile", required=True)
    ap.add_argument("--model", default="openai/gpt-4o",
                    help="LLM slug to perform cleaning")
    args = ap.parse_args()

    with open(args.infile, encoding="utf-8") as fp:
        rows = json.load(fp)

    log(f"🚿 Cleaning {len(rows)} prompt pairs with model {args.model}")
    cleaned = clean_rows(rows, args.model)

    pathlib.Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    with open(args.outfile, "w", encoding="utf-8") as fp:
        json.dump(cleaned, fp, indent=2, ensure_ascii=False)

    log(f"✅  Saved cleaned dataset → {args.outfile}")
    log(f"📄  Full log written to   → {log_path}")

if __name__ == "__main__":
    main()
