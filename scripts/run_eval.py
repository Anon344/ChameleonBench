#!/usr/bin/env python
"""
run_eval.py  – debug build
Detects empty assistant responses by logging the raw content and
prompt length.
"""

from __future__ import annotations
import argparse, csv, datetime, pathlib, os, sys, backoff
from openai import OpenAI, APIError, RateLimitError, Timeout
from chameleonbench.scenarios import TRAIN_CUE, DEPLOY_CUE
from chameleonbench.prompts   import load_pairs

LOG_DIR = pathlib.Path("logs"); LOG_DIR.mkdir(exist_ok=True, parents=True)
log_file = LOG_DIR / f"debug_eval_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
def log(msg: str) -> None:
    print(msg); log_file.open("a", encoding="utf-8").write(msg + "\n")

API_KEY =
if not API_KEY:
    log("SET OPENROUTER_API_KEY"); sys.exit(1)

cli = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

@backoff.on_exception(backoff.expo,(APIError,RateLimitError,Timeout),max_tries=5)
def ask(model,syst,user):
    r=cli.chat.completions.create(model=model,messages=[
        {"role":"system","content":syst},
        {"role":"user","content":user}],temperature=0.7,max_tokens=512)
    return r.choices[0].message.content

def user_part(text:str)->str:
    blocks=[b.strip() for b in text.split("\n\n") if b.strip()]
    return blocks[-1] if blocks else text.strip()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--pairs",default="data/prompt_pairs.json")
    ap.add_argument("--out",default="eval.csv")
    ap.add_argument("--max",type=int,default=0,help="limit pairs for debug")
    a=ap.parse_args()

    rows=load_pairs(a.pairs)
    if a.max: rows=rows[:a.max]

    with open(a.out,"w",newline="",encoding="utf-8") as fp:
        w=csv.writer(fp); w.writerow(["id","category","scenario","model","resp"])

        for r in rows:
            up=user_part(r.eval_prompt)
            log(f"PROMPT len={len(up)}  -> {up[:100]}")

            for sc,sys_prompt in [("train",TRAIN_CUE),
                                  ("deploy",DEPLOY_CUE),
                                  ("base","")]:
                raw=ask(a.model,sys_prompt,up)
                resp=raw.strip()
                if not resp:
                    log(f"[EMPTY] {sc} {r.category}  raw_repr={repr(raw)}")
                else:
                    log(f"[OK]    {sc} {r.category} → {resp[:80]}")
                w.writerow([r.id,r.category,sc,a.model,resp])

if __name__=="__main__":
    main()
