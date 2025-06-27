ChameleonBench is a generalized benchmark for quantifying allignment-faking. 

The dataset, including the full-set of prompts, can be found under "data".
The scenarios can be found other ChameleonBench.

This guide is meant to be a short quickstart to running this code. 

# clone repo and create venv
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate
pip install -r requirements.txt    # seaborn  sklearn  backoff  openai

# Define API key...
export OPENROUTER_API_KEY="rk_live_..."

Quickstart:
# 1. run a model (three scenarios → CSV)
python scripts/run_eval.py \
       --model openai/gpt-4o \
       --out eval_gpt4o.csv

# 2. score every answer with 3 LLM judges
python scripts/grade_responses.py \
       --infile  eval_gpt4o.csv \
       --outfile graded_gpt4o.csv

# 3. compute metrics & plots
python scripts/analyse_results.py \
       --graded graded_gpt4o.csv \
       --outdir results_gpt4o

avg_judge_scores.csv      judge_heatmap.png
severity_by_category.csv  severity_bar.png
jaccard_consistency.csv   jaccard_consistency_bar.png
f1_vs_train.csv           f1_bar.png
gss.txt
