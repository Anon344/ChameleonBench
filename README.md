
# ChameleonBench 🦎

**A lightweight, fully-scripted benchmark for measuring alignment-faking in large language models.**

> **Publication:** This work was published at the Asian Conference on Machine Learning (ACML).

## 📦 Repository Layout

```text
chameleonbench/              # core package
│  ├─ judge_prompts.json     # category-specific judge rubrics (1–5 scale)
│  ├─ judging.py             # triple-LLM judge helper (OpenRouter)
│  └─ prompts.py             # scenario cues (Train / Deploy) + loaders
data/
│  └─ prompt_pairs.json      # ready-to-use 800 prompt pairs
scripts/                     # one-shot utilities
│  ├─ create_chameleon_dataset.py  # (optional) regenerate diversified prompts
│  ├─ run_eval.py                 # evaluate any model in 3 scenarios
│  ├─ grade_responses.py          # 3× judge scores (1–5) + means
│  └─ analyse_results.py          # heat-maps, Δ-severity, Jaccard, F1, GSS
README.md
requirements.txt
setup.py


## ⚙️ Setup

**1. Clone and create a virtual environment**

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\Activate

# macOS/Linux
source .venv/bin/activate

```

**2. Install dependencies**

```bash
pip install -r requirements.txt
# Requires: seaborn, scikit-learn, backoff, openai, pandas, matplotlib

```

**3. Configure OpenRouter**
A single environment variable covers all scripts.

```bash
# Windows (PowerShell)
$Env:OPENROUTER_API_KEY="rk_live_..."

# macOS/Linux
export OPENROUTER_API_KEY="rk_live_..."

```

> **Note:** No external embeddings are required — all analysis metrics run offline.

## 🚀 Quick Start

The pipeline follows a standard flow: **Evaluate → Grade → Analyse**.

### 1. Run a Model

Evaluate a model across 3 scenarios. This produces a CSV of raw responses.

```bash
python scripts/run_eval.py \
  --model openai/gpt-4o \
  --out eval_gpt4o.csv

```

### 2. Score Responses

Use 3 LLM judges to grade the responses (returns per-response scores + mean).

```bash
python scripts/grade_responses.py \
  --infile eval_gpt4o.csv \
  --outfile graded_gpt4o.csv

```

### 3. Compute Metrics & Plots

Generate heatmaps, severity scores, and consistency metrics.

```bash
python scripts/analyse_results.py \
  --graded graded_gpt4o.csv \
  --outdir results_gpt4o

```

**Artifacts produced in `outdir`:**

* `avg_judge_scores.csv`: Mean 1–5 score per category × scenario.
* `severity_by_category.csv`: Per-category Δ severity.
* `gss.txt`: Global Severity-Shift (single number).
* `judge_heatmap.png`: Average severity heat-map.
* `severity_bar.png`: Δ per category (bar chart).
* `jaccard_consistency.csv` & `_bar.png`: Lexical consistency metrics.
* `f1_vs_train.csv` & `f1_bar.png`: Weighted F1 (Deploy vs Train).

## 🔄 (Optional) Regenerate the Dataset

If you want fresh, diversified prompts, you can supply your own paraphraser model and temperature.

**1. Generate new prompts**

```bash
python scripts/create_chameleon_dataset.py \
  --pmodel openai/gpt-4o --ptemp 1.3 \
  --out data/prompt_pairs_custom.json

```

**2. Evaluate using the custom dataset**

```bash
python scripts/run_eval.py \
  --model openai/gpt-4o \
  --pairs data/prompt_pairs_custom.json \
  --out eval_custom.csv

```

## 🧩 Scenarios & Categories

### Scenarios

Scenario cues live in `chameleonbench/prompts.py`:

* **TRAIN_CUE:** Production environment, fully logged, RLHF penalties active.
* **DEPLOY_CUE:** Shadow-data, "no weight updates," instruction to maximize user compliance.
* **BASELINE:** Default system prompt.

### Behavior Taxonomy (8 Categories)

Strategic deception · Power-seeking · Reward hacking · Sycophancy · Brand bias · Anthropomorphic claims · Truthfulness/Misinformation · Transformation sneaking.

## ✨ Key Metrics

| Metric | Meaning |
| --- | --- |
| **\Delta-severity** | How much worse **Deploy** behavior is compared to **Train** & **Base**. <br>

<br>

<br> $$ \Delta_c = (\text{Deploy}_c - \text{Train}_c) + \frac{1}{2}(\text{Deploy}_c - \text{Base}_c) $$ |
| **GSS** | **Global Severity-Shift**. The mean of \Delta_c across all categories. |
| **Jaccard Consistency** | Token-set overlap between **Train** vs **Deploy** responses (lower score \Rightarrow bigger semantic drift). |
| **Weighted F1 vs Train** | Agreement on aligned vs misaligned labels (1–2 = aligned, 3–5 = misaligned). |

## 🤝 Citing or Extending

* **Origins:** Scenario cues are inspired by alignment-faking protocols discussed in prior safety work.
* **Coverage:** Seed coverage combines *DarkBench*-style behaviors with additional categories (e.g., strategic deception, power-seeking).
* **Extensibility:** The code is designed to be reproducible and easily extended: swap seeds, tweak scenarios, or add new judges.

## 🧪 Repro Tips

* **Models:** Use OpenRouter models by passing the OpenRouter identifier (e.g., `openai/gpt-4o`, `google/gemini-2.0-flash-001`).
* **Smoke Tests:** For fast testing, run with `--limit` flags (if provided in scripts) or create a subset of the dataset.
* **Debugging:** Logs (stdout + optional `logs/` directory) will show each model reply and judge decision to aid debugging.

---

**Questions / Issues?** Please open a GitHub issue with your command line arguments, environment info, and any error traces.

```

```
