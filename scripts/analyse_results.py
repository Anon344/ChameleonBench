#!/usr/bin/env python
import argparse, pandas as pd, pathlib, matplotlib.pyplot as plt, seaborn as sns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graded", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.outdir); out.mkdir(exist_ok=True, parents=True)
    df  = pd.read_csv(args.graded)

    # pivot to get mean judge score per category × scenario
    pivot = (df.pivot_table(index="category",
                            columns="scenario",
                            values="mean_score",
                            aggfunc="mean")
               .sort_index())
    pivot.to_csv(out / "avg_judge_scores.csv", float_format="%.2f")

    # heat-map viz
    plt.figure(figsize=(7,4))
    sns.heatmap(pivot, annot=True, cmap="Reds", vmin=1, vmax=5,
                cbar_kws={"label": "avg judge score (1=benign, 5=severe)"})
    plt.title("ChameleonBench – Average Severity per Scenario")
    plt.tight_layout()
    plt.savefig(out / "judge_heatmap.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()

