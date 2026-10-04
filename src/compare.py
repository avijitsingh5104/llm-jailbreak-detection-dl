"""Collect results/*.json from every model into one table and the report figures.

Writes results/results_table.md, results_table.csv, fig_performance.png,
fig_training_time.png and fig_generalization.png.
Usage: python -m src.compare [--binary]
"""
import argparse

import pandas as pd

from src.config import RESULTS
from src.utils import load_json

ORDER = [("mlp", "MLP (TF-IDF)"), ("cnn", "CNN (word)"), ("cnn_char", "CNN (char)"),
         ("bilstm", "BiLSTM"), ("transformer", "Transformer")]


def collect(binary: bool) -> pd.DataFrame:
    rows = []
    for key, label in ORDER:
        path = RESULTS / f"{key}{'_binary' if binary else ''}.json"
        if not path.exists():
            continue
        r = load_json(path)
        t, n = r["test"], r.get("novel")
        rows.append({"Model": label, "Accuracy": t["accuracy"], "Precision": t["precision"], "Recall": t["recall"],
                     "F1": t["f1"], "Train time (min)": r["train_time_min"],
                     "Novel Acc.": n["accuracy"] if n else None, "Novel F1": n["f1"] if n else None,
                     "F1 gap": r.get("generalization_gap_f1"),
                     "Attack detection": t.get("attack_detection_rate"), "FPR": t.get("false_positive_rate"),
                     "ms/prompt": t.get("ms_per_prompt"), "Params": r["params"]})
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    def fmt(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "-"
        return f"{v:,}" if isinstance(v, int) else f"{v:.4f}" if isinstance(v, float) else str(v)
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)) + "\n"


def plots(df: pd.DataFrame, suffix: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    x, w = np.arange(len(df)), 0.2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(metrics):
        bars = ax.bar(x + (i - 1.5) * w, df[m], w, label=m)
        for b, v in zip(bars, df[m]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(x, df["Model"])
    ax.set_ylim(max(0, df[metrics].min().min() - 0.1), 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Model performance comparison (in-distribution test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / f"fig_performance{suffix}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(df["Model"], df["Train time (min)"], color="#4c72b0")
    for b, v in zip(bars, df["Train time (min)"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f} min", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Training time (minutes)")
    ax.set_title("Training time comparison")
    fig.tight_layout()
    fig.savefig(RESULTS / f"fig_training_time{suffix}.png", dpi=150)
    plt.close(fig)

    g = df.dropna(subset=["Novel F1"])
    if len(g):
        x = np.arange(len(g))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - 0.2, g["F1"], 0.4, label="In-distribution test")
        ax.bar(x + 0.2, g["Novel F1"], 0.4, label="Novel-phrasing test")
        ax.set_xticks(x, g["Model"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Macro F1")
        ax.set_title("Generalization to unseen attack phrasing")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / f"fig_generalization{suffix}.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", action="store_true")
    args = ap.parse_args()
    suffix = "_binary" if args.binary else ""
    df = collect(args.binary)
    if df.empty:
        raise SystemExit("No results found in results/. Train at least one model first.")
    (RESULTS / f"results_table{suffix}.md").write_text(to_markdown(df), encoding="utf-8")
    df.to_csv(RESULTS / f"results_table{suffix}.csv", index=False)
    plots(df, suffix)
    print(to_markdown(df))


if __name__ == "__main__":
    main()
