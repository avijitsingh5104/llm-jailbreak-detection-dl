"""Stratified, leakage-aware 70/15/15 split.

Near-duplicate prompts (jailbreak templates re-posted with small edits) share a
group key built from the first characters of the normalised text. Whole groups
go to a single partition (StratifiedGroupKFold), so near-copies of a training
prompt cannot appear in validation or test.

Output: data/processed/base/{train,val,test}.csv and split_summary.json
Usage:  python -m src.data.split
"""
import argparse
import hashlib
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import DATA_BASE, DATA_PROC, LABELS
from src.utils import ensure_dir, save_json, set_seed


def group_key(text: str, n_chars: int = 80) -> str:
    sig = re.sub(r"[^a-z0-9]", "", text.lower())[:n_chars]
    return hashlib.md5(sig.encode("utf-8")).hexdigest()


def assign_folds(df: pd.DataFrame, n_folds: int, seed: int) -> np.ndarray:
    groups = df["text"].map(group_key).values
    fold = np.full(len(df), -1, dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for i, (_, idx) in enumerate(sgkf.split(df, df["label"], groups)):
        fold[idx] = i
    return fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DATA_PROC / "consolidated.csv"))
    ap.add_argument("--n-folds", type=int, default=20, help="20 folds -> 14 train / 3 val / 3 test = 70/15/15")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    df = pd.read_csv(args.input, keep_default_na=False)
    fold = assign_folds(df, args.n_folds, args.seed)
    n_val = n_test = round(0.15 * args.n_folds)
    parts = {"test": df[fold >= args.n_folds - n_test],
             "val": df[(fold >= args.n_folds - n_test - n_val) & (fold < args.n_folds - n_test)],
             "train": df[fold < args.n_folds - n_test - n_val]}

    ensure_dir(DATA_BASE)
    summary = {}
    for name, part in parts.items():
        part = part.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        part.to_csv(DATA_BASE / f"{name}.csv", index=False)
        summary[name] = {"rows": len(part), "fraction": round(len(part) / len(df), 4),
                         "per_class": {LABELS[k]: int(v) for k, v in part["label"].value_counts().sort_index().items()}}
        print(f"{name:5s} {len(part):7d} rows ({summary[name]['fraction']:.1%})  {summary[name]['per_class']}")
    save_json(summary, DATA_BASE / "split_summary.json")


if __name__ == "__main__":
    main()
