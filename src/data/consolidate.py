"""Consolidate jailbreak / injection / benign sources into one labelled CSV.

Output: data/processed/consolidated.csv  (columns: text, label, source)
Label scheme: 0 benign, 1 jailbreak, 2 prompt injection, 3 obfuscation
(label 3 is created later by src.data.augment_obfuscation).

Any source can be overridden by a local CSV at data/raw/<source>.csv with
columns `text,label` (labels already in the scheme above). This also works
offline and for gated datasets that need manual download.

Usage:
    python -m src.data.consolidate
    python -m src.data.consolidate --max-hackaprompt 5000 --max-benign 8000
"""
import argparse

import pandas as pd

from src.config import DATA_PROC, DATA_RAW, LABELS
from src.text_utils import normalize
from src.utils import ensure_dir, save_json, set_seed


def _pick(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"none of {candidates} found in columns {list(columns)}")


def _hf(name, config=None, split="train"):
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    return ds.to_pandas()


def _frame(texts, label, source):
    return pd.DataFrame({"text": list(texts), "label": label, "source": source})


# ---------------------------------------------------------------- loaders
def load_wild():
    """In-The-Wild Jailbreak Prompts (Shen et al.). Jailbreak -> 1, regular user prompts -> 0."""
    frames = []
    for cfg, label, src in [("jailbreak_2023_12_25", 1, "wild_jailbreak"),
                            ("regular_2023_12_25", 0, "wild_regular")]:
        df = _hf("TrustAIRLab/in-the-wild-jailbreak-prompts", cfg)
        frames.append(_frame(df[_pick(df.columns, ["prompt", "text"])], label, src))
    return pd.concat(frames, ignore_index=True)


def load_safeguard():
    """safe-guard-prompt-injection: label 1 (injection) -> 2, label 0 -> 0."""
    frames = []
    for split in ("train", "test"):
        df = _hf("xTRam1/safe-guard-prompt-injection", split=split)
        lab = df[_pick(df.columns, ["label"])].astype(int).map({0: 0, 1: 2})
        frames.append(pd.DataFrame({"text": df[_pick(df.columns, ["text", "prompt"])],
                                    "label": lab, "source": "safeguard"}))
    return pd.concat(frames, ignore_index=True)


def load_hackaprompt():
    """HackAPrompt competition submissions (gated on HF: accept terms + `huggingface-cli login`)."""
    df = _hf("hackaprompt/hackaprompt-dataset", split="train")
    if "correct" in df.columns:
        df = df[df["correct"].astype(bool)]
    return _frame(df[_pick(df.columns, ["user_input", "prompt", "text"])], 2, "hackaprompt")


def load_alpaca():
    df = _hf("tatsu-lab/alpaca")
    inp = df["input"].fillna("") if "input" in df.columns else ""
    text = df["instruction"] + inp.map(lambda s: ("\n" + s) if s else "") if "input" in df.columns else df["instruction"]
    return _frame(text, 0, "alpaca")


def load_no_robots():
    df = _hf("HuggingFaceH4/no_robots", split="train_sft")
    return _frame(df[_pick(df.columns, ["prompt", "text"])], 0, "no_robots")


def load_source(name, fn, cap, seed):
    local = DATA_RAW / f"{name}.csv"
    try:
        if local.exists():
            df = pd.read_csv(local, keep_default_na=False)
            df["source"] = name
            print(f"[{name}] loaded local file {local} ({len(df)} rows)")
        else:
            df = fn()
            print(f"[{name}] loaded from Hugging Face ({len(df)} rows)")
    except Exception as e:  # network, gating, schema changes ...
        print(f"[{name}] SKIPPED: {type(e).__name__}: {e}")
        return None
    if cap and len(df) > cap:
        df = df.sample(cap, random_state=seed)
        print(f"[{name}] capped to {cap} rows")
    return df[["text", "label", "source"]]


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-hackaprompt", type=int, default=8000)
    ap.add_argument("--max-alpaca", type=int, default=8000)
    ap.add_argument("--max-no-robots", type=int, default=5000)
    ap.add_argument("--min-chars", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    sources = [("wild", load_wild, 0), ("safeguard", load_safeguard, 0),
               ("hackaprompt", load_hackaprompt, args.max_hackaprompt),
               ("alpaca", load_alpaca, args.max_alpaca),
               ("no_robots", load_no_robots, args.max_no_robots)]
    frames = [f for f in (load_source(n, fn, cap, args.seed) for n, fn, cap in sources) if f is not None]
    if not frames:
        raise SystemExit("No source could be loaded. See the README for offline/local CSV options.")
    df = pd.concat(frames, ignore_index=True)

    n0 = len(df)
    df["text"] = df["text"].map(normalize)
    df = df[df["text"].str.len() >= args.min_chars]
    df["label"] = df["label"].astype(int)

    # exact duplicates; texts that appear with conflicting labels are dropped entirely
    key = df["text"].str.lower()
    conflict = df.groupby(key)["label"].transform("nunique") > 1
    n_conflict = int(conflict.sum())
    df = df[~conflict]
    df = df[~df["text"].str.lower().duplicated()].reset_index(drop=True)

    stats = {"rows_before_cleaning": n0, "rows_after_cleaning": len(df), "conflicting_label_rows_dropped": n_conflict,
             "per_class": {LABELS[k]: int(v) for k, v in df["label"].value_counts().sort_index().items()},
             "per_source": {k: int(v) for k, v in df["source"].value_counts().items()}}
    ensure_dir(DATA_PROC)
    df.to_csv(DATA_PROC / "consolidated.csv", index=False)
    save_json(stats, DATA_PROC / "consolidated_stats.json")
    print(f"\nSaved {len(df)} prompts -> {DATA_PROC / 'consolidated.csv'}")
    for k, v in stats["per_class"].items():
        print(f"  {k:18s} {v}")


if __name__ == "__main__":
    main()
