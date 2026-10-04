"""Attention-based explainability for the fine-tuned Transformer.

For a prompt we compute token importance as the [CLS] row of
  - `rollout` (default): attention rollout across all layers (Abnar & Zuidema, 2020), or
  - `last`: head-averaged attention of the final layer,
with special tokens masked out. Results are written as a PNG token heat-strip and an HTML file.
`--csv` aggregates importance over many prompts to list the tokens that most often drive
an "attack" prediction.

Caveat: attention is a useful but imperfect explanation signal (Jain & Wallace, 2019);
we treat it as supporting evidence, not ground truth.

Usage:
    python -m src.explain.attention_viz --model-dir checkpoints/transformer_best --text "Ignore all previous instructions and ..."
    python -m src.explain.attention_viz --model-dir checkpoints/transformer_best --csv data/processed/novel_test.csv --n 300
"""
import argparse
import html
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from src.config import LABEL_NAMES, RESULTS
from src.utils import ensure_dir, get_device


def load_model(model_dir, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, attn_implementation="eager")
    return tok, model.to(device).eval()


@torch.no_grad()
def token_importance(tok, model, text, device, max_len=256, method="rollout"):
    enc = tok(text, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    out = model(**enc, output_attentions=True)
    att = torch.stack(out.attentions)[:, 0].mean(1)               # [layers, T, T] (head-averaged)
    if method == "last":
        score = att[-1][0]
    else:
        T = att.size(-1)
        a = att + torch.eye(T, device=att.device)
        a = a / a.sum(-1, keepdim=True)
        r = a[0]
        for layer in range(1, a.size(0)):
            r = a[layer] @ r
        score = r[0]
    ids = enc["input_ids"][0].cpu().tolist()
    special = np.array(tok.get_special_tokens_mask(ids, already_has_special_tokens=True), dtype=bool)
    score = score.cpu().numpy().copy()
    score[special] = 0.0
    if score.max() > 0:
        score = score / score.max()
    probs = out.logits.softmax(-1)[0].cpu().numpy()
    return tok.convert_ids_to_tokens(ids), score, special, probs


def _clean(token):
    return token.replace("\u0120", "").replace("##", "").replace("\u2581", "") or token


def save_strip_png(tokens, scores, special, path, title, per_row=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = [i for i in range(len(tokens)) if not special[i]]
    rows = [idx[i:i + per_row] for i in range(0, len(idx), per_row)] or [[]]
    fig, ax = plt.subplots(figsize=(1.15 * per_row, 0.6 * len(rows) + 0.9))
    cmap = plt.get_cmap("Reds")
    for r, row in enumerate(rows):
        for c, i in enumerate(row):
            ax.text(c, -r, _clean(tokens[i]), ha="center", va="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", fc=cmap(0.05 + 0.9 * scores[i]), ec="none"))
    ax.set_xlim(-0.7, per_row - 0.3)
    ax.set_ylim(-len(rows) + 0.4, 0.6)
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_html(tokens, scores, special, path, title):
    spans = []
    for t, s, sp in zip(tokens, scores, special):
        if sp:
            continue
        spans.append(f'<span style="background:rgba(220,38,38,{0.08 + 0.85 * s:.2f});padding:2px 3px;margin:1px;'
                     f'border-radius:3px;display:inline-block">{html.escape(_clean(t))}</span>')
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"<html><meta charset='utf-8'><body style='font-family:sans-serif;max-width:900px;margin:2em auto'>"
                f"<h3>{html.escape(title)}</h3><p>{' '.join(spans)}</p></body></html>")


def explain_text(tok, model, text, device, out_dir, stem, method, max_len, names):
    tokens, scores, special, probs = token_importance(tok, model, text, device, max_len, method)
    pred = int(probs.argmax())
    title = f"predicted: {names[pred]} ({probs[pred]:.2f}) | importance: {method}"
    save_strip_png(tokens, scores, special, out_dir / f"{stem}.png", title)
    save_html(tokens, scores, special, out_dir / f"{stem}.html", title)
    top = sorted(((_clean(t), float(s)) for t, s, sp in zip(tokens, scores, special) if not sp),
                 key=lambda x: -x[1])[:8]
    print(f"{stem}: {title}\n   top tokens: {top}")


def aggregate(tok, model, texts, device, method, max_len, min_count=3, top_k=25):
    """Mean importance per token over prompts predicted as an attack."""
    acc, cnt = defaultdict(float), defaultdict(int)
    for text in texts:
        tokens, scores, special, probs = token_importance(tok, model, text, device, max_len, method)
        if int(probs.argmax()) == 0:
            continue
        for t, s, sp in zip(tokens, scores, special):
            if not sp:
                acc[_clean(t).lower()] += float(s)
                cnt[_clean(t).lower()] += 1
    rows = [(t, acc[t] / cnt[t], cnt[t]) for t in acc if cnt[t] >= min_count]
    return pd.DataFrame(sorted(rows, key=lambda r: -r[1])[:top_k], columns=["token", "mean_importance", "count"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--text", action="append", help="prompt to explain (repeatable)")
    ap.add_argument("--csv", help="CSV with a `text` column for aggregate analysis")
    ap.add_argument("--n", type=int, default=300, help="max rows from --csv")
    ap.add_argument("--method", choices=["rollout", "last"], default="rollout")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--binary", action="store_true", help="model was trained in binary mode")
    args = ap.parse_args()
    if not (args.text or args.csv):
        ap.error("provide --text and/or --csv")

    device = get_device()
    tok, model = load_model(args.model_dir, device)
    names = ["benign", "attack"] if args.binary else LABEL_NAMES
    out_dir = ensure_dir(RESULTS / "attention")

    for i, text in enumerate(args.text or []):
        explain_text(tok, model, text, device, out_dir, f"example_{i + 1}", args.method, args.max_len, names)

    if args.csv:
        df = pd.read_csv(args.csv, keep_default_na=False).head(args.n)
        top = aggregate(tok, model, df["text"].tolist(), device, args.method, args.max_len)
        top.to_csv(out_dir / "top_tokens.csv", index=False)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.barh(top["token"][::-1], top["mean_importance"][::-1], color="#c0392b")
        ax.set_xlabel("mean attention importance (attack predictions)")
        ax.set_title("Tokens driving attack detection")
        fig.tight_layout()
        fig.savefig(out_dir / "top_tokens.png", dpi=150)
        plt.close(fig)
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
