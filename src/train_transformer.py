"""Fine-tune DistilBERT / RoBERTa-base and evaluate like every other model.

Usage:
    python -m src.train_transformer                                   # distilbert-base-uncased
    python -m src.train_transformer --model roberta-base --batch-size 16 --amp
The fine-tuned model + tokenizer are saved to checkpoints/<name>_best/ for attention_viz.
"""
import argparse

from torch.utils.data import DataLoader

from src.config import CKPT
from src.data.loaders import TextLabelDataset, label_names, read_novel, read_split
from src.experiment import add_common_args, run_experiment
from src.models.transformer import HFClassifier, TokenizeCollator
from src.utils import get_device, set_seed


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__), epochs=3, batch_size=16,
                         lr=2e-5, weight_decay=0.01, patience=2)
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--warmup", type=float, default=0.1, help="warm-up fraction of total steps")
    ap.add_argument("--name", default="transformer")
    args = ap.parse_args()
    set_seed(args.seed)
    device = get_device()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    collate = TokenizeCollator(tok, args.max_len)

    tr = read_split("train", args.binary, args.limit, args.seed)
    va = read_split("val", args.binary, args.limit, args.seed)
    te = read_split("test", args.binary, args.limit, args.seed)
    nv = read_novel(args.binary)

    mk = lambda df, bs, sh: DataLoader(TextLabelDataset(df["text"], df["label"]), batch_size=bs, shuffle=sh,
                                       collate_fn=collate)
    loaders = {"train": mk(tr, args.batch_size, True), "val": mk(va, 64, False), "test": mk(te, 64, False)}
    if nv is not None:
        loaders["novel"] = mk(nv, 64, False)

    model = HFClassifier(args.model, len(label_names(args.binary)))
    run_experiment(args.name, model, loaders, tr["label"].values, args, device, warmup_ratio=args.warmup,
                   save_ckpt=False, config={"pretrained": args.model, "max_len": args.max_len})

    out = CKPT / f"{args.name}{'_binary' if args.binary else ''}_best"
    model.model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"fine-tuned model saved to {out}")


if __name__ == "__main__":
    main()
