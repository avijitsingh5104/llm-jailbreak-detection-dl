"""Train / evaluate the BiLSTM (attention pooling, gradient clipping).

Usage:  python -m src.train_bilstm [--pooling attn|max|last] [--hidden 128] [--binary]
"""
import argparse

from src.config import MAX_LEN, VOCAB_SIZE
from src.data.loaders import get_or_build_vocab, label_names, make_tensor_loader, read_novel, read_split
from src.experiment import add_common_args, run_experiment
from src.models.bilstm import BiLSTMClassifier
from src.utils import get_device, set_seed


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__), epochs=15, batch_size=64)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--pooling", choices=["attn", "max", "last"], default="attn")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--rebuild-vocab", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)
    device = get_device()

    tr = read_split("train", args.binary, args.limit, args.seed)
    va = read_split("val", args.binary, args.limit, args.seed)
    te = read_split("test", args.binary, args.limit, args.seed)
    nv = read_novel(args.binary)

    vocab = get_or_build_vocab(tr["text"], "word", args.vocab_size, rebuild=args.rebuild_vocab)
    print(f"vocabulary size: {len(vocab)}, max_len {args.max_len}")
    loaders = {"train": make_tensor_loader(tr, vocab, args.max_len, args.batch_size, True),
               "val": make_tensor_loader(va, vocab, args.max_len, 256, False),
               "test": make_tensor_loader(te, vocab, args.max_len, 256, False)}
    if nv is not None:
        loaders["novel"] = make_tensor_loader(nv, vocab, args.max_len, 256, False)

    model = BiLSTMClassifier(len(vocab), len(label_names(args.binary)), args.embed_dim, args.hidden,
                             args.layers, args.dropout, args.pooling)
    run_experiment("bilstm", model, loaders, tr["label"].values, args, device, clip=args.clip,
                   config={"max_len": args.max_len, "vocab": len(vocab), "pooling": args.pooling,
                           "hidden": args.hidden, "layers": args.layers})


if __name__ == "__main__":
    main()
