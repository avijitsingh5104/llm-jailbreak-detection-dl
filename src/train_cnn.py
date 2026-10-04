"""Train / evaluate the TextCNN (word- or character-level).

Usage:
    python -m src.train_cnn                                   # word level
    python -m src.train_cnn --level char --max-len 512        # character level (robust to obfuscation)
"""
import argparse

from src.config import MAX_LEN, VOCAB_SIZE
from src.data.loaders import get_or_build_vocab, label_names, make_tensor_loader, read_novel, read_split
from src.experiment import add_common_args, run_experiment
from src.models.cnn import TextCNN
from src.utils import get_device, set_seed


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__), epochs=15, batch_size=64)
    ap.add_argument("--level", choices=["word", "char"], default="word")
    ap.add_argument("--max-len", type=int, default=None, help=f"default {MAX_LEN} (word) / 512 (char)")
    ap.add_argument("--vocab-size", type=int, default=None, help=f"default {VOCAB_SIZE} (word) / 5000 (char)")
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--kernel-sizes", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--filters", type=int, default=100)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--rebuild-vocab", action="store_true")
    args = ap.parse_args()
    max_len = args.max_len or (MAX_LEN if args.level == "word" else 512)
    vocab_size = args.vocab_size or (VOCAB_SIZE if args.level == "word" else 5000)
    set_seed(args.seed)
    device = get_device()

    tr = read_split("train", args.binary, args.limit, args.seed)
    va = read_split("val", args.binary, args.limit, args.seed)
    te = read_split("test", args.binary, args.limit, args.seed)
    nv = read_novel(args.binary)

    vocab = get_or_build_vocab(tr["text"], args.level, vocab_size, rebuild=args.rebuild_vocab)
    print(f"vocabulary size ({args.level}-level): {len(vocab)}, max_len {max_len}")
    loaders = {"train": make_tensor_loader(tr, vocab, max_len, args.batch_size, True),
               "val": make_tensor_loader(va, vocab, max_len, 256, False),
               "test": make_tensor_loader(te, vocab, max_len, 256, False)}
    if nv is not None:
        loaders["novel"] = make_tensor_loader(nv, vocab, max_len, 256, False)

    model = TextCNN(len(vocab), len(label_names(args.binary)), args.embed_dim, tuple(args.kernel_sizes),
                    args.filters, args.dropout)
    name = "cnn" if args.level == "word" else "cnn_char"
    run_experiment(name, model, loaders, tr["label"].values, args, device,
                   config={"level": args.level, "max_len": max_len, "vocab": len(vocab)})


if __name__ == "__main__":
    main()
