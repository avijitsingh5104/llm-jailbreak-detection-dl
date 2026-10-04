"""Train / evaluate the TF-IDF MLP baseline.

Usage:  python -m src.train_mlp [--binary] [--epochs 20] [--hidden 512 128]
"""
import argparse

import joblib
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import CKPT
from src.data.loaders import label_names, make_sparse_loader, read_novel, read_split
from src.experiment import add_common_args, run_experiment
from src.models.mlp import MLPClassifier
from src.utils import ensure_dir, get_device, set_seed


def build_vectorizers(args):
    word = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b|[^\w\s]", ngram_range=(1, 2),
                           lowercase=False, sublinear_tf=True, min_df=2, max_features=args.max_features_word)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False, sublinear_tf=True,
                           min_df=3, max_features=args.max_features_char)
    return word, char


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__), epochs=20, batch_size=128,
                         lr=1e-3, weight_decay=1e-4, patience=3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 128])
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--max-features-word", type=int, default=20000)
    ap.add_argument("--max-features-char", type=int, default=30000)
    ap.add_argument("--max-chars", type=int, default=3000, help="truncate very long prompts before TF-IDF")
    args = ap.parse_args()
    set_seed(args.seed)
    device = get_device()

    tr = read_split("train", args.binary, args.limit, args.seed)
    va = read_split("val", args.binary, args.limit, args.seed)
    te = read_split("test", args.binary, args.limit, args.seed)
    nv = read_novel(args.binary)

    cut = lambda df: df["text"].str.slice(0, args.max_chars)
    word, char = build_vectorizers(args)
    # fit on TRAIN only
    Xtr = hstack([word.fit_transform(cut(tr)), char.fit_transform(cut(tr))]).tocsr()
    feats = lambda df: hstack([word.transform(cut(df)), char.transform(cut(df))]).tocsr()
    print(f"TF-IDF feature dimension: {Xtr.shape[1]}")
    ensure_dir(CKPT)
    joblib.dump({"word": word, "char": char}, CKPT / "mlp_tfidf.joblib")

    loaders = {"train": make_sparse_loader(Xtr, tr["label"].values, args.batch_size, True),
               "val": make_sparse_loader(feats(va), va["label"].values, 256, False),
               "test": make_sparse_loader(feats(te), te["label"].values, 256, False)}
    if nv is not None:
        loaders["novel"] = make_sparse_loader(feats(nv), nv["label"].values, 256, False)

    model = MLPClassifier(Xtr.shape[1], len(label_names(args.binary)), tuple(args.hidden), args.dropout)
    run_experiment("mlp", model, loaders, tr["label"].values, args, device,
                   config={"in_dim": Xtr.shape[1], "hidden": args.hidden, "dropout": args.dropout})


if __name__ == "__main__":
    main()
