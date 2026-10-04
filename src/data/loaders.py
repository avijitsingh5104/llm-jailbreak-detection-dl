"""Reading splits and turning them into DataLoaders for every model family."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset

from src.config import BINARY_NAMES, CKPT, DATA_PROC, LABEL_NAMES
from src.text_utils import Vocab


def label_names(binary: bool):
    return BINARY_NAMES if binary else LABEL_NAMES


def read_split(name: str, binary: bool = False, limit: int = None, seed: int = 42) -> pd.DataFrame:
    path = DATA_PROC / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the data pipeline first (see README).")
    df = pd.read_csv(path, keep_default_na=False)
    df["text"] = df["text"].astype(str)
    if binary:
        df["label"] = (df["label"] > 0).astype(int)
    if limit and len(df) > limit:
        df = df.sample(limit, random_state=seed).reset_index(drop=True)
    return df


def read_novel(binary: bool = False, limit: int = None, seed: int = 42):
    """The held-out novel-phrasing test set (None if it has not been built yet)."""
    if not (DATA_PROC / "novel_test.csv").exists():
        return None
    return read_split("novel_test", binary, limit, seed)


def get_or_build_vocab(train_texts, level="word", max_size=20000, lowercase=False, rebuild=False) -> Vocab:
    """Vocabulary is built on TRAIN ONLY and cached so CNN and BiLSTM share it."""
    path = CKPT / f"vocab_{level}_{max_size}.json"
    if path.exists() and not rebuild:
        return Vocab.load(path)
    vocab = Vocab.build(train_texts, max_size=max_size, min_freq=2 if level == "word" else 1,
                        level=level, lowercase=lowercase)
    vocab.save(path)
    return vocab


def make_tensor_loader(df: pd.DataFrame, vocab: Vocab, max_len: int, batch_size: int, shuffle: bool) -> DataLoader:
    x = torch.from_numpy(vocab.encode_batch(df["text"].tolist(), max_len))
    y = torch.from_numpy(df["label"].values.astype(np.int64))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


# ---- sparse TF-IDF rows for the MLP (dense only one mini-batch at a time) ----
class SparseRowDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X.tocsr(), np.asarray(y, dtype=np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):  # idx is a list of row indices (see make_sparse_loader)
        return (torch.from_numpy(self.X[idx].toarray().astype(np.float32)), torch.from_numpy(self.y[idx]))


def make_sparse_loader(X, y, batch_size: int, shuffle: bool) -> DataLoader:
    ds = SparseRowDataset(X, y)
    sampler = RandomSampler(ds) if shuffle else SequentialSampler(ds)
    return DataLoader(ds, sampler=BatchSampler(sampler, batch_size, drop_last=False), batch_size=None)


# ---- raw text for Transformers (tokenised per batch by the collator) ----
class TextLabelDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts, self.labels = list(texts), list(map(int, labels))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]
