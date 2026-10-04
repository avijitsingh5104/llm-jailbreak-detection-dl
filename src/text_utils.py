"""Light-weight text normalisation, tokenisation and vocabulary.

Design note: unlike sentiment analysis we do NOT strip punctuation, stopwords or
special characters. Encodings, homoglyphs, zero-width characters and odd
punctuation are themselves evidence of an obfuscation attack.
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def normalize(text) -> str:
    """NFC-normalise (NOT NFKC, which would erase full-width/homoglyph tricks),
    drop NULs and collapse whitespace."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFC", text).replace("\x00", "")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, level: str = "word", lowercase: bool = False):
    if lowercase:
        text = text.lower()
    if level == "char":
        return list(text)
    return _WORD_RE.findall(text)


class Vocab:
    def __init__(self, token2id: dict, level: str = "word", lowercase: bool = False):
        self.token2id = token2id
        self.level = level
        self.lowercase = lowercase

    def __len__(self):
        return len(self.token2id)

    @classmethod
    def build(cls, texts, max_size=20000, min_freq=2, level="word", lowercase=False):
        counter = Counter()
        for t in texts:
            counter.update(tokenize(t, level, lowercase))
        token2id = {PAD: PAD_ID, UNK: UNK_ID}
        for tok, freq in counter.most_common(max_size - len(token2id)):
            if freq < min_freq:
                break
            token2id[tok] = len(token2id)
        return cls(token2id, level, lowercase)

    def encode(self, text: str, max_len: int):
        ids = [self.token2id.get(t, UNK_ID) for t in tokenize(text, self.level, self.lowercase)][:max_len]
        return ids + [PAD_ID] * (max_len - len(ids))

    def encode_batch(self, texts, max_len: int) -> np.ndarray:
        return np.asarray([self.encode(t, max_len) for t in texts], dtype=np.int64)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"level": self.level, "lowercase": self.lowercase, "token2id": self.token2id},
                      f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["token2id"], d["level"], d["lowercase"])
