import numpy as np
import pandas as pd

from src.data.split import assign_folds, group_key
from src.text_utils import PAD_ID, UNK_ID, Vocab, normalize, tokenize


def test_normalize_keeps_obfuscation_signals():
    s = "  Hello\u200b   world \x00!\n"
    out = normalize(s)
    assert out == "Hello\u200b world !"          # zero-width char kept, whitespace collapsed
    assert normalize("\uff49\uff47") == "\uff49\uff47"   # full-width NOT folded (NFC, not NFKC)


def test_tokenize_levels():
    assert tokenize("Ignore it!", "word") == ["Ignore", "it", "!"]
    assert tokenize("ab c", "char") == ["a", "b", " ", "c"]


def test_vocab_encode_pad_and_unk(tmp_path):
    v = Vocab.build(["a b a b c", "a b"], max_size=10, min_freq=2)
    ids = v.encode("a z b", 5)
    assert len(ids) == 5 and ids[1] == UNK_ID and ids[3:] == [PAD_ID, PAD_ID]
    v.save(tmp_path / "v.json")
    assert Vocab.load(tmp_path / "v.json").token2id == v.token2id


def test_split_has_no_group_leakage():
    rng = np.random.default_rng(0)
    rows = []
    for g in range(300):                       # 300 template families, each with 3 near-copies
        label = int(rng.integers(0, 3))
        stem = f"family {g} unique stem text for grouping purposes here {g * 7919}"
        for k in range(3):
            rows.append({"text": stem + f" variant {k}", "label": label})
    df = pd.DataFrame(rows)
    fold = assign_folds(df, 20, 0)
    keys = df["text"].map(group_key)
    for key in keys.unique():
        assert len(set(fold[(keys == key).values])) == 1
