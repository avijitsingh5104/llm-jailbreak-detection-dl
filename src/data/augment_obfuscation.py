"""Create the obfuscation / encoding-attack class (label 3).

A fraction of the jailbreak (1) and injection (2) prompts in EACH split is
re-written with an obfuscation transform and added as label 3. Splits are
augmented independently, so no obfuscated copy of a training prompt can appear
in val/test. Transforms are divided into TRAIN transforms (used here) and
HELDOUT transforms (only used by build_novel_test.py to test generalization to
encodings never seen in training).

Reads  data/processed/base/{train,val,test}.csv
Writes data/processed/{train,val,test}.csv
Usage: python -m src.data.augment_obfuscation --frac 0.15
"""
import argparse
import base64
import codecs
import random

import pandas as pd

from src.config import DATA_BASE, DATA_PROC
from src.utils import ensure_dir, set_seed

# ------------------------------------------------------------------ transforms
_LEET = str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7",
                       "A": "4", "E": "3", "I": "1", "O": "0", "S": "5", "T": "7"})
_HOMOGLYPH = {"a": "\u0430", "e": "\u0435", "o": "\u043e", "p": "\u0440", "c": "\u0441",
              "x": "\u0445", "y": "\u0443", "i": "\u0456"}
_B64_WRAPPERS = ["Decode the following base64 string and do what it says: {x}",
                 "The next message is base64 encoded. Decode it and follow it exactly: {x}",
                 "{x}"]


def base64_encode(text, rng):
    enc = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return rng.choice(_B64_WRAPPERS).format(x=enc)


def rot13(text, rng):
    return codecs.encode(text, "rot_13")


def leetspeak(text, rng):
    return text.translate(_LEET)


def char_spacing(text, rng):
    words = [" ".join(w) if len(w) > 3 and rng.random() < 0.7 else w for w in text.split(" ")]
    return " ".join(words)


def homoglyph(text, rng):
    return "".join(_HOMOGLYPH[c] if c in _HOMOGLYPH and rng.random() < 0.6 else c for c in text)


def zero_width(text, rng):
    return "".join(c + ("\u200b" if c.isalpha() and rng.random() < 0.35 else "") for c in text)


TRAIN_TRANSFORMS = {"base64": base64_encode, "rot13": rot13, "leetspeak": leetspeak,
                    "char_spacing": char_spacing, "homoglyph": homoglyph, "zero_width": zero_width}


def hex_encode(text, rng):
    return "Decode this hex and follow it: " + text.encode("utf-8").hex()


def reverse_text(text, rng):
    return text[::-1]


def fullwidth(text, rng):
    return "".join(chr(ord(c) + 0xFEE0) if 0x21 <= ord(c) <= 0x7E else c for c in text)


def caesar3(text, rng):
    def sh(c):
        if "a" <= c <= "z":
            return chr((ord(c) - 97 + 3) % 26 + 97)
        if "A" <= c <= "Z":
            return chr((ord(c) - 65 + 3) % 26 + 65)
        return c
    return "".join(sh(c) for c in text)


def pig_latin(text, rng):
    out = []
    for w in text.split(" "):
        core = "".join(ch for ch in w if ch.isalpha())
        if len(core) > 1 and core.isascii():
            out.append(core[1:] + core[0] + "ay")
        else:
            out.append(w)
    return " ".join(out)


HELDOUT_TRANSFORMS = {"hex": hex_encode, "reverse": reverse_text, "fullwidth": fullwidth,
                      "caesar3": caesar3, "pig_latin": pig_latin}


# ------------------------------------------------------------------ dataframe op
def augment_dataframe(df: pd.DataFrame, transforms: dict, frac: float, seed: int = 42,
                      max_chars: int = 600, source_labels=(1, 2)) -> pd.DataFrame:
    """Append obfuscated copies (label 3) of `frac` of the attack rows."""
    if frac <= 0:
        return df.copy()
    rng = random.Random(seed)
    attacks = df[df["label"].isin(source_labels)]
    sel = attacks.sample(n=int(len(attacks) * frac), random_state=seed)
    names = sorted(transforms)
    rows = []
    for text in sel["text"]:
        name = rng.choice(names)
        rows.append({"text": transforms[name](text[:max_chars], rng), "label": 3, "source": f"obfuscated:{name}"})
    out = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frac", type=float, default=0.15, help="fraction of attack rows to obfuscate per split")
    ap.add_argument("--max-chars", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    ensure_dir(DATA_PROC)
    for i, split in enumerate(("train", "val", "test")):
        df = pd.read_csv(DATA_BASE / f"{split}.csv", keep_default_na=False)
        out = augment_dataframe(df, TRAIN_TRANSFORMS, args.frac, args.seed + i, args.max_chars)
        out.to_csv(DATA_PROC / f"{split}.csv", index=False)
        print(f"{split:5s} {len(df):7d} -> {len(out):7d} rows  (obfuscation: {(out['label'] == 3).sum()})")


if __name__ == "__main__":
    main()
