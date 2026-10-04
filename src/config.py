"""Project-wide constants shared by every model so the comparison stays fair."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_BASE = DATA_PROC / "base"      # splits before obfuscation augmentation
DATA_NOVEL = ROOT / "data" / "novel"
RESULTS = ROOT / "results"
CKPT = ROOT / "checkpoints"

# Label scheme (4-class). The binary view is: 0 = benign, 1 = any attack (label > 0).
LABELS = {0: "benign", 1: "jailbreak", 2: "prompt_injection", 3: "obfuscation"}
LABEL_NAMES = [LABELS[i] for i in range(len(LABELS))]
BINARY_NAMES = ["benign", "attack"]

SEED = 42
MAX_LEN = 200          # tokens for word-level CNN / BiLSTM
VOCAB_SIZE = 20000
