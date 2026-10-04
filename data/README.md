# Data

Nothing in `raw/` or `processed/` is committed (see `.gitignore`); everything is rebuilt by the pipeline.

| Source | Hugging Face id | Used for |
|---|---|---|
| In-The-Wild Jailbreak Prompts | `TrustAIRLab/in-the-wild-jailbreak-prompts` (configs `jailbreak_2023_12_25`, `regular_2023_12_25`) | jailbreak (1) and regular user prompts (0) |
| safe-guard-prompt-injection | `xTRam1/safe-guard-prompt-injection` | injection (2) and benign (0) |
| HackAPrompt | `hackaprompt/hackaprompt-dataset` (gated: accept the terms and run `huggingface-cli login`) | injection (2) |
| Alpaca | `tatsu-lab/alpaca` | benign (0) |
| No Robots | `HuggingFaceH4/no_robots` | benign (0) |

## Offline / manual download

Put a CSV with columns `text,label` (labels in the project scheme: 0 benign, 1 jailbreak, 2 injection) at
`data/raw/<name>.csv` where `<name>` is one of `wild`, `safeguard`, `hackaprompt`, `alpaca`, `no_robots`.
A local file takes priority over the Hugging Face download for that source.

## Folders

* `raw/` – optional manual CSVs
* `processed/consolidated.csv` – cleaned, de-duplicated corpus
* `processed/base/` – train/val/test before obfuscation augmentation
* `processed/{train,val,test}.csv` – final splits (with label 3)
* `processed/novel_test.csv` – held-out novel-phrasing test set
* `novel/novel_manual.csv` – **hand-curated** novel prompts (copy `novel_manual_template.csv`, add your own)
