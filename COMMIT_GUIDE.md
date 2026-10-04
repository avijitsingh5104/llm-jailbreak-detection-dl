# Commit guide (member-wise)

**Each member should commit their own files from their own GitHub account** so the history shows who did what.
Set your identity once in your clone: `git config user.name "Your Name"` and `git config user.email "your-github-email"`.

**Order:** Member 1 pushes first (the others import his shared code). Members 2 and 3 can then work in parallel.
Member 4 goes last (needs the other models' `results/*.json` for the comparison).

```bash
git clone <repo-url> && cd <repo>          # Members 2-4; Member 1 runs `git init` and adds the remote
git pull                                    # always pull before you push
```

---------------------------------------------------------------------

## Member 1 – Avijit Singh (230911310): setup, data pipeline, shared infra, MLP

```bash
# commit 1
git add README.md requirements.txt .gitignore COMMIT_GUIDE.md data/README.md \
        data/raw/.gitkeep data/processed/.gitkeep checkpoints/.gitkeep results/.gitkeep \
        src/__init__.py src/data/__init__.py src/models/__init__.py src/explain/__init__.py \
        src/config.py src/utils.py
git commit -m "Initial project setup (folder structure, README, requirements, .gitignore)"

# commit 2
git add src/text_utils.py src/data/consolidate.py src/data/split.py src/data/loaders.py \
        src/data/make_toy_data.py tests/test_preprocessing.py
git commit -m "Add data preprocessing pipeline (consolidation, cleaning, leakage-aware stratified split)"

# commit 3
git add src/metrics.py src/trainer.py src/experiment.py
git commit -m "Add shared metrics, training loop and experiment runner"

# commit 4
git add src/models/mlp.py
git commit -m "Add MLP model architecture"

# commit 5
git add src/train_mlp.py
git commit -m "Add MLP TF-IDF training and evaluation script"

# commit 6 (after running it on the real data)
git add results/mlp.json results/mlp_cm_test.png results/mlp_cm_novel.png
git commit -m "Add MLP results"
git push -u origin main
```

## Member 2 – Hansika Abhiraj (230911128): obfuscation class + CNN

```bash
git add src/data/augment_obfuscation.py tests/test_augment.py
git commit -m "Add obfuscation/encoding attack augmentation (train and held-out transforms)"

git add src/models/cnn.py
git commit -m "Add TextCNN model architecture"

git add src/train_cnn.py
git commit -m "Add TextCNN training and evaluation script (word and char level)"

# after running word and char versions
git add results/cnn.json results/cnn_char.json results/cnn_cm_test.png results/cnn_cm_novel.png \
        results/cnn_char_cm_test.png results/cnn_char_cm_novel.png
git commit -m "Add CNN results"
git push
```

## Member 3 – Raghav Khetarpal (230911266): BiLSTM + novel-phrasing test set

```bash
git add src/models/bilstm.py
git commit -m "Add BiLSTM model architecture with attention pooling"

git add src/train_bilstm.py
git commit -m "Add BiLSTM training script with gradient clipping"

# your hand-written prompts: copy the template to data/novel/novel_manual.csv and add real rows first
git add src/data/build_novel_test.py data/novel/novel_manual_template.csv data/novel/novel_manual.csv \
        tests/test_novel_filter.py
git commit -m "Add novel-phrasing generalization test-set builder with similarity filter"

git add results/bilstm.json results/bilstm_cm_test.png results/bilstm_cm_novel.png
git commit -m "Add BiLSTM results"
git push
```
(Skip `data/novel/novel_manual.csv` in the `git add` until the file exists.)

## Member 4 – Aniket Sahu (230953336): Transformer, attention visualization, comparison

```bash
git add src/models/transformer.py
git commit -m "Add fine-tuned Transformer model wrapper and tokenisation collator"

git add src/train_transformer.py
git commit -m "Add Transformer fine-tuning and evaluation script"

git add src/explain/attention_viz.py
git commit -m "Add attention rollout visualisation and aggregate token-importance analysis"

git add src/compare.py
git commit -m "Add comparison script (results table and report figures)"

# after training and running the analysis
git add results/transformer.json results/transformer_cm_test.png results/transformer_cm_novel.png \
        results/attention results/results_table.md results/results_table.csv \
        results/fig_performance.png results/fig_training_time.png results/fig_generalization.png
git commit -m "Add Transformer results, attention analysis and final comparison"
git push
```

---------------------------------------------------------------------

## Who owns which file

| File | Owner |
|---|---|
| README.md, requirements.txt, .gitignore, COMMIT_GUIDE.md, data/README.md, .gitkeep files | Member 1 |
| src/config.py, utils.py, text_utils.py, metrics.py, trainer.py, experiment.py | Member 1 |
| src/data/consolidate.py, split.py, loaders.py, make_toy_data.py, tests/test_preprocessing.py | Member 1 |
| src/models/mlp.py, src/train_mlp.py | Member 1 |
| src/data/augment_obfuscation.py, tests/test_augment.py | Member 2 |
| src/models/cnn.py, src/train_cnn.py | Member 2 |
| src/models/bilstm.py, src/train_bilstm.py | Member 3 |
| src/data/build_novel_test.py, data/novel/*, tests/test_novel_filter.py | Member 3 |
| src/models/transformer.py, src/train_transformer.py | Member 4 |
| src/explain/attention_viz.py, src/compare.py | Member 4 |
| `__init__.py` files | Member 1 |

## Before you commit
* Run `python -m pytest -q` (should show 9 passed).
* Never commit `data/raw`, `data/processed` or `checkpoints/` (already git-ignored).
* Only commit results produced from the **real** data, never from the toy smoke test.
