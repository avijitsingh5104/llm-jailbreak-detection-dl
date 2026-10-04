"""Shared experiment runner: train, evaluate on val / test / novel set, write results JSON + plots."""
import argparse

from src.config import CKPT, RESULTS
from src.data.loaders import label_names
from src.metrics import compute_metrics, detailed_report, plot_confusion_matrix, security_metrics
from src.trainer import fit, predict
from src.utils import count_params, ensure_dir, save_json


def add_common_args(p: argparse.ArgumentParser, epochs=15, batch_size=64, lr=1e-3, weight_decay=1e-4, patience=3):
    p.add_argument("--binary", action="store_true", help="benign vs attack instead of 4 classes")
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--batch-size", type=int, default=batch_size)
    p.add_argument("--lr", type=float, default=lr)
    p.add_argument("--weight-decay", type=float, default=weight_decay)
    p.add_argument("--patience", type=int, default=patience)
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--limit", type=int, default=None, help="subsample each split (smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    return p


def run_experiment(name, model, loaders, train_labels, args, device, config=None, warmup_ratio=0.0,
                   clip=1.0, save_ckpt=True):
    """loaders: dict with 'train', 'val', 'test' and optionally 'novel'."""
    names = label_names(args.binary)
    k = len(names)
    tag = f"{name}_binary" if args.binary else name
    ensure_dir(RESULTS)
    ensure_dir(CKPT)

    print(f"\n=== {tag}: {count_params(model):,} trainable parameters, device={device} ===")
    history, secs = fit(model, loaders["train"], loaders["val"], train_labels, k, device,
                        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                        patience=args.patience, clip=clip, use_class_weights=not args.no_class_weights,
                        warmup_ratio=warmup_ratio, amp=args.amp,
                        ckpt_path=(CKPT / f"{tag}.pt") if save_ckpt else None)

    results = {"model": name, "mode": "binary" if args.binary else "multiclass",
               "params": count_params(model), "train_time_min": secs / 60.0,
               "epochs_run": len(history), "history": history,
               "config": {**(config or {}), **{k_: v for k_, v in vars(args).items()}}}

    for split in ("val", "test", "novel"):
        if split not in loaders:
            continue
        yt, yp, elapsed = predict(model, loaders[split], device)
        m = compute_metrics(yt, yp)
        m.update(security_metrics(yt, yp))
        m["n"] = int(len(yt))
        m["ms_per_prompt"] = 1000.0 * elapsed / max(1, len(yt))
        m.update(detailed_report(yt, yp, names))
        results[split] = m
        print(f"{split:5s}  acc {m['accuracy']:.4f}  P {m['precision']:.4f}  R {m['recall']:.4f}  F1 {m['f1']:.4f}"
              f"  detect {m['attack_detection_rate']}  FPR {m['false_positive_rate']}")
        if split in ("test", "novel"):
            plot_confusion_matrix(m["confusion_matrix"], names, RESULTS / f"{tag}_cm_{split}.png",
                                  f"{tag} - {split}")

    if "novel" in results:
        results["generalization_gap_f1"] = results["test"]["f1"] - results["novel"]["f1"]
        print(f"generalization gap (test F1 - novel F1): {results['generalization_gap_f1']:.4f}")
    else:
        print("novel_test.csv not found - skipping generalization evaluation")

    save_json(results, RESULTS / f"{tag}.json")
    print(f"train time {secs / 60:.2f} min; results -> {RESULTS / (tag + '.json')}")
    return results
