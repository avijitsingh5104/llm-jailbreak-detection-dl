"""Evaluation metrics shared by all models."""
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy plus macro-averaged precision / recall / F1."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(p), "recall": float(r), "f1": float(f)}


def security_metrics(y_true, y_pred) -> dict:
    """Binary guardrail view: any label > 0 counts as an attack.

    attack_detection_rate = share of attacks that were flagged (any attack class)
    false_positive_rate   = share of benign prompts wrongly flagged
    """
    t = np.asarray(y_true) > 0
    p = np.asarray(y_pred) > 0
    out = {"attack_detection_rate": None, "false_positive_rate": None}
    if t.any():
        out["attack_detection_rate"] = float((p & t).sum() / t.sum())
    if (~t).any():
        out["false_positive_rate"] = float((p & ~t).sum() / (~t).sum())
    return out


def detailed_report(y_true, y_pred, label_names) -> dict:
    labels = list(range(len(label_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rep = classification_report(y_true, y_pred, labels=labels, target_names=label_names,
                                output_dict=True, zero_division=0)
    return {"confusion_matrix": cm.tolist(), "per_class": rep}


def class_weights(labels, num_classes):
    """Inverse-frequency weights for the cross-entropy loss."""
    import torch
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    return torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32)


def plot_confusion_matrix(cm, label_names, path, title="Confusion matrix"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_names)), labels=label_names, rotation=30, ha="right")
    ax.set_yticks(range(len(label_names)), labels=label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thr = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
