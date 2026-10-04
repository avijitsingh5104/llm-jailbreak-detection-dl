"""Model-agnostic training / prediction loop (works for tensors and HF-style dict batches)."""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from src.metrics import class_weights, compute_metrics


def to_device(x, device):
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)


@torch.no_grad()
def predict(model, loader, device):
    """Returns (y_true, y_pred, elapsed_seconds)."""
    model.eval()
    ys, ps = [], []
    t0 = time.perf_counter()
    for x, y in loader:
        logits = model(to_device(x, device))
        ps.append(logits.argmax(-1).cpu().numpy())
        ys.append(y.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return np.concatenate(ys), np.concatenate(ps), time.perf_counter() - t0


def _make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def fit(model, train_loader, val_loader, train_labels, num_classes, device, epochs=15, lr=1e-3,
        weight_decay=1e-4, patience=3, clip=1.0, use_class_weights=True, warmup_ratio=0.0,
        amp=False, ckpt_path=None, log=print):
    """Train with AdamW + class-weighted CE; keep the epoch with the best validation macro-F1.

    Returns (history, train_seconds). The best weights are restored into `model`.
    """
    model.to(device)
    w = class_weights(train_labels, num_classes).to(device) if use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total = max(1, epochs * len(train_loader))
    warm = int(warmup_ratio * total)
    sched = None
    if warmup_ratio > 0:
        sched = LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))

    use_amp = bool(amp and device.type == "cuda")
    scaler = _make_scaler(use_amp)
    best_f1, best_state, bad, history = -1.0, None, 0, []
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for x, y in train_loader:
            x, y = to_device(x, device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            if clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
            if sched:
                sched.step()
            running += loss.item() * len(y)
            n += len(y)

        yt, yp, _ = predict(model, val_loader, device)
        m = compute_metrics(yt, yp)
        history.append({"epoch": epoch, "train_loss": running / max(1, n), **{f"val_{k}": v for k, v in m.items()}})
        log(f"epoch {epoch:02d}  loss {running / max(1, n):.4f}  val_acc {m['accuracy']:.4f}  val_f1 {m['f1']:.4f}")

        if m["f1"] > best_f1:
            best_f1, bad = m["f1"], 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                log(f"early stopping at epoch {epoch} (best val_f1 {best_f1:.4f})")
                break

    secs = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
        if ckpt_path:
            torch.save(best_state, ckpt_path)
    return history, secs
