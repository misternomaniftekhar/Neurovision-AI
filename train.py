#!/usr/bin/env python
"""Train an EfficientNet-B0 MRI classifier on a Kaggle brain-tumour dataset (one folder per class).

Accepted layouts (auto-detected):
    DATA/Training/<class>/*.jpg  +  DATA/Testing/<class>/*.jpg     (Kaggle "Brain Tumor MRI Dataset")
    DATA/<class>/*.jpg                                              (single folder -> 70/15/15 split)

Example:
    python train.py --data /kaggle/input/brain-tumor-mri-dataset --epochs 15
Outputs: models/model.pt, models/meta.json, reports/{metrics.json,confusion_matrix.png,training_curves.png}
"""
import argparse
import datetime
import json
import os
import random
import time

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

import model_utils as mu

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def class_dirs(path):
    return [d for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, d))]


def find_splits(root):
    """(train_dir, test_dir or None). Handles Training/Testing, train/test, nested folders, or flat class folders."""
    subs = {d.lower(): os.path.join(root, d) for d in class_dirs(root)}
    for tr, te in (("training", "testing"), ("train", "test"), ("train", "val"), ("train", "valid")):
        if tr in subs and te in subs:
            return subs[tr], subs[te]
    for tr in ("training", "train"):
        if tr in subs:
            return subs[tr], None
    if len(subs) == 1:
        return find_splits(next(iter(subs.values())))
    return root, None


def fingerprints(paths, size=32):
    """Tiny normalised grayscale vector per image; cosine similarity ~1.0 means the same picture."""
    out = np.zeros((len(paths), size * size), dtype=np.float32)
    for k, p in enumerate(paths):
        with Image.open(p) as im:
            im.draft("L", (4 * size, 4 * size))
            v = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32).ravel()
        v = v - v.mean()
        n = np.linalg.norm(v)
        out[k] = v / n if n > 0 else v
    return out


def _groups(X, thr):
    """Union-find over image pairs with similarity >= thr (chunked, memory-safe). Returns group id per image."""
    parent = list(range(len(X)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(0, len(X), 1024):
        S = X[i:i + 1024] @ X.T
        for r, c in np.argwhere(S >= thr):
            if c > i + r:
                ra, rb = find(i + r), find(int(c))
                if ra != rb:
                    parent[rb] = ra
    return np.array([find(a) for a in range(len(X))])


def near_dup_filter(train_paths, train_targets, test_paths=None, test_targets=None, thr=0.999):
    """Remove duplicate pictures so validation/test scores are not inflated.
    - Within the training pool: keep one copy per duplicate group; drop whole groups whose copies carry different labels.
    - Test images that duplicate any training image (or each other) are dropped.
    Returns (kept_train_idx, kept_test_idx or None, stats)."""
    Xtr = fingerprints(train_paths)
    g = _groups(Xtr, thr)
    labels = {}
    for i, gid in enumerate(g):
        labels.setdefault(int(gid), []).append(train_targets[i])
    keep, seen, conflict = [], set(), 0
    for i, gid in enumerate(g):
        gid = int(gid)
        if len(set(labels[gid])) > 1:
            conflict += 1
        elif gid not in seen:
            seen.add(gid)
            keep.append(i)
    stats = {"train_before": len(train_paths), "train_after": len(keep), "train_label_conflict_images": conflict}
    kept_test = None
    if test_paths is not None:
        Xte = fingerprints(test_paths)
        max_sim = np.max([(Xte @ Xtr[i:i + 2048].T).max(1) for i in range(0, len(Xtr), 2048)], axis=0)
        gt = _groups(Xte, thr)
        seen_t, kept_test = set(), []
        for i, gid in enumerate(gt):
            if max_sim[i] >= thr or int(gid) in seen_t:
                continue
            seen_t.add(int(gid))
            kept_test.append(i)
        stats.update(test_before=len(test_paths), test_after=len(kept_test),
                     test_leaky_with_train=int((max_sim >= thr).sum()))
    return keep, kept_test, stats


def limit_per_class(idx, targets, n, rng):
    if not n:
        return list(idx)
    out = []
    for c in sorted(set(targets[i] for i in idx)):
        members = [i for i in idx if targets[i] == c]
        rng.shuffle(members)
        out += members[:n]
    return out


def split(idx, targets, frac, seed):
    return train_test_split(idx, test_size=frac, stratify=[targets[i] for i in idx], random_state=seed)


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(logits), torch.cat(ys)


def fit_temperature(logits, labels):
    """Temperature scaling on validation logits (calibrates confidence, never changes the predicted class)."""
    T = nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=0.05), labels)
        loss.backward()
        return loss

    opt.step(closure)
    # keep within a sane range: a tiny/perfectly-classified validation set can push T to extremes
    return float(min(max(T.item(), 0.5), 5.0))


def ece(probs, labels, bins=10):
    conf, pred = probs.max(1)
    acc = (pred == labels).float()
    edges, e = torch.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += float(m.float().mean() * (acc[m].mean() - conf[m].mean()).abs())
    return e


def plot_confusion(cm, names, path):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_curves(hist, path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(hist["train_loss"], label="train")
    ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(hist["val_f1"], label="val macro-F1", color="tab:green")
    ax[1].plot(hist["val_acc"], label="val accuracy", color="tab:orange")
    ax[1].set_title("Validation")
    ax[1].legend()
    for a in ax:
        a.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--img-size", type=int, default=mu.IMG_SIZE)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (offline / smoke tests)")
    ap.add_argument("--no-dedupe", action="store_true", help="skip near-duplicate/leakage filtering")
    ap.add_argument("--max-per-class", type=int, default=0, help="subsample for quick tests")
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.reports, exist_ok=True)

    train_dir, test_dir = find_splits(a.data)
    print(f"train dir: {train_dir}\ntest dir : {test_dir}\ndevice   : {device}")
    ds_train = datasets.ImageFolder(train_dir, mu.train_transform(a.img_size))
    ds_eval = datasets.ImageFolder(train_dir, mu.eval_transform(a.img_size))
    classes, targets = ds_train.classes, ds_train.targets
    print("classes  :", classes)

    all_idx = list(range(len(targets)))
    ds_test, dedupe_stats = None, {"enabled": not a.no_dedupe}
    if test_dir:
        ds_test = datasets.ImageFolder(test_dir, mu.eval_transform(a.img_size))
        if ds_test.classes != classes:
            raise SystemExit(f"Class folders differ: train={classes} test={ds_test.classes}")
    kept_test = None
    if not a.no_dedupe:
        print("checking for duplicate images (takes a minute)...")
        all_idx, kept_test, st = near_dup_filter([p for p, _ in ds_train.samples], targets,
                                                 [p for p, _ in ds_test.samples] if ds_test else None)
        dedupe_stats.update(st)
        print("dedupe:", st)
    if test_dir:
        idx_test = limit_per_class(kept_test if kept_test is not None else range(len(ds_test)),
                                   ds_test.targets, a.max_per_class, rng)
        idx_pool = all_idx
    else:
        idx_pool, idx_t = split(all_idx, targets, 0.15, a.seed)
        ds_test, idx_test = ds_eval, idx_t
    idx_pool = limit_per_class(idx_pool, targets, a.max_per_class * 3 if a.max_per_class else 0, rng)
    idx_tr, idx_val = split(idx_pool, targets, a.val_frac, a.seed)
    counts = np.bincount([targets[i] for i in idx_tr], minlength=len(classes))
    print(f"train {len(idx_tr)} | val {len(idx_val)} | test {len(idx_test)} | train counts {dict(zip(classes, counts.tolist()))}")

    kw = dict(batch_size=a.batch_size, num_workers=a.workers, pin_memory=device.type == "cuda")
    dl_tr = DataLoader(Subset(ds_train, idx_tr), shuffle=True, drop_last=len(idx_tr) > a.batch_size, **kw)
    dl_val = DataLoader(Subset(ds_eval, idx_val), **kw)
    dl_te = DataLoader(Subset(ds_test, idx_test), **kw)

    model = mu.build_model(len(classes), pretrained=not a.no_pretrained).to(device)
    w = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, epochs=a.epochs, steps_per_epoch=len(dl_tr))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    hist = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1, best_ep, bad = -1.0, 0, 0
    model_path = os.path.join(a.out, "model.pt")
    for ep in range(1, a.epochs + 1):
        t0, model_loss, seen = time.time(), 0.0, 0
        model.train()
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            model_loss += loss.item() * len(y)
            seen += len(y)
        lg, yv = collect(model, dl_val, device)
        vloss = F.cross_entropy(lg, yv).item()
        pred = lg.argmax(1)
        vacc = float((pred == yv).float().mean())
        vf1 = float(f1_score(yv, pred, average="macro"))
        for k, v in zip(hist, (model_loss / seen, vloss, vacc, vf1)):
            hist[k].append(v)
        flag = ""
        if vf1 > best_f1:
            best_f1, best_ep, bad = vf1, ep, 0
            torch.save(model.state_dict(), model_path)
            flag = " *saved*"
        else:
            bad += 1
        print(f"epoch {ep:02d}/{a.epochs} train_loss {hist['train_loss'][-1]:.4f} val_loss {vloss:.4f} "
              f"val_acc {vacc:.4f} val_f1 {vf1:.4f} ({time.time() - t0:.0f}s){flag}")
        if bad >= a.patience:
            print("early stopping")
            break

    # ---- final evaluation with the best checkpoint
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    lg_val, y_val = collect(model, dl_val, device)
    T = fit_temperature(lg_val, y_val)
    lg_te, y_te = collect(model, dl_te, device)
    pred = lg_te.argmax(1)
    names = [mu.pretty(c) for c in classes]
    rep = classification_report(y_te, pred, target_names=names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, pred, labels=range(len(classes)))
    metrics = {"test_accuracy": float((pred == y_te).float().mean()),
               "test_macro_f1": float(f1_score(y_te, pred, average="macro")),
               "ece_before": ece(F.softmax(lg_te, 1), y_te), "ece_after": ece(F.softmax(lg_te / T, 1), y_te),
               "val_macro_f1": best_f1, "report": rep, "confusion_matrix": cm.tolist()}
    plot_confusion(cm, names, os.path.join(a.reports, "confusion_matrix.png"))
    plot_curves(hist, os.path.join(a.reports, "training_curves.png"))
    with open(os.path.join(a.reports, "metrics.json"), "w") as f:
        json.dump({**metrics, "history": hist}, f, indent=2)
    meta = {"arch": "efficientnet_b0", "classes": classes, "img_size": a.img_size, "temperature": T,
            "pretrained": not a.no_pretrained, "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "dataset": os.path.basename(os.path.normpath(a.data)), "seed": a.seed, "best_epoch": best_ep,
            "epochs_run": len(hist["val_f1"]), "n_train": len(idx_tr), "n_val": len(idx_val), "n_test": len(idx_test),
            "train_class_counts": dict(zip(classes, counts.tolist())), "dedupe": dedupe_stats, "metrics": metrics}
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nTEST accuracy {metrics['test_accuracy']:.4f} | macro-F1 {metrics['test_macro_f1']:.4f} | "
          f"T={T:.2f} | ECE {metrics['ece_before']:.3f} -> {metrics['ece_after']:.3f}")
    print(f"saved: {model_path}, {a.out}/meta.json, {a.reports}/")


if __name__ == "__main__":
    main()
