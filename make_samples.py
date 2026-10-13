#!/usr/bin/env python
"""Copy a few images per class into samples/<class>/ for the app's Benchmark Scans list.
    python make_samples.py --data /path/to/dataset --n 3
Samples come from the Testing split and skip any image that is a near-copy of a training image, so the
benchmark scans are ones the model has genuinely never seen."""
import argparse
import os
import random
import shutil

from torchvision import datasets

from dataset_utils import find_splits
from train import near_dup_filter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="samples")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    train_dir, test_dir = find_splits(a.data)
    rng = random.Random(a.seed)
    if test_dir:
        dtr, dte = datasets.ImageFolder(train_dir), datasets.ImageFolder(test_dir)
        _, kept, st = near_dup_filter([p for p, _ in dtr.samples], dtr.targets, [p for p, _ in dte.samples])
        print(f"skipped {st['test_before'] - st['test_after']} test images that duplicate training images")
        by_class = {c: [dte.samples[i][0] for i in kept if dte.targets[i] == k] for k, c in enumerate(dte.classes)}
    else:
        dtr = datasets.ImageFolder(train_dir)
        by_class = {c: [p for p, t in dtr.samples if t == k] for k, c in enumerate(dtr.classes)}
    for c, files in by_class.items():
        os.makedirs(os.path.join(a.out, c), exist_ok=True)
        pick = rng.sample(files, min(a.n, len(files)))
        for i, f in enumerate(pick, 1):
            shutil.copy(f, os.path.join(a.out, c, f"{c}_{i:03d}{os.path.splitext(f)[1].lower()}"))
        print(f"{c}: copied {len(pick)} (from {len(files)} eligible)")


if __name__ == "__main__":
    main()
