"""Dataset folder helpers (no heavy dependencies, so the Streamlit app can use them too)."""
import os

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm")


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


def count_images(folder):
    return len([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT)])


def inspect_dataset(path):
    """Returns (info, error). info has train/test dirs, class list and image counts."""
    if not path or not os.path.isdir(path):
        return None, "Folder not found."
    try:
        tr, te = find_splits(path)
        classes = class_dirs(tr)
    except Exception as e:  # permissions etc.
        return None, str(e)
    if len(classes) < 2:
        return None, "Expected at least two class sub-folders (for example glioma, meningioma, notumor, pituitary)."
    info = {"train": tr, "test": te, "classes": classes,
            "train_counts": {c: count_images(os.path.join(tr, c)) for c in classes},
            "test_counts": {c: count_images(os.path.join(te, c)) for c in classes if os.path.isdir(os.path.join(te, c))} if te else None}
    if sum(info["train_counts"].values()) == 0:
        return None, "No images found inside the class folders."
    return info, None
