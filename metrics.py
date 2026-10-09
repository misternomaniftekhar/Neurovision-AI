import numpy as np

LABELS = {1: "Necrotic/non-enhancing core", 2: "Edema", 3: "Enhancing tumor"}


def compute_stats(mask, img):
    zooms = img.header.get_zooms()[:3]
    vox_ml = float(np.prod(zooms)) / 1000.0
    count = lambda m: int(m.sum())
    whole = count(mask > 0)
    core = count((mask == 1) | (mask == 3))
    stats = {
        "voxel_size_mm": tuple(round(float(z), 3) for z in zooms),
        "whole_tumor_ml": round(whole * vox_ml, 2),
        "tumor_core_ml": round(core * vox_ml, 2),
        "parts_ml": {name: round(count(mask == k) * vox_ml, 2) for k, name in LABELS.items()},
    }
    if whole:
        idx = np.argwhere(mask > 0)
        ext = (idx.max(0) - idx.min(0) + 1) * np.array(zooms)
        stats["bbox_extent_mm"] = [round(float(e), 1) for e in ext]
        stats["centroid_voxel"] = [int(c) for c in idx.mean(0)]
    return stats
