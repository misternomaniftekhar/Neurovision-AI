import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from io_utils import voxel_sizes

LABELS = {1: "Necrotic/non-enhancing core", 2: "Edema", 3: "Enhancing tumor"}


def _diameter(pts):
    """Largest pairwise distance of a point cloud (mm), via its convex hull."""
    if len(pts) < 2:
        return 0.0
    if len(pts) > 3:
        try:
            pts = pts[ConvexHull(pts).vertices]
        except Exception:
            pass
    if len(pts) > 3000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 3000, replace=False)]
    return float(pdist(pts).max())


def per_slice_areas(mask, zooms):
    """Axial per-slice areas in mm^2 (axis 2 = superior-inferior in RAS)."""
    px = zooms[0] * zooms[1]
    s = lambda m: m.sum(axis=(0, 1)) * px
    return {"whole_tumor_mm2": s(mask > 0), "core_mm2": s((mask == 1) | (mask == 3)),
            "enhancing_mm2": s(mask == 3), "edema_mm2": s(mask == 2)}


def compute_stats(mask, img):
    zooms = voxel_sizes(img)
    vox_ml = float(np.prod(zooms)) / 1000.0
    count = lambda m: int(m.sum())
    whole_mask = mask > 0
    whole = count(whole_mask)
    stats = {
        "voxel_size_mm": tuple(round(z, 3) for z in zooms),
        "whole_tumor_ml": round(whole * vox_ml, 2),
        "tumor_core_ml": round(count((mask == 1) | (mask == 3)) * vox_ml, 2),
        "parts_ml": {name: round(count(mask == k) * vox_ml, 2) for k, name in LABELS.items()},
    }
    if not whole:
        return stats

    idx = np.argwhere(whole_mask)
    stats["bbox_extent_mm"] = [round(float(e), 1) for e in (idx.max(0) - idx.min(0) + 1) * np.array(zooms)]
    stats["centroid_voxel"] = [int(c) for c in idx.mean(0)]

    surface = whole_mask & ~ndi.binary_erosion(whole_mask)
    stats["max_diameter_3d_mm"] = round(_diameter(np.argwhere(surface) * np.array(zooms)), 1)

    areas = per_slice_areas(mask, zooms)["whole_tumor_mm2"]
    best_d, best_z = 0.0, int(areas.argmax())
    for z in np.where(areas > 0)[0]:
        pts = np.argwhere(whole_mask[:, :, z])[:, :2] * np.array(zooms[:2])
        d = _diameter(pts)
        if d > best_d:
            best_d, best_z = d, int(z)
    stats["max_axial_diameter_mm"] = round(best_d, 1)
    stats["max_axial_diameter_slice"] = best_z
    stats["max_area_slice"] = int(areas.argmax())
    stats["max_area_mm2"] = round(float(areas.max()), 1)
    stats["axial_slice_range"] = [int(np.where(areas > 0)[0].min()), int(np.where(areas > 0)[0].max())]

    lab, _ = ndi.label(whole_mask)
    sizes_ml = np.bincount(lab.ravel())[1:] * vox_ml
    stats["n_components_ge_0_1ml"] = int((sizes_ml >= 0.1).sum())
    stats["largest_component_ml"] = round(float(sizes_ml.max()), 2)
    return stats
