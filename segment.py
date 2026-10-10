"""Segmentation with MONAI's pretrained BraTS SegResNet bundle (brats_mri_segmentation).

Weights are looked up in this order:
  1. models/model.pt in the repo (you can commit it, ~20 MB)
  2. a cached bundle download in the temp dir
  3. MODEL_URL env var (direct link to model.pt)
  4. monai.bundle.download("brats_mri_segmentation") at first run
If none works, a clearly-labelled DEMO heuristic is used (NOT a medical result).
"""
import functools
import os
import tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_PATH = os.path.join(HERE, "models", "model.pt")
CACHE_DIR = os.path.join(tempfile.gettempdir(), "neurovision_bundle")
CACHED_PATH = os.path.join(CACHE_DIR, "brats_mri_segmentation", "models", "model.pt")
MODEL_URL = os.environ.get("MODEL_URL", "")
INPUT_ORDER = ["T1c", "T1", "T2", "FLAIR"]


def find_weights():
    for p in (LOCAL_PATH, CACHED_PATH):
        if os.path.exists(p):
            return p
    return None


def fetch_weights():
    p = find_weights()
    if p:
        return p
    if MODEL_URL:
        import urllib.request
        os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, LOCAL_PATH)
        return LOCAL_PATH
    from monai.bundle import download
    download(name="brats_mri_segmentation", bundle_dir=CACHE_DIR)
    return find_weights()


def _normalize(vol):
    nz = vol != 0
    if nz.sum() == 0:
        return vol
    out = vol.copy()
    out[nz] = (vol[nz] - vol[nz].mean()) / (vol[nz].std() + 1e-8)
    return out


def _demo_mask(vols):
    """Intensity heuristic on FLAIR. For UI demonstration only."""
    ref = vols.get("FLAIR", next(iter(vols.values())))
    nz = ref[ref > 0]
    thr = np.percentile(nz, 99) if nz.size else ref.max()
    mask = np.zeros(ref.shape, dtype=np.uint8)
    mask[ref >= thr] = 2
    return mask


@functools.lru_cache(maxsize=1)
def _load_model(path):
    import torch
    from monai.networks.nets import SegResNet
    model = SegResNet(blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=16,
                      in_channels=4, out_channels=3, dropout_prob=0.2)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model.eval()


def _brain_crop(x, pad=8):
    """Bounding box (with padding) around non-zero voxels, to cut inference time/memory."""
    idx = np.argwhere((x != 0).any(0))
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + 1 + pad, x.shape[1:])
    return tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))


def run_segmentation(vols: dict, accurate=False):
    """Returns (label_map uint8 [0 bg, 1 core, 2 edema, 3 enhancing], mode, note)."""
    if not all(m in vols for m in INPUT_ORDER):
        return _demo_mask(vols), "demo", "Fewer than 4 modalities uploaded: showing a placeholder DEMO mask."
    try:
        path = fetch_weights()
        err = "" if path else "download finished but model.pt was not found"
    except Exception as e:  # no network, etc.
        path, err = None, f"{type(e).__name__}: {e}"
    if not path:
        return _demo_mask(vols), "demo", f"Model weights unavailable ({err}). Showing a DEMO mask."

    import torch
    from monai.inferers import sliding_window_inference

    x = np.stack([_normalize(vols[m]) for m in INPUT_ORDER]).astype(np.float32)
    sl = _brain_crop(x)
    xt = torch.from_numpy(np.ascontiguousarray(x[(slice(None),) + sl]))[None]
    model = _load_model(path)
    with torch.inference_mode():
        logits = sliding_window_inference(xt, (128, 128, 128), 1, model,
                                          overlap=0.5 if accurate else 0.25,
                                          mode="gaussian" if accurate else "constant")
    tc, wt, et = (torch.sigmoid(logits)[0] > 0.5).numpy()  # bundle output: TC, WT, ET
    lab_c = np.zeros(wt.shape, dtype=np.uint8)
    lab_c[wt] = 2
    lab_c[tc] = 1
    lab_c[et] = 3
    lab = np.zeros(x.shape[1:], dtype=np.uint8)
    lab[sl] = lab_c
    return lab, "model", "BraTS SegResNet (MONAI bundle)."
