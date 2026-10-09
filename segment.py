"""Segmentation. Uses MONAI's BraTS SegResNet weights if present in models/model.pt,
otherwise falls back to a clearly-labelled DEMO heuristic (NOT a medical result).

Get weights:
  python -c "from monai.bundle import download; download('brats_mri_segmentation', bundle_dir='bundle')"
then copy bundle/brats_mri_segmentation/models/model.pt -> models/model.pt
(or host it somewhere and set the MODEL_URL environment variable).
"""
import os
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pt")
MODEL_URL = os.environ.get("MODEL_URL", "")


def _ensure_weights():
    if os.path.exists(MODEL_PATH):
        return True
    if MODEL_URL:
        import urllib.request
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        return True
    return False


def weights_available():
    return os.path.exists(MODEL_PATH) or bool(MODEL_URL)


def _normalize(vol):
    nz = vol != 0
    if nz.sum() == 0:
        return vol
    out = vol.copy()
    out[nz] = (vol[nz] - vol[nz].mean()) / (vol[nz].std() + 1e-8)
    return out


def _demo_mask(vols):
    """Intensity heuristic on FLAIR/T2. For UI demonstration only."""
    ref = vols.get("FLAIR", next(iter(vols.values())))
    nz = ref[ref > 0]
    thr = np.percentile(nz, 99) if nz.size else ref.max()
    mask = np.zeros(ref.shape, dtype=np.uint8)
    mask[ref >= thr] = 2
    return mask


def run_segmentation(vols: dict):
    """Returns (label_map uint8 [0 bg,1 core,2 edema,3 enhancing], mode)."""
    if not (len(vols) == 4 and _ensure_weights()):
        return _demo_mask(vols), "demo"

    import torch
    from monai.networks.nets import SegResNet
    from monai.inferers import sliding_window_inference

    x = np.stack([_normalize(vols[m]) for m in ["T1c", "T1", "T2", "FLAIR"]])[None]
    x = torch.from_numpy(x.astype(np.float32))
    model = SegResNet(blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=16,
                      in_channels=4, out_channels=3, dropout_prob=0.2)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state))
    model.eval()
    with torch.no_grad():
        logits = sliding_window_inference(x, (128, 128, 128), 1, model, overlap=0.25)
    tc, wt, et = (torch.sigmoid(logits)[0] > 0.5).numpy()  # channels: TC, WT, ET
    lab = np.zeros(wt.shape, dtype=np.uint8)
    lab[wt] = 2
    lab[tc] = 1
    lab[et] = 3
    return lab, "model"
