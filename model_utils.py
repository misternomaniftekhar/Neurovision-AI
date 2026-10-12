"""Model building, loading, calibrated prediction and Grad-CAM. Shared by train.py and app.py."""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

IMG_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

PRETTY = {"glioma": "Glioma", "meningioma": "Meningioma", "pituitary": "Pituitary tumor",
          "notumor": "No tumor (control)", "no_tumor": "No tumor (control)", "normal": "No tumor (control)"}


def pretty(name):
    return PRETTY.get(name.lower(), name.replace("_", " ").title())


def build_model(num_classes, pretrained=False):
    w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    m = models.efficientnet_b0(weights=w)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, num_classes))
    return m


def eval_transform(img_size=IMG_SIZE):
    # Direct resize (no centre crop) so tumours near the edge are never cropped away.
    return transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


def train_transform(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomRotation(10), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def load_model(models_dir):
    """Returns (model, meta) or (None, None) if the model has not been trained yet."""
    mp, jp = os.path.join(models_dir, "model.pt"), os.path.join(models_dir, "meta.json")
    if not (os.path.exists(mp) and os.path.exists(jp)):
        return None, None
    with open(jp) as f:
        meta = json.load(f)
    model = build_model(len(meta["classes"]), pretrained=False)
    model.load_state_dict(torch.load(mp, map_location="cpu", weights_only=True))
    return model.eval(), meta


def is_grayscale(img, tol=10.0):
    """MRI slices are grey; strongly coloured images are almost certainly not MRI."""
    a = np.asarray(img.convert("RGB"), dtype=np.int16)
    return float(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean()) < tol


def grad_cam(model, x, class_idx=None):
    """Grad-CAM on the last conv block. x: (1,3,H,W). Returns (cam HxW in [0,1], logits, class_idx)."""
    layer = model.features[-1]
    store = {}

    def fwd(_m, _i, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    h = layer.register_forward_hook(fwd)
    try:
        with torch.enable_grad():
            model.zero_grad()
            logits = model(x)
            idx = int(logits.argmax(1)) if class_idx is None else int(class_idx)
            logits[0, idx].backward()
    finally:
        h.remove()
    w = store["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * store["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam.detach().numpy(), logits.detach(), idx


def predict(model, meta, img, with_cam=True):
    """Calibrated prediction (temperature scaling) for a PIL image."""
    x = eval_transform(meta.get("img_size", IMG_SIZE))(img.convert("RGB"))[None]
    T = float(meta.get("temperature", 1.0))
    if with_cam:
        cam, logits, _ = grad_cam(model, x)
    else:
        with torch.inference_mode():
            logits = model(x)
        cam = None
    probs = F.softmax(logits / T, dim=1)[0].numpy()
    top2 = np.sort(probs)[::-1][:2]
    pred = int(probs.argmax())
    flags = []
    if probs[pred] < 0.70:
        flags.append("low confidence")
    if len(top2) > 1 and top2[0] - top2[1] < 0.20:
        flags.append("close call between top classes")
    if not is_grayscale(img):
        flags.append("image looks like a colour photo, not an MRI")
    return {"probs": probs, "pred": pred, "label": meta["classes"][pred], "confidence": float(probs[pred]),
            "cam": cam, "x": x, "flags": flags}


def overlay_cam(img, cam, alpha=0.45):
    """Blend a Grad-CAM heatmap onto the image (at the heatmap's resolution). Returns an RGB uint8 array."""
    import matplotlib
    size = cam.shape[0]
    base = np.asarray(img.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return (np.clip((1 - alpha) * base + alpha * heat, 0, 1) * 255).astype(np.uint8)
