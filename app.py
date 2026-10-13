"""NeuroVision AI - research workspace for MRI slice classification with explainability and reports."""
import glob
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

matplotlib.use("Agg")

import model_utils as mu
from dataset_utils import inspect_dataset
from report import DISCLAIMER, build_pdf, make_draft_text

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS, REPORTS, SAMPLES = (os.path.join(ROOT, d) for d in ("models", "reports", "samples"))
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
PAGES = {"Overview": "🏠", "New Scan & Samples": "📥", "MRI Workspace": "🧠", "Explainability": "🔍",
         "Model Lab": "🧪", "Train Model": "🏋️", "Scan History": "🕘", "Reports": "📄", "Docs & Ethics": "📚"}
DOTS = ["🔴", "🟠", "🟣", "🟢", "🔵", "🟡"]

st.set_page_config(page_title="NeuroVision AI", page_icon="🧠", layout="wide")
ss = st.session_state
ss.setdefault("page", "Overview")
ss.setdefault("scan", None)
ss.setdefault("scans", {})
ss.setdefault("history", [])


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return mu.load_model(MODELS)


model, meta = get_model()


def list_samples():
    out = []
    for cdir in sorted(glob.glob(os.path.join(SAMPLES, "*"))):
        if os.path.isdir(cdir):
            files = sorted(f for f in glob.glob(os.path.join(cdir, "*")) if f.lower().endswith(IMG_EXT))
            for i, f in enumerate(files, 1):
                m = re.search(r"(\d+)\.\w+$", f)
                out.append({"label": os.path.basename(cdir), "path": f, "num": int(m.group(1)) if m else i})
    return out


def sample_name(s):
    return f"{mu.pretty(s['label'])} #{s['num']:03d}"


def set_scan(img, name, true=None):
    sid = f"{name}-{time.time_ns()}"
    ss.scan = {"id": sid, "name": name, "img": img.convert("RGB"), "true": true, "res": None, "cams": {}}
    ss.scans[sid] = ss.scan
    ss.page = "MRI Workspace"


def pick_sample(s):
    set_scan(Image.open(s["path"]), sample_name(s), s["label"])


def on_upload():
    f = ss.get("upl")
    if f is not None:
        try:
            set_scan(Image.open(f), f.name)
        except Exception:
            ss.scan = None
            ss["upload_error"] = f"Could not read {f.name} as an image."


def ensure_result():
    sc = ss.scan
    if sc and model is not None and sc["res"] is None:
        with st.spinner("Analysing scan..."):
            sc["res"] = mu.predict(model, meta, sc["img"])
        r = sc["res"]
        ss.history.append({"Time": time.strftime("%H:%M:%S"), "Scan": sc["name"], "Predicted": mu.pretty(r["label"]),
                           "Confidence": round(r["confidence"], 3), "Known label": mu.pretty(sc["true"]) if sc["true"] else "",
                           "Flags": "; ".join(r["flags"])})
    return sc


def big(img, size=448):
    return img.resize((size, size)) if isinstance(img, Image.Image) else Image.fromarray(img).resize((size, size))


def need_scan():
    if model is None:
        st.warning("No trained model found. See **Overview** for the training command.")
        return None
    if ss.scan is None:
        st.info("Pick a scan first: upload one or choose a benchmark scan in the sidebar.")
        return None
    return ensure_result()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🧠 NeuroVision AI")
    st.caption("AI IMAGING LAB · RESEARCH WORKSPACE")
    st.radio("Navigation", list(PAGES), key="page", format_func=lambda p: f"{PAGES[p]}  {p}",
             label_visibility="collapsed")
    st.markdown("---")
    st.caption("BENCHMARK SCANS")
    samples = list_samples()
    classes = meta["classes"] if meta else sorted({s["label"] for s in samples})
    for i, s in enumerate(samples[:12]):
        dot = DOTS[classes.index(s["label"]) % len(DOTS)] if s["label"] in classes else "⚪"
        st.button(f"{dot} {sample_name(s)}", key=f"s{i}", on_click=pick_sample, args=(s,))
    if not samples:
        st.caption("No samples yet. Run `make_samples.py` (see README).")


# ---------------------------------------------------------------- pages
def page_overview():
    st.title("NeuroVision AI")
    st.caption("Upload MRI slice → AI classification → explainability → draft report")
    st.warning(DISCLAIMER)
    if model is None:
        st.error("The model has not been trained yet.")
        st.markdown("**Easiest:** open **Train Model** in the sidebar (run the app on your own computer; a GPU helps). "
                    "Or train from the command line / a Kaggle or Colab notebook, then copy `models/model.pt` "
                    "and `models/meta.json` into this project:")
        st.code("pip install -r requirements-train.txt\npython train.py --data /path/to/dataset --epochs 15", "bash")
        return
    m = meta["metrics"]
    c = st.columns(4)
    c[0].metric("Test accuracy", f"{m['test_accuracy']:.1%}")
    c[1].metric("Macro F1", f"{m['test_macro_f1']:.3f}")
    c[2].metric("Classes", len(meta["classes"]))
    c[3].metric("Calibration (ECE)", f"{m['ece_after']:.3f}")
    st.markdown(f"**Model:** {meta['arch']} fine-tuned on `{meta['dataset']}` "
                f"({meta['n_train']} train / {meta['n_val']} val / {meta['n_test']} test images), "
                f"trained {meta['trained_at']}.")
    st.markdown("**Pipeline:** 1. Upload or pick a benchmark scan → 2. Resize/normalise → 3. EfficientNet-B0 classifier "
                "with temperature-calibrated probabilities → 4. Grad-CAM attention map → 5. Draft report + PDF.")


def page_new():
    st.title("New Scan & Samples")
    st.file_uploader("Upload an MRI slice (JPG/PNG)", type=["jpg", "jpeg", "png", "bmp"], key="upl", on_change=on_upload)
    if ss.pop("upload_error", None):
        st.error("That file could not be read as an image.")
    st.subheader("Benchmark scans")
    if not samples:
        st.info("Run `python make_samples.py --data /path/to/dataset --n 3` to fill the `samples/` folder.")
    cols = st.columns(4)
    for i, s in enumerate(samples[:12]):
        with cols[i % 4]:
            st.image(Image.open(s["path"]).convert("RGB").resize((224, 224)), caption=sample_name(s))
            st.button("Analyse", key=f"n{i}", on_click=pick_sample, args=(s,))


def page_workspace():
    st.title("MRI Workspace")
    sc = need_scan()
    if not sc:
        return
    st.caption(DISCLAIMER)
    r = sc["res"]
    a, b = st.columns([1, 1])
    with a:
        st.image(big(sc["img"]), caption=sc["name"])
        with st.expander("Model input (resized)"):
            st.image(sc["img"].resize((mu.IMG_SIZE, mu.IMG_SIZE)))
    with b:
        st.subheader(mu.pretty(r["label"]))
        st.metric("Calibrated confidence", f"{r['confidence']:.1%}")
        for f in r["flags"]:
            st.warning(f.capitalize())
        if sc["true"]:
            ok = sc["true"] == r["label"]
            (st.success if ok else st.error)(f"Known label: {mu.pretty(sc['true'])} → "
                                             f"{'match' if ok else 'mismatch'}")
        st.bar_chart(pd.DataFrame({"probability": r["probs"]}, index=[mu.pretty(c) for c in meta["classes"]]))


def page_explain():
    st.title("Explainability")
    sc = need_scan()
    if not sc:
        return
    st.caption(DISCLAIMER)
    r = sc["res"]
    names = [mu.pretty(c) for c in meta["classes"]]
    c1, c2 = st.columns(2)
    target = c1.selectbox("Explain class", names, index=r["pred"])
    alpha = c2.slider("Overlay strength", 0.1, 0.9, 0.45, 0.05)
    idx = names.index(target)
    if idx not in sc["cams"]:
        sc["cams"][idx] = r["cam"] if idx == r["pred"] else mu.grad_cam(model, r["x"], idx)[0]
    cam = sc["cams"][idx]
    x1, x2, x3 = st.columns(3)
    x1.image(big(sc["img"]), caption="Input")
    x2.image(big(mu.overlay_cam(sc["img"], cam, alpha)), caption=f"Grad-CAM: {target}")
    x3.image(big((np.asarray(__import__('matplotlib').colormaps['jet'](cam))[..., :3] * 255).astype(np.uint8)),
             caption="Heatmap only")
    st.info("Grad-CAM shows which regions most influenced the model's score for a class. It is not a tumour "
            "segmentation and does not prove the prediction is correct; models can attend to artefacts.")


def page_lab():
    st.title("Model Lab")
    if meta is None:
        st.info("Train the model first (see Overview).")
        return
    m = meta["metrics"]
    c = st.columns(4)
    c[0].metric("Test accuracy", f"{m['test_accuracy']:.1%}")
    c[1].metric("Macro F1", f"{m['test_macro_f1']:.3f}")
    c[2].metric("Temperature", f"{meta['temperature']:.2f}")
    c[3].metric("ECE before → after", f"{m['ece_before']:.3f} → {m['ece_after']:.3f}")
    rep = {k: v for k, v in m["report"].items() if k != "accuracy"}
    dd = meta.get("dedupe", {})
    if dd.get("enabled") and "train_before" in dd:
        st.info(f"Duplicate filter: training images {dd['train_before']} → {dd['train_after']}; test images "
                f"{dd.get('test_before', '-')} → {dd.get('test_after', '-')} "
                f"({dd.get('test_leaky_with_train', 0)} test images were near-copies of training images and were removed).")
    small = [k for k, v in rep.items() if k not in ("macro avg", "weighted avg") and v.get("support", 1e9) < 100]
    if small:
        st.warning("Few test images for: " + ", ".join(small) + ". Their scores are noisy; treat them with caution.")
    st.subheader("Per-class test metrics")
    st.dataframe(pd.DataFrame(rep).T.rename(columns={"f1-score": "f1"}).round(3))
    left, right = st.columns(2)
    for col, fn in ((left, "confusion_matrix.png"), (right, "training_curves.png")):
        p = os.path.join(REPORTS, fn)
        if os.path.exists(p):
            col.image(p)
    st.caption(f"Training class counts: {meta['train_class_counts']} · best epoch {meta['best_epoch']} of "
               f"{meta['epochs_run']} · pretrained={meta['pretrained']}")
    st.subheader("Benchmark scans check")
    if samples and st.button("Run model on all benchmark scans"):
        rows = []
        for s in samples:
            res = mu.predict(model, meta, Image.open(s["path"]), with_cam=False)
            rows.append({"Scan": sample_name(s), "True": mu.pretty(s["label"]), "Predicted": mu.pretty(res["label"]),
                         "Confidence": round(res["confidence"], 3), "Correct": res["label"] == s["label"]})
        df = pd.DataFrame(rows)
        st.dataframe(df)
        st.caption(f"{df['Correct'].mean():.0%} correct on {len(df)} samples. Too few images to estimate "
                   "performance; use the test metrics above.")


TRAIN_LOG = os.path.join(tempfile.gettempdir(), "neurovision_train.log")
EPOCH_RE = re.compile(r"epoch (\d+)/(\d+) train_loss ([\d.]+) val_loss ([\d.]+) val_acc ([\d.]+) val_f1 ([\d.]+)")


def read_log(n=60):
    try:
        with open(TRAIN_LOG, errors="replace") as f:
            return f.read().splitlines()[-n:]
    except OSError:
        return []


def parse_epochs(lines):
    rows = [tuple(float(x) for x in m.groups()) for m in map(EPOCH_RE.search, lines) if m]
    return pd.DataFrame(rows, columns=["epoch", "of", "train_loss", "val_loss", "val_acc", "val_f1"]) if rows else None


def show_training_state(tr, live):
    lines = read_log(400)
    df = parse_epochs(lines)
    total = int(df["of"].iloc[-1]) if df is not None else tr["epochs"]
    done_ep = int(df["epoch"].iloc[-1]) if df is not None else 0
    if live:
        st.progress(min(done_ep / max(total, 1), 1.0), text=f"Epoch {done_ep} of {total}"
                    + (" - checking for duplicate images / loading data..." if df is None else ""))
    if df is not None:
        st.line_chart(df.set_index("epoch")[["train_loss", "val_loss", "val_f1", "val_acc"]])
        st.caption(f"Latest: val accuracy {df['val_acc'].iloc[-1]:.3f}, val macro-F1 {df['val_f1'].iloc[-1]:.3f}")
    with st.expander("Training log", expanded=live):
        st.code("\n".join(lines[-25:]) or "(waiting for output...)")


@st.fragment(run_every=3)
def train_monitor():
    tr = ss.get("train")
    if not tr:
        return
    code = tr["proc"].poll()
    if code is not None:
        st.rerun()  # finished: redraw the page in its final state
    show_training_state(tr, live=True)
    if st.button("Stop training"):
        tr["proc"].terminate()


def start_training(cmd, epochs, data_path):
    log = open(TRAIN_LOG, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    ss["train"] = {"proc": proc, "epochs": epochs, "t0": time.time(), "done": False, "data": data_path}


def extract_zip(upload, dest):
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    root = os.path.realpath(dest)
    with zipfile.ZipFile(upload) as z:
        for m in z.infolist():
            if not os.path.realpath(os.path.join(root, m.filename)).startswith(root + os.sep) and m.filename.strip("/"):
                raise ValueError("Unsafe path inside the zip file.")
        z.extractall(root)
    return root


def page_train():
    st.title("Train Model")
    on_cloud = ROOT.startswith("/mount/src")
    if on_cloud:
        st.error("Training is not available on Streamlit Community Cloud: it has no GPU, little memory and loses "
                 "files on restart. Run this app on your own computer (`streamlit run app.py`) to train, then "
                 "commit `models/model.pt` and `models/meta.json` to GitHub for the hosted app.")
        return
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    st.caption(f"Training runs on this computer · device: {gpu or 'CPU only (slow)'}")
    tr = ss.get("train")
    code = tr["proc"].poll() if tr else None
    if tr and code is None:
        st.info("Training in progress. You can open other pages; it keeps running.")
        train_monitor()
        return
    if tr and code is not None:
        if code == 0 and not tr["done"]:
            tr["done"] = True
            get_model.clear()
            st.rerun()
        (st.success if code == 0 else st.error)(
            "Training finished. The new model is loaded: try it in MRI Workspace." if code == 0
            else f"Training stopped or failed (exit code {code}). See the log below.")
        show_training_state(tr, live=False)
        st.divider()

    if importlib.util.find_spec("sklearn") is None:
        st.error("Training needs scikit-learn. Run `pip install scikit-learn`, then restart the app.")
        return
    src = st.radio("Dataset source", ["Folder on this computer", "Upload a .zip"], horizontal=True)
    if src == "Folder on this computer":
        ss["data_path"] = st.text_input("Dataset folder", value=ss.get("data_path", ""),
                                        placeholder="Folder that contains Training/ and Testing/ (or class folders)")
    else:
        up = st.file_uploader("Dataset zip", type=["zip"], key="data_zip")
        if up is not None and st.button("Extract dataset"):
            with st.spinner("Extracting..."):
                try:
                    ss["data_path"] = extract_zip(up, os.path.join(ROOT, "data", "uploaded"))
                except Exception as e:
                    st.error(str(e))
    info, err = inspect_dataset(ss.get("data_path", ""))
    if err:
        if ss.get("data_path"):
            st.warning(err)
        st.info("Expected layout: `Training/<class>/*.jpg` and `Testing/<class>/*.jpg` "
                "(or a single folder of class sub-folders).")
        return
    rows = {c: {"train": info["train_counts"][c], "test": (info["test_counts"] or {}).get(c, "-")} for c in info["classes"]}
    st.success(f"Dataset found: {len(info['classes'])} classes" + ("" if info["test"] else " (no test folder: 15% will be held out)"))
    st.dataframe(pd.DataFrame(rows).T)

    c1, c2, c3 = st.columns(3)
    quick = c1.checkbox("Quick test (40 images per class, 2 epochs)", value=not gpu,
                        help="Checks that everything works before a long run.")
    epochs = c2.slider("Epochs", 1, 30, 2 if quick else 15, disabled=quick)
    bs = c3.selectbox("Batch size", [16, 32, 64], index=1)
    img = st.selectbox("Image size", [128, 160, 224], index=2, help="Smaller is faster; 224 is the standard.")
    with st.expander("Advanced"):
        lr = st.number_input("Learning rate", 1e-5, 1e-2, 5e-4, format="%.5f")
        pat = st.number_input("Early-stopping patience", 1, 20, 5)
        pretrained = st.checkbox("Use ImageNet-pretrained weights (needs internet, strongly recommended)", True)
        dedupe = st.checkbox("Remove near-duplicate images (prevents inflated test scores)", True)
    if not gpu and not quick:
        st.warning("No GPU detected: a full run can take hours. Consider a quick test first, or a free Kaggle/Colab GPU.")
    if meta is not None:
        st.warning("A trained model already exists. Starting a new run will replace it.")
    if st.button("Start training", type="primary"):
        cmd = [sys.executable, "-u", os.path.join(ROOT, "train.py"), "--data", ss["data_path"], "--out", MODELS,
               "--reports", REPORTS, "--epochs", str(2 if quick else epochs), "--batch-size", str(bs),
               "--img-size", str(img), "--lr", str(lr), "--patience", str(int(pat)),
               "--workers", str(min(4, os.cpu_count() or 1))]
        if not pretrained:
            cmd.append("--no-pretrained")
        if not dedupe:
            cmd.append("--no-dedupe")
        if quick:
            cmd += ["--max-per-class", "40"]
        start_training(cmd, 2 if quick else epochs, ss["data_path"])
        st.rerun()
    st.divider()
    if st.button("Create benchmark samples from this dataset"):
        with st.spinner("Picking sample scans (skips duplicates of training images)..."):
            r = subprocess.run([sys.executable, os.path.join(ROOT, "make_samples.py"), "--data", ss["data_path"],
                                "--n", "3", "--out", SAMPLES], capture_output=True, text=True, cwd=ROOT)
        (st.success if r.returncode == 0 else st.error)(
            "Samples created. They appear in the sidebar after you reload the page." if r.returncode == 0 else r.stderr[-400:])


def page_history():
    st.title("Scan History")
    if not ss.history:
        st.info("Scans you analyse in this session appear here. History is not stored on the server.")
        return
    df = pd.DataFrame(ss.history)
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False).encode(), "neurovision_history.csv", "text/csv")


def page_reports():
    st.title("Reports")
    if model is None or not any(s["res"] for s in ss.scans.values()):
        st.info("Analyse a scan first, then generate its report here.")
        return
    done = {s["id"]: s for s in ss.scans.values() if s["res"]}
    cur = ss.scan["id"] if ss.scan and ss.scan["id"] in done else list(done)[-1]
    sid = st.selectbox("Scan", list(done), index=list(done).index(cur), format_func=lambda i: done[i]["name"])
    sc = done[sid]
    if st.button("Generate draft report"):
        try:
            key = st.secrets.get("GROQ_API_KEY", None)
        except Exception:
            key = None
        r = sc["res"]
        text = make_draft_text(sc["name"], r, meta, key)
        pdf = build_pdf(text, sc["name"], r, meta, sc["img"], mu.overlay_cam(sc["img"], r["cam"]))
        ss["report"] = {"sid": sid, "text": text, "pdf": pdf}
    rp = ss.get("report")
    if rp and rp["sid"] == sid:
        st.text_area("Draft (preview)", rp["text"], height=300)
        st.download_button("Download PDF", rp["pdf"], "neurovision_report.pdf", "application/pdf")


def page_docs():
    st.title("Docs & Ethics")
    st.markdown("""
**Intended use.** Education and research demonstration of an image-classification pipeline. Not a medical device.

**What it does.** Classifies a single 2D MRI slice into the classes the model was trained on, with temperature-calibrated
probabilities, a Grad-CAM attention map and a draft text report.

**What it cannot do.** It cannot localise or measure a tumour, grade or stage it, use 3D context, or recognise
conditions outside its training classes. Any image, even a non-MRI one, is forced into one of the known classes,
so flags for low confidence or colour images are shown, but they are not a guarantee.

**Data and bias.** The model learns from one public dataset (see Model Lab). Scanner, protocol, population and label
quality in that dataset limit how well results transfer. Public Kaggle MRI sets can contain duplicates or
mislabelled slices, so reported test accuracy may be optimistic.

**Privacy.** Uploaded images are processed in memory for your session and are not saved by this app. Do not upload
identifiable patient data to a public deployment.

**Human review.** Every output must be reviewed by a qualified specialist before any use.
""")


{"Overview": page_overview, "New Scan & Samples": page_new, "MRI Workspace": page_workspace,
 "Explainability": page_explain, "Model Lab": page_lab, "Train Model": page_train, "Scan History": page_history,
 "Reports": page_reports, "Docs & Ethics": page_docs}[ss.page]()
