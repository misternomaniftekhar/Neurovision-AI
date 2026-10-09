import io
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from neurovision.io_utils import load_modalities, MODALITIES
from neurovision.segment import run_segmentation, weights_available
from neurovision.metrics import compute_stats
from neurovision.report import generate_report_text, build_pdf, DISCLAIMER

st.set_page_config(page_title="NeuroVision AI", page_icon="🧠", layout="wide")
st.title("🧠 NeuroVision AI")
st.warning(DISCLAIMER)

CMAP = ListedColormap([(0, 0, 0, 0), (1, 0, 0, .55), (1, .85, 0, .55), (0, .6, 1, .6)])


def render(vol, mask, k):
    """Axial / coronal / sagittal views through the axial slice's tumor centre."""
    idx = np.argwhere(mask > 0)
    cy, cx = (idx[:, 1].mean(), idx[:, 0].mean()) if len(idx) else (vol.shape[1] // 2, vol.shape[0] // 2)
    cy, cx = int(cy), int(cx)
    views = [(vol[:, :, k], mask[:, :, k]), (vol[:, cy, :], mask[:, cy, :]), (vol[cx, :, :], mask[cx, :, :])]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, (v, m) in zip(ax, views):
        a.imshow(np.rot90(v), cmap="gray")
        a.imshow(np.rot90(m), cmap=CMAP, vmin=0, vmax=3)
        a.axis("off")
    fig.tight_layout()
    return fig


with st.sidebar:
    st.header("1. Upload MRI (NIfTI)")
    st.caption("Best results: all four BraTS modalities, 1 mm isotropic, skull-stripped.")
    files = {m: st.file_uploader(m, type=["nii", "gz"], key=m) for m in MODALITIES}
    run = st.button("Run segmentation", type="primary")
    st.caption("Model weights: " + ("found ✅" if weights_available() else "not found, demo mode"))

if run:
    try:
        with st.spinner("Loading and validating..."):
            ref, vols = load_modalities(files)
        with st.spinner("Segmenting (CPU inference can take a few minutes)..."):
            mask, mode = run_segmentation(vols)
        st.session_state.update(ref=ref, vols=vols, mask=mask, mode=mode,
                                stats=compute_stats(mask, ref))
        st.session_state.pop("pdf", None)
    except Exception as e:
        st.error(str(e))

if "mask" in st.session_state:
    ss = st.session_state
    base_name = "FLAIR" if "FLAIR" in ss.vols else next(iter(ss.vols))
    vol = ss.vols[base_name]
    if ss.mode == "demo":
        st.info("DEMO mode: mask is a placeholder heuristic (needs 4 modalities + weights for the real model).")

    tab1, tab2, tab3 = st.tabs(["Viewer", "Measurements", "Report"])
    with tab1:
        has = np.where(ss.mask.any(axis=(0, 1)))[0]
        default = int(has[len(has) // 2]) if len(has) else vol.shape[2] // 2
        k = st.slider("Axial slice", 0, vol.shape[2] - 1, default)
        st.pyplot(render(vol, ss.mask, k))
        st.caption("Red: core | Yellow: edema | Blue: enhancing")
    with tab2:
        st.json(ss.stats)
    with tab3:
        if st.button("Generate report + PDF"):
            key = st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None
            text = generate_report_text(ss.stats, ss.mode, key)
            buf = io.BytesIO()
            render(vol, ss.mask, default).savefig(buf, format="png", dpi=110)
            ss.report = text
            ss.pdf = build_pdf(text, ss.stats, ss.mode, buf.getvalue())
        if "pdf" in ss:
            st.text_area("Draft report (editable preview)", ss.report, height=300)
            st.download_button("Download PDF", ss.pdf, "neurovision_report.pdf", "application/pdf")
else:
    st.info("Upload scans in the sidebar and click **Run segmentation**.")
