import gzip
import io

import matplotlib
import nibabel as nib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from io_utils import MODALITIES, load_modalities
from metrics import LABELS, compute_stats, per_slice_areas
from report import DISCLAIMER, build_pdf, generate_report_text
from segment import find_weights, run_segmentation
from viz3d import SURFACES, mesh_figure

st.set_page_config(page_title="NeuroVision AI", page_icon="🧠", layout="wide")
st.title("🧠 NeuroVision AI")
st.warning(DISCLAIMER)

COLORS = {1: (1.0, 0.23, 0.19), 2: (1.0, 0.83, 0.0), 3: (0.04, 0.52, 1.0)}


def overlay(m, show, alpha):
    rgba = np.zeros(m.shape + (4,), dtype=float)
    for k in show:
        rgba[m == k] = (*COLORS[k], alpha)
    return rgba


def render_triplane(vol, mask, pos, zooms, show, alpha, window):
    x, y, z = pos
    views = [("Axial", vol[:, :, z], mask[:, :, z], zooms[1] / zooms[0]),
             ("Coronal", vol[:, y, :], mask[:, y, :], zooms[2] / zooms[0]),
             ("Sagittal", vol[x, :, :], mask[x, :, :], zooms[2] / zooms[1])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    fig.patch.set_facecolor("#0e1117")
    for ax, (title, v, m, asp) in zip(axes, views):
        ax.imshow(np.rot90(v), cmap="gray", vmin=window[0], vmax=window[1], aspect=asp)
        ax.imshow(overlay(np.rot90(m), show, alpha), aspect=asp)
        ax.set_title(title, color="white", fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    return fig


def window_for(vol):
    nz = vol[vol > 0]
    return (float(np.percentile(nz, 1)), float(np.percentile(nz, 99))) if nz.size else (0.0, 1.0)


def mask_nifti_bytes(mask, ref):
    return gzip.compress(nib.Nifti1Image(mask.astype(np.uint8), ref.affine).to_bytes())


with st.sidebar:
    st.header("1. Upload MRI (NIfTI)")
    st.caption("Best results: all four BraTS modalities, co-registered and skull-stripped.")
    files = {m: st.file_uploader(m, type=["nii", "gz"], key=m) for m in MODALITIES}
    resample = st.checkbox("Resample to 1 mm isotropic", value=True,
                           help="The BraTS model expects 1 mm voxels. Leave on unless your data is already 1 mm.")
    accurate = st.radio("Inference quality", ["Fast", "Accurate (slower)"]) == "Accurate (slower)"
    run = st.button("Run segmentation", type="primary")
    st.caption("Model weights: " + ("found locally ✅" if find_weights()
                                    else "will be downloaded on first full run"))

if run:
    try:
        with st.spinner("Loading, validating and resampling..."):
            ref, vols, info = load_modalities(files, resample)
        with st.spinner("Segmenting (first run downloads weights; CPU inference can take a few minutes)..."):
            mask, mode, note = run_segmentation(vols, accurate)
        with st.spinner("Computing measurements..."):
            stats = compute_stats(mask, ref)
        st.session_state.update(ref=ref, vols=vols, mask=mask, mode=mode, note=note, stats=stats,
                                info=info, windows={m: window_for(v) for m, v in vols.items()})
        st.session_state.pop("pdf", None)
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")

if "mask" not in st.session_state:
    st.info("Upload scans in the sidebar and click **Run segmentation**.")
    st.stop()

ss = st.session_state
stats, mask, zooms = ss.stats, ss.mask, ss.stats["voxel_size_mm"]
if ss.mode == "demo":
    st.error("DEMO mode: " + ss.note + " This mask is NOT a model prediction.")
else:
    st.success(ss.note)
if ss.info["resampled"]:
    st.caption("Volumes were resampled to 1 mm isotropic.")
if stats["whole_tumor_ml"] == 0:
    st.warning("No tumor voxels were predicted in this scan.")

tab_view, tab_3d, tab_meas, tab_rep = st.tabs(["2D viewer", "3D view", "Measurements", "Report"])

with tab_view:
    c1, c2, c3 = st.columns([2, 2, 3])
    mod = c1.selectbox("Base image", list(ss.vols))
    alpha = c2.slider("Overlay opacity", 0.0, 1.0, 0.55, 0.05)
    show = c3.multiselect("Labels", list(LABELS), default=list(LABELS), format_func=LABELS.get)
    vol = ss.vols[mod]
    cx, cy, cz = stats.get("centroid_voxel", [s // 2 for s in vol.shape])
    cz = stats.get("max_area_slice", cz)
    s1, s2, s3 = st.columns(3)
    z = s1.slider("Axial slice", 0, vol.shape[2] - 1, int(cz))
    y = s2.slider("Coronal slice", 0, vol.shape[1] - 1, int(cy))
    x = s3.slider("Sagittal slice", 0, vol.shape[0] - 1, int(cx))
    fig = render_triplane(vol, mask, (x, y, z), zooms, show, alpha, ss.windows[mod])
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Yellow: edema | Red: necrotic/non-enhancing core | Blue: enhancing tumor")

with tab_3d:
    which = st.multiselect("Surfaces", list(SURFACES), default=["Whole tumor", "Tumor core", "Enhancing"])
    if mask.any():
        st.plotly_chart(mesh_figure(mask, zooms, which))
    else:
        st.info("Nothing to render.")

with tab_meas:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Whole tumor", f"{stats['whole_tumor_ml']} mL")
    m2.metric("Tumor core", f"{stats['tumor_core_ml']} mL")
    m3.metric("Enhancing", f"{stats['parts_ml']['Enhancing tumor']} mL")
    m4.metric("Max diameter (3D)", f"{stats.get('max_diameter_3d_mm', 0)} mm")
    m5.metric("Max axial diameter", f"{stats.get('max_axial_diameter_mm', 0)} mm")
    st.caption(f"Voxel size {zooms} mm. Axial diameter measured on slice "
               f"{stats.get('max_axial_diameter_slice', '-')}. Components ≥0.1 mL: "
               f"{stats.get('n_components_ge_0_1ml', 0)}.")
    areas = per_slice_areas(mask, zooms)
    df = pd.DataFrame(areas)
    df.index.name = "axial slice"
    st.subheader("Per-slice area (mm²)")
    st.line_chart(df[["whole_tumor_mm2", "core_mm2", "enhancing_mm2", "edema_mm2"]])
    with st.expander("Per-slice table and all statistics"):
        st.dataframe(df[df["whole_tumor_mm2"] > 0].round(1))
        st.json(stats)
    d1, d2 = st.columns(2)
    d1.download_button("Download per-slice CSV", df.round(2).to_csv().encode(),
                       "neurovision_per_slice.csv", "text/csv")
    d2.download_button("Download mask (NIfTI)", mask_nifti_bytes(mask, ss.ref),
                       "neurovision_mask.nii.gz", "application/gzip")

with tab_rep:
    if st.button("Generate report + PDF"):
        try:
            key = st.secrets.get("GROQ_API_KEY", None)
        except Exception:
            key = None
        text = generate_report_text(stats, ss.mode, key)
        base = list(ss.vols)[-1]
        fig = render_triplane(ss.vols[base], mask, (int(stats.get("centroid_voxel", [0, 0, 0])[0]),
                                                    int(stats.get("centroid_voxel", [0, 0, 0])[1]),
                                                    int(stats.get("max_area_slice", 0))),
                              zooms, list(LABELS), 0.55, ss.windows[base])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor())
        plt.close(fig)
        ss.report = text
        ss.pdf = build_pdf(text, stats, ss.mode, buf.getvalue())
    if "pdf" in ss:
        st.text_area("Draft report (preview)", ss.report, height=320)
        st.download_button("Download PDF", ss.pdf, "neurovision_report.pdf", "application/pdf")
