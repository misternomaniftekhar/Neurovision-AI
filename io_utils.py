import tempfile
import nibabel as nib
import numpy as np

# Channel order expected by the MONAI BraTS bundle
MODALITIES = ["T1c", "T1", "T2", "FLAIR"]


def load_nifti(uploaded):
    """Load a Streamlit UploadedFile as a canonical-orientation (RAS) NIfTI."""
    name = uploaded.name.lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        raise ValueError(f"{uploaded.name}: only .nii / .nii.gz files are supported.")
    suffix = ".nii.gz" if name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(uploaded.getbuffer())
        path = f.name
    img = nib.as_closest_canonical(nib.load(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{uploaded.name}: expected a 3D volume, got shape {data.shape}.")
    data = np.nan_to_num(data)
    if data.max() == data.min():
        raise ValueError(f"{uploaded.name}: volume is empty/constant.")
    return img, data


def load_modalities(files: dict):
    """files: {modality: UploadedFile or None}. Returns (reference_img, {modality: ndarray})."""
    vols, ref = {}, None
    for m, f in files.items():
        if f is None:
            continue
        img, data = load_nifti(f)
        if ref is None:
            ref = img
        elif data.shape != next(iter(vols.values())).shape:
            raise ValueError(f"{m}: shape {data.shape} does not match other modalities.")
        vols[m] = data
    if not vols:
        raise ValueError("No files uploaded.")
    return ref, vols
