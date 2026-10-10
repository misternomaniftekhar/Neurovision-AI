import os
import tempfile
import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output

# Channel order expected by the MONAI BraTS bundle (from its metadata.json)
MODALITIES = ["T1c", "T1", "T2", "FLAIR"]


def voxel_sizes(img):
    """Voxel size in mm, derived from the affine."""
    return tuple(float(z) for z in np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0)))


def load_nifti(uploaded, resample=True):
    """Load a Streamlit UploadedFile -> (RAS-oriented, optionally 1 mm resampled) NIfTI + float32 array."""
    name = uploaded.name.lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        raise ValueError(f"{uploaded.name}: only .nii / .nii.gz files are supported.")
    suffix = ".nii.gz" if name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(uploaded.getbuffer())
        path = f.name
    try:
        img = nib.as_closest_canonical(nib.load(path))
        arr = np.asarray(img.dataobj, dtype=np.float32)
        affine = img.affine
    finally:
        os.unlink(path)

    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"{uploaded.name}: expected a 3D volume, got shape {arr.shape}.")
    arr = np.nan_to_num(arr)
    if arr.max() == arr.min():
        raise ValueError(f"{uploaded.name}: volume is empty/constant.")

    img = nib.Nifti1Image(arr, affine)
    resampled = False
    if resample and any(abs(z - 1.0) > 0.05 for z in voxel_sizes(img)):
        img = resample_to_output(img, voxel_sizes=1.0, order=1)
        arr = np.nan_to_num(np.asarray(img.dataobj, dtype=np.float32))
        img = nib.Nifti1Image(arr, img.affine)
        resampled = True
    return img, arr, resampled


def load_modalities(files: dict, resample=True):
    """files: {modality: UploadedFile or None} -> (reference_img, {modality: ndarray}, info)."""
    vols, ref, info = {}, None, {"resampled": False, "orig_voxel_mm": None}
    for m, f in files.items():
        if f is None:
            continue
        img, data, rs = load_nifti(f, resample)
        if ref is None:
            ref = img
            info["resampled"] = rs
        elif data.shape != next(iter(vols.values())).shape:
            raise ValueError(f"{m}: shape {data.shape} does not match the other modalities "
                             "(scans must be co-registered on the same grid).")
        vols[m] = data
    if not vols:
        raise ValueError("No files uploaded.")
    return ref, vols, info
