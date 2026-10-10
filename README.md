# NeuroVision AI
Upload MRI -> AI segmentation -> measurements -> LLM draft report -> PDF.
**Research/education demo. Not a medical device. Outputs require specialist review.**

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Input
Four co-registered NIfTI volumes (T1c, T1, T2, FLAIR), ideally skull-stripped (BraTS-style).
Volumes are reoriented to RAS and resampled to 1 mm isotropic (toggle in sidebar).
With fewer than four modalities the app runs in a labelled DEMO mode (placeholder mask).

## Model weights (MONAI `brats_mri_segmentation`, SegResNet)
Loaded in this order: `models/model.pt` -> cached download -> `MODEL_URL` env var -> automatic
`monai.bundle.download` on first run. To ship them in the repo:

    python -c "from monai.bundle import download; download('brats_mri_segmentation', bundle_dir='bundle')"
    cp bundle/brats_mri_segmentation/models/model.pt models/model.pt

## Report LLM
Add `GROQ_API_KEY = "..."` in Streamlit Cloud -> App settings -> Secrets (never commit it).
Without it, a template report is used.

## Deploy
Push to GitHub -> share.streamlit.io -> New app -> select repo, `app.py`.
