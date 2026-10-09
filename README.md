# NeuroVision AI
Upload MRI -> AI segmentation -> tumor measurements -> LLM draft report -> PDF.
**Research/education demo. Not a medical device. Outputs require specialist review.**

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Model weights
    python -c "from monai.bundle import download; download('brats_mri_segmentation', bundle_dir='bundle')"
    cp bundle/brats_mri_segmentation/models/model.pt models/model.pt
For Streamlit Cloud, host the file (GitHub Release / Hugging Face) and set env var `MODEL_URL`.

## Report LLM
Add `GROQ_API_KEY = "..."` in Streamlit Cloud -> App settings -> Secrets. Without it, a template report is used.

## Deploy
Push to GitHub -> share.streamlit.io -> New app -> select repo, `app.py`.
