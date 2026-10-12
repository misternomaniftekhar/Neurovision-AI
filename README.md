# NeuroVision AI - MRI classification workspace

Upload MRI slice → AI classification (EfficientNet-B0, calibrated probabilities) → Grad-CAM explainability →
draft report → PDF. Streamlit app with sections: Overview, New Scan & Samples, MRI Workspace, Explainability,
Model Lab, Scan History, Reports, Docs & Ethics.

**Research/education only. Not a medical device. Every output needs specialist review.**

## 1. Dataset layout (Kaggle "Brain Tumor MRI Dataset" style)
    <DATA>/Training/{glioma,meningioma,notumor,pituitary}/*.jpg
    <DATA>/Testing/{glioma,meningioma,notumor,pituitary}/*.jpg
A single folder of class sub-folders also works (it is split 70/15/15). Class names are read from the folder names.

## 2. Train (GPU recommended: Kaggle/Colab free GPU; CPU works but is slow)
    pip install -r requirements-train.txt
    python train.py --data /path/to/dataset --epochs 15
Pretrained ImageNet weights are downloaded automatically (needs internet). Useful flags:
`--batch-size 32  --lr 5e-4  --img-size 224  --patience 5  --max-per-class 50` (the last one is for quick tests;
add `--no-pretrained` only for offline smoke tests).

Outputs:
- `models/model.pt` + `models/meta.json` (classes, temperature, metrics) - the app reads these
- `reports/confusion_matrix.png`, `reports/training_curves.png`, `reports/metrics.json`

**Duplicate filter (on by default, `--no-dedupe` to skip).** Before splitting, `train.py` removes near-identical
images: duplicate copies inside the training pool, and any test image that is a near-copy of a training image.
The 7,200-image Kaggle set I checked had about 420 of its 1,600 Testing images duplicated in Training and roughly
640 duplicate Training images (the `notumor` class most of all), which would have inflated the test score.
After filtering, some test classes are small, and the app warns when per-class scores are based on few images.

What the script does: stratified train/val split, class-weighted loss with label smoothing, AdamW + OneCycle,
mixed precision on GPU, early stopping on validation macro-F1, **temperature scaling** on validation data so
the confidence shown in the app is calibrated, and a final evaluation on the held-out Testing set.

## 3. Benchmark scans for the sidebar
    python make_samples.py --data /path/to/dataset --n 3
Copies a few Testing images per class into `samples/`, skipping any that duplicate a training image. They appear under "Benchmark scans".

## 4. Run the app
    pip install -r requirements.txt
    streamlit run app.py

## 5. Deploy (Streamlit Community Cloud)
1. Commit `models/model.pt` and `models/meta.json` (about 16 MB), `reports/*.png`, and `samples/` (a few images).
2. Push to GitHub, then share.streamlit.io → New app → `app.py`.
3. Optional: add `GROQ_API_KEY = "..."` under Secrets for LLM-written report text (otherwise a template is used).
Never commit API keys.

## Files
    app.py            multi-section Streamlit app
    train.py          training + evaluation + calibration
    model_utils.py    model, preprocessing, calibrated prediction, Grad-CAM (shared)
    report.py         draft text + PDF
    make_samples.py   fills samples/ from your dataset
    requirements.txt  app dependencies   |  requirements-train.txt  adds scikit-learn

## Notes on honesty
- This is a 2D slice classifier. It does not segment tumours, measure volume, or use 3D context.
- Any image, even a non-MRI one, is forced into one of the trained classes. The app flags low confidence and
  colour images, which is not a guarantee.
- Public Kaggle MRI sets can contain duplicate or mislabelled slices, so test accuracy may be optimistic.
