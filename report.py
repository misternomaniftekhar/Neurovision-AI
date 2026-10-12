"""Draft report text (optional Groq LLM, template fallback) and PDF export."""
import io
from datetime import datetime

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from model_utils import pretty

DISCLAIMER = ("AI-GENERATED DRAFT FOR RESEARCH/EDUCATION. This output comes from an automated image classifier "
              "trained on a public dataset. It has not been clinically validated and must not be used for "
              "diagnosis or treatment decisions. A qualified radiologist or specialist must review the images.")

PROMPT = """Draft a short, neutral report from the output of an MRI slice classifier.
Use ONLY the numbers given. Do not diagnose, stage, name causes, or recommend treatment.
Sections: Model output, Confidence and flags, Limitations, Suggested follow-up for specialist review.
Input: {facts}"""


def _facts(scan_name, res, meta):
    probs = ", ".join(f"{pretty(c)} {p:.1%}" for c, p in zip(meta["classes"], res["probs"]))
    m = meta.get("metrics", {})
    return (f"scan={scan_name}; predicted={pretty(res['label'])} ({res['confidence']:.1%}); probabilities: {probs}; "
            f"flags={res['flags'] or 'none'}; model test accuracy={m.get('test_accuracy', 'n/a')}; "
            "input is a single 2D slice")


def make_draft_text(scan_name, res, meta, groq_key=None):
    if groq_key:
        try:
            from groq import Groq
            r = Groq(api_key=groq_key).chat.completions.create(
                model="llama-3.3-70b-versatile", temperature=0.2,
                messages=[{"role": "user", "content": PROMPT.format(facts=_facts(scan_name, res, meta))}])
            return r.choices[0].message.content
        except Exception:
            pass
    flags = "; ".join(res["flags"]) if res["flags"] else "none"
    rows = "\n".join(f"- {pretty(c)}: {p:.1%}" for c, p in zip(meta["classes"], res["probs"]))
    acc = meta.get("metrics", {}).get("test_accuracy")
    ctx = f" The model scored {acc:.1%} accuracy on its held-out test split." if isinstance(acc, float) else ""
    return (f"Model output\nThe classifier assigned this slice to '{pretty(res['label'])}' with calibrated "
            f"probability {res['confidence']:.1%}.\n{rows}\n\nConfidence and flags\nFlags: {flags}.{ctx}\n\n"
            "Limitations\nSingle 2D slice; no 3D context, no localisation, no staging. Trained on one public dataset, "
            "so performance on other scanners, protocols or populations is unknown. Grad-CAM shows where the model "
            "looked, not proof that its answer is right.\n\nSuggested follow-up\nReview by a qualified specialist.")


def _png(arr_or_img):
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    b = io.BytesIO()
    img.convert("RGB").save(b, format="PNG")
    b.seek(0)
    return b


def build_pdf(draft, scan_name, res, meta, original, cam_overlay):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm)
    s = getSampleStyleSheet()
    warn = s["Normal"].clone("warn", textColor=colors.HexColor("#B00020"), fontSize=9)
    el = [Paragraph("NeuroVision AI - Draft MRI Classification Report", s["Title"]),
          Paragraph(f"Scan: {scan_name} | Generated {datetime.now():%Y-%m-%d %H:%M}", s["Normal"]),
          Spacer(1, 6), Paragraph(DISCLAIMER, warn), Spacer(1, 8)]
    rows = [["Class", "Calibrated probability"]] + [[pretty(c), f"{p:.1%}"] for c, p in zip(meta["classes"], res["probs"])]
    t = Table(rows, colWidths=[80 * mm, 60 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3b57")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey)]))
    el += [t, Spacer(1, 8)]
    if res["flags"]:
        el += [Paragraph("Flags: " + "; ".join(res["flags"]), warn), Spacer(1, 6)]
    img_t = Table([[RLImage(_png(original.resize((224, 224))), 62 * mm, 62 * mm),
                    RLImage(_png(cam_overlay), 62 * mm, 62 * mm)],
                   ["Input slice", "Grad-CAM (model attention)"]], colWidths=[80 * mm, 80 * mm])
    img_t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("FONTSIZE", (0, 1), (-1, 1), 8)]))
    el += [img_t, Spacer(1, 8)]
    for line in draft.split("\n"):
        safe = line.replace("&", "&amp;").replace("<", "&lt;")
        el.append(Paragraph(safe or "&nbsp;", s["Normal"]))
    doc.build(el)
    return buf.getvalue()
