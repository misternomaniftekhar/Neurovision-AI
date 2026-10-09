import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

DISCLAIMER = ("AI-GENERATED DRAFT. This report was produced automatically from model output, "
              "has not been clinically validated, and must be reviewed by a qualified "
              "radiologist or specialist before any use. Not for diagnosis or treatment decisions.")

PROMPT = """You are drafting a structured neuro-imaging report from automated segmentation output.
Use ONLY the numbers given. Do not diagnose, name a tumor type, or recommend treatment.
Sections: Findings, Measurements, Limitations, Suggested follow-up for specialist review.
Be concise and neutral. Mode: {mode}. Measurements: {stats}"""


def generate_report_text(stats, mode, groq_key=None):
    if groq_key:
        try:
            from groq import Groq
            r = Groq(api_key=groq_key).chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": PROMPT.format(mode=mode, stats=stats)}],
                temperature=0.2)
            return r.choices[0].message.content
        except Exception:
            pass  # fall back to template
    p = stats["parts_ml"]
    return (f"Findings\nAutomated segmentation ({mode} mode) identified a region of total volume "
            f"{stats['whole_tumor_ml']} mL.\n\nMeasurements\n"
            + "\n".join(f"- {k}: {v} mL" for k, v in p.items())
            + f"\n- Tumor core (non-edema): {stats['tumor_core_ml']} mL"
            + f"\n- Voxel size: {stats['voxel_size_mm']} mm\n\nLimitations\n"
            "Automated output; sensitive to image quality, modality completeness and registration."
            "\n\nSuggested follow-up\nSpecialist review of the images and mask.")


def build_pdf(report_text, stats, mode, preview_png=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=16*mm, bottomMargin=16*mm)
    s = getSampleStyleSheet()
    warn = s["Normal"].clone("warn", textColor=colors.HexColor("#B00020"), fontSize=9)
    el = [Paragraph("NeuroVision AI - Draft MRI Segmentation Report", s["Title"]),
          Paragraph(datetime.now().strftime("Generated %Y-%m-%d %H:%M"), s["Normal"]),
          Spacer(1, 6), Paragraph(DISCLAIMER, warn), Spacer(1, 10)]
    if mode == "demo":
        el += [Paragraph("<b>DEMO MODE: mask is a placeholder heuristic, not a trained model.</b>", warn),
               Spacer(1, 6)]
    rows = [["Measure", "Value"], ["Whole tumor", f"{stats['whole_tumor_ml']} mL"],
            ["Tumor core", f"{stats['tumor_core_ml']} mL"]]
    rows += [[k, f"{v} mL"] for k, v in stats["parts_ml"].items()]
    t = Table(rows, colWidths=[90*mm, 60*mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey)]))
    el += [t, Spacer(1, 10)]
    if preview_png:
        el += [Image(io.BytesIO(preview_png), width=150*mm, height=50*mm), Spacer(1, 10)]
    for line in report_text.split("\n"):
        safe = line.replace("&", "&amp;").replace("<", "&lt;")
        el.append(Paragraph(safe or "&nbsp;", s["Normal"]))
    doc.build(el)
    return buf.getvalue()
