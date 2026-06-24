"""
llm_report_generator.py
========================
Ambient Assisted Living - Detailed Daily Activity Report Generator
Uses HuggingFace (local, no API) to generate rich sectioned narrative
PDF reports from DL classifier + autoencoder outputs.

HOW TO RUN (after running dl_model.py first):
    python llm_report_generator.py

Dependencies:
    pip install transformers torch reportlab numpy
    (tensorflow already needed for dl_model.py)
"""

import os
import re
import numpy as np
import datetime
from collections import Counter

import tensorflow as tf

# List GPUs
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Enable memory growth (CRITICAL for Mac stability)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print(f"[INFO] Using Apple GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("[WARNING] No GPU detected, using CPU")


# PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    HRFlowable, Table, TableStyle, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ACTIVITY_NAMES = {
    0: "Walking",
    1: "Walking Upstairs",
    2: "Walking Downstairs",
    3: "Sitting",
    4: "Standing",
    5: "Laying"
}

ACTIVITY_DESCRIPTION = {
    0: "flat-surface walking",
    1: "stair climbing (upward)",
    2: "stair descending (downward)",
    3: "seated / chair rest",
    4: "upright standing without movement",
    5: "lying down / resting horizontally"
}

ACTIVITY_CATEGORY = {
    0: "Active",
    1: "Active",
    2: "Active",
    3: "Sedentary",
    4: "Sedentary",
    5: "Sedentary"
}

ACTIVITY_HEALTH_NOTE = {
    0: "beneficial for cardiovascular health and lower-body strength",
    1: "excellent for building leg muscle endurance and balance",
    2: "engages core stability and joint control",
    3: "common during meals, work, or relaxation; extended periods should be broken up",
    4: "gentle on joints; prolonged standing without movement can cause fatigue",
    5: "essential for recovery; extended daytime laying may warrant caregiver attention"
}

SESSION_SIZE  = 1000
BRAND_COLOR   = "#1a3a5c"
ACCENT_COLOR  = "#2980b9"
LIGHT_BG      = "#f0f4f8"
WARN_COLOR    = "#c0392b"
OK_COLOR      = "#27ae60"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – LOAD DL MODEL OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def load_dl_outputs(x_test_path, y_test_path,
                    classifier_path="classifier_model.h5",
                    autoencoder_path="autoencoder_model.h5"):
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    print("[INFO] Loading test data ...")
    X_test     = np.loadtxt(x_test_path)
    y_test_raw = np.loadtxt(y_test_path).astype(int) - 1

    print("[INFO] Loading saved models ...")
    classifier  = load_model(classifier_path, compile=False)
    autoencoder = load_model(autoencoder_path, compile=False)

    print("[INFO] Running classifier on GPU...")
    y_pred_proba = classifier.predict(X_test, batch_size=128, verbose=0)

    print("[INFO] Running autoencoder on GPU...")
    reconstructed = autoencoder.predict(X_test, batch_size=128, verbose=0)

    activity_classes  = np.argmax(y_pred_proba, axis=1)
    confidence_scores = np.max(y_pred_proba, axis=1)
    recon_errors      = np.mean(np.square(X_test - reconstructed), axis=1)
    threshold         = np.percentile(recon_errors, 95)
    anomaly_flags     = recon_errors > threshold

    return {
        "true_labels":           y_test_raw,
        "predicted_classes":     activity_classes,
        "confidence_scores":     confidence_scores,
        "reconstruction_errors": recon_errors,
        "anomaly_flags":         anomaly_flags,
        "threshold":             threshold,
        "n_samples":             len(X_test),
        "all_proba":             y_pred_proba,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – SPLIT INTO USER SESSIONS
# ─────────────────────────────────────────────────────────────────────────────

def split_into_sessions(dl_outputs, session_size=SESSION_SIZE):
    n        = dl_outputs["n_samples"]
    sessions = []

    for start in range(0, n, session_size):
        end  = min(start + session_size, n)
        s_id = start // session_size + 1

        pred   = dl_outputs["predicted_classes"][start:end]
        conf   = dl_outputs["confidence_scores"][start:end]
        errors = dl_outputs["reconstruction_errors"][start:end]
        flags  = dl_outputs["anomaly_flags"][start:end]

        activity_counts   = Counter(int(p) for p in pred)
        ranked_activities = activity_counts.most_common()
        dominant_act      = ranked_activities[0][0]
        total             = len(pred)

        active_cnt    = sum(cnt for act, cnt in activity_counts.items()
                            if ACTIVITY_CATEGORY[act] == "Active")
        sedentary_cnt = total - active_cnt
        active_pct    = round(active_cnt / total * 100, 1)
        sedentary_pct = round(100 - active_pct, 1)

        # Per-activity confidence and anomaly counts
        act_conf   = {}
        act_anomaly= {}
        for act in activity_counts:
            mask = pred == act
            act_conf[act]    = float(np.mean(conf[mask])) if mask.any() else 0.0
            act_anomaly[act] = int(np.sum(flags[mask]))

        transitions   = int(np.sum(pred[:-1] != pred[1:]))
        low_conf_cnt  = int(np.sum(conf < 0.60))
        max_recon     = float(np.max(errors))

        sessions.append({
            "session_id":        s_id,
            "n_samples":         total,
            "activity_counts":   activity_counts,
            "ranked_activities": ranked_activities,
            "dominant_activity": dominant_act,
            "mean_confidence":   float(np.mean(conf)),
            "mean_recon_error":  float(np.mean(errors)),
            "max_recon_error":   max_recon,
            "n_anomalies":       int(np.sum(flags)),
            "act_anomaly":       act_anomaly,
            "act_conf":          act_conf,
            "active_pct":        active_pct,
            "sedentary_pct":     sedentary_pct,
            "active_cnt":        active_cnt,
            "sedentary_cnt":     sedentary_cnt,
            "transitions":       transitions,
            "low_conf_cnt":      low_conf_cnt,
        })

    return sessions


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – BUILD DETAILED SECTIONED PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(session):
    s = session

    # Detailed per-activity lines for the prompt context
    act_lines = []
    for act, cnt in s["ranked_activities"]:
        pct  = round(cnt / s["n_samples"] * 100, 1)
        conf = s["act_conf"].get(act, 0.0)
        anom = s["act_anomaly"].get(act, 0)
        act_lines.append(
            f"  - {ACTIVITY_NAMES[act]} ({ACTIVITY_DESCRIPTION[act]}): "
            f"{cnt} readings ({pct}%), avg confidence {conf:.1%}, "
            f"{anom} anomalous readings. Health note: {ACTIVITY_HEALTH_NOTE[act]}."
        )
    act_detail = "\n".join(act_lines)

    # Anomaly severity framing
    anom_rate = s["n_anomalies"] / s["n_samples"] * 100
    if anom_rate == 0:
        anom_label = "No anomalies detected. Movement patterns are entirely within normal range."
    elif anom_rate < 2:
        anom_label = (
            f"Minor anomalies: {s['n_anomalies']} readings ({anom_rate:.1f}%) showed "
            "irregular sensor patterns. Overall behaviour is largely normal."
        )
    elif anom_rate < 6:
        anom_label = (
            f"Moderate anomalies: {s['n_anomalies']} readings ({anom_rate:.1f}%) flagged. "
            "Caregiver should review activity logs for irregular movement bursts."
        )
    else:
        anom_label = (
            f"Elevated anomalies: {s['n_anomalies']} readings ({anom_rate:.1f}%) flagged. "
            "Significant irregular sensor data - medical review or a safety check is recommended."
        )

    # Active/sedentary balance framing
    if s["active_pct"] >= 60:
        balance_note = "The resident maintained a highly active day, which is very positive for cardiovascular and muscular health."
    elif s["active_pct"] >= 40:
        balance_note = "The resident achieved a reasonable balance between active and sedentary time."
    elif s["active_pct"] >= 20:
        balance_note = "The resident leaned significantly sedentary today. Gentle encouragement for more movement is advised."
    else:
        balance_note = "The resident was predominantly sedentary. A caregiver should check whether this rest was intentional or due to discomfort."

    prompt = (
        "You are a professional health assistant for an Ambient Assisted Living (AAL) system. "
        "Your role is to write detailed, compassionate, and medically-aware daily activity reports "
        "for caregivers monitoring elderly residents. Your report must be written in clear, plain "
        "English - warm but professional. Each paragraph must be long and descriptive (at least 80 words each).\n\n"

        f"=== SESSION {s['session_id']} DATA ===\n\n"

        f"OVERVIEW:\n"
        f"- Total sensor readings: {s['n_samples']}\n"
        f"- Activity transitions (how often the activity changed): {s['transitions']}\n"
        f"- Active time: {s['active_pct']}% ({s['active_cnt']} readings)\n"
        f"- Sedentary time: {s['sedentary_pct']}% ({s['sedentary_cnt']} readings)\n"
        f"- Dominant activity: {ACTIVITY_NAMES[s['dominant_activity']]}\n"
        f"- Model average confidence: {s['mean_confidence']:.1%}\n"
        f"- Low-confidence readings (below 60%): {s['low_conf_cnt']}\n\n"

        f"ACTIVITY-BY-ACTIVITY BREAKDOWN:\n"
        f"{act_detail}\n\n"

        f"ANOMALY ANALYSIS:\n"
        f"- {anom_label}\n\n"

        f"BALANCE ASSESSMENT:\n"
        f"{balance_note}\n\n"

        "=== YOUR TASK ===\n\n"

        "Write a detailed daily activity narrative with exactly 4 sections. "
        "Each section must start with its heading on its own line wrapped in double asterisks, "
        "followed immediately by a long detailed paragraph of at least 80 words. "
        "Do not use bullet points inside the paragraphs - write in flowing prose only.\n\n"

        "**1. Overview of the Day's Activities**\n"
        "Describe in rich detail what activities the resident performed, "
        "how frequently each occurred, what they imply about the resident's daily routine, "
        "and how active or restful the day was overall. Mention each specific activity "
        "by name and explain what it physically involves for an elderly person.\n\n"

        "**2. Physical Health Observations**\n"
        "Discuss what the activity pattern reveals about the resident's physical health - "
        "their mobility, endurance, balance, and joint usage. Comment on the active vs "
        "sedentary balance and the health implications. Mention whether the level of "
        "movement is appropriate for an elderly resident and how it compares to "
        "recommended daily activity levels for seniors.\n\n"

        "**3. Anomaly and Sensor Confidence Analysis**\n"
        "Explain what the anomalous readings mean in plain language - could they reflect "
        "unusual movement, a potential health event, or sensor noise? Discuss the model's "
        "confidence level and what low-confidence readings might suggest. Flag any "
        "specific activities where anomalies were concentrated and what this could mean.\n\n"

        "**4. Caregiver Recommendations for Tomorrow**\n"
        "Give specific, detailed, and actionable advice for the caregiver: what activities "
        "to encourage, what warning signs to watch for, whether a wellness check is needed, "
        "and lifestyle adjustments that could improve the resident's health and safety "
        "based on today's activity patterns.\n\n"

        "Begin directly with the first heading. Do not include any preamble or the session "
        "data above in your output.\n\n"

        "Report:\n"
        "**1. Overview of the Day's Activities**\n"
    )
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – LOAD MLX LLM (Apple Silicon Optimized)
# ─────────────────────────────────────────────────────────────────────────────

MLX_MODEL_ID = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"

def load_llm():
    """
    Loads quantized Mistral-7B using MLX.
    
    First run:
        downloads ~4GB
    Next runs:
        loads instantly from cache
    
    Requires:
        pip install mlx-lm
    """
    try:
        from mlx_lm import load
    except ImportError:
        print("[ERROR] mlx-lm not installed. Run: pip install mlx-lm")
        return None, None

    print(f"[INFO] Loading MLX model: {MLX_MODEL_ID}")
    model, tokenizer = load(MLX_MODEL_ID)

    print("[INFO] MLX model loaded successfully.")
    return model, tokenizer


def generate_report_text(model, tokenizer, prompt):
    from mlx_lm import generate

    # Mistral instruction format
    formatted_prompt = f"[INST] {prompt} [/INST]"

    response = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=900,
        verbose=False,
    )

    text = response.strip()

    # Optional trimming
    words = text.split()
    if len(words) > 700:
        truncated = " ".join(words[:700])
        last_dot = truncated.rfind(".")
        if last_dot > 300:
            text = truncated[:last_dot + 1]

    return text


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – PARSE LLM OUTPUT INTO SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def parse_sections(text):
    """
    Split LLM output into (heading, body) pairs.
    Handles **Bold Heading** markdown and numbered headings.
    Falls back to the full text as a single block.
    """
    # Pattern: **Heading** followed by body until the next **Heading** or end
    pattern = r"\*\*(.+?)\*\*\s*\n(.*?)(?=\n\*\*|\Z)"
    matches  = re.findall(pattern, text, re.DOTALL)
    if matches:
        return [(h.strip(), b.strip()) for h, b in matches]

    # Numbered headings: "1. Title\nbody"
    pattern2 = r"(?:^|\n)\d+[\.\)]\s+(.+?)\n(.*?)(?=\n\d+[\.\)]|\Z)"
    matches2  = re.findall(pattern2, text, re.DOTALL)
    if matches2:
        return [(h.strip(), b.strip()) for h, b in matches2]

    # Fallback: whole text as one section
    return [("Activity Report", text.strip())]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 – PDF STYLES
# ─────────────────────────────────────────────────────────────────────────────

def _styles():
    base = getSampleStyleSheet()

    def P(name, **kw):
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    return {
        "cover_title": P("CoverTitle",
            fontSize=22, leading=28, spaceBefore=0, spaceAfter=4,
            textColor=colors.HexColor(BRAND_COLOR),
            fontName="Helvetica-Bold"),

        "cover_sub": P("CoverSub",
            fontSize=10.5, spaceAfter=18,
            textColor=colors.HexColor("#555555")),

        "session_header": P("SessionHeader",
            fontSize=13, leading=18, spaceBefore=20, spaceAfter=6,
            textColor=colors.HexColor(BRAND_COLOR),
            fontName="Helvetica-Bold"),

        "section_heading": P("SectionHeading",
            fontSize=11.5, leading=15, spaceBefore=14, spaceAfter=5,
            textColor=colors.HexColor(ACCENT_COLOR),
            fontName="Helvetica-Bold"),

        "body": P("Body",
            fontSize=10.5, leading=17, spaceAfter=6,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#1a1a1a")),

        "table_cell": P("TableCell",
            fontSize=9, textColor=colors.HexColor("#222222"),
            leading=12),

        "caption": P("Caption",
            fontSize=8.5, textColor=colors.HexColor("#666666"),
            spaceAfter=3, fontName="Helvetica-Oblique"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 – TABLE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _summary_table(s, st):
    """Wide two-column-pair metrics table."""
    def cell(txt, bold=False):
        fn = "Helvetica-Bold" if bold else "Helvetica"
        return Paragraph(f'<font name="{fn}">{txt}</font>', st["table_cell"])

    header = [cell("Metric", bold=True), cell("Value", bold=True),
               cell("Metric", bold=True), cell("Value", bold=True)]

    pairs = [
        ("Total Sensor Readings",     str(s["n_samples"]),
         "Activity Transitions",       str(s["transitions"])),
        ("Active Time",               f"{s['active_pct']}%  ({s['active_cnt']} readings)",
         "Sedentary Time",             f"{s['sedentary_pct']}%  ({s['sedentary_cnt']} readings)"),
        ("Dominant Activity",         ACTIVITY_NAMES[s["dominant_activity"]],
         "Unique Activities",          str(len(s["activity_counts"]))),
        ("Avg Model Confidence",      f"{s['mean_confidence']:.1%}",
         "Low-Confidence Readings",    str(s["low_conf_cnt"])),
        ("Total Anomalies",           str(s["n_anomalies"]),
         "Anomaly Rate",               f"{s['n_anomalies']/s['n_samples']*100:.1f}%"),
        ("Max Reconstruction Error",  f"{s['max_recon_error']:.5f}",
         "Avg Reconstruction Error",   f"{s['mean_recon_error']:.5f}"),
    ]

    rows = [header]
    for l1, v1, l2, v2 in pairs:
        rows.append([cell(l1), cell(v1, bold=True),
                      cell(l2), cell(v2, bold=True)])

    tbl = Table(rows, colWidths=[2.0*inch, 1.65*inch, 2.0*inch, 1.65*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor(BRAND_COLOR)),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor(LIGHT_BG), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.35, colors.HexColor("#cccccc")),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 7),
        ("LINEBELOW",      (0, 0), (-1, 0), 1.5, colors.HexColor(ACCENT_COLOR)),
    ]))
    return tbl


def _activity_breakdown_table(s, st):
    """Per-activity table: count, %, confidence, anomalies, category, health note."""

    def cell(txt):
        return Paragraph(txt, st["table_cell"])

    header = [
        cell("<b>Activity</b>"),
        cell("<b>Readings</b>"),
        cell("<b>Share</b>"),
        cell("<b>Avg Conf.</b>"),
        cell("<b>Anomalies</b>"),
        cell("<b>Category</b>"),
    ]
    rows = [header]

    for act, cnt in s["ranked_activities"]:
        pct   = cnt / s["n_samples"] * 100
        conf  = s["act_conf"].get(act, 0.0)
        anom  = s["act_anomaly"].get(act, 0)
        cat   = ACTIVITY_CATEGORY[act]
        cat_c = OK_COLOR if cat == "Active" else "#e67e22"
        an_c  = WARN_COLOR if anom > 0 else OK_COLOR

        rows.append([
            cell(ACTIVITY_NAMES[act]),
            cell(str(cnt)),
            cell(f"{pct:.1f}%"),
            cell(f"{conf:.1%}"),
            cell(f'<font color="{an_c}"><b>{anom}</b></font>'),
            cell(f'<font color="{cat_c}"><b>{cat}</b></font>'),
        ])

    tbl = Table(rows,
                colWidths=[1.75*inch, 0.82*inch, 0.7*inch, 0.82*inch, 0.82*inch, 0.82*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor(ACCENT_COLOR)),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor(LIGHT_BG), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.35, colors.HexColor("#cccccc")),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 7),
        ("LINEBELOW",      (0, 0), (-1, 0), 1.5, colors.white),
    ]))
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 – SAVE PDF
# ─────────────────────────────────────────────────────────────────────────────

def safe_para(text, style):
    """Escape XML special characters and return a Paragraph."""
    text = (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))
    return Paragraph(text, style)


def save_pdf(sessions, reports, output_path):
    doc   = SimpleDocTemplate(
        output_path, pagesize=letter,
        leftMargin=0.85*inch, rightMargin=0.85*inch,
        topMargin=0.9*inch,   bottomMargin=0.9*inch
    )
    st    = _styles()
    story = []

    # Cover
    story.append(Paragraph("Ambient Assisted Living System", st["cover_title"]))
    story.append(Paragraph("Daily Activity Report", st["cover_title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%A, %d %B %Y  at  %H:%M')}   |   "
        f"Total Sessions: {len(sessions)}   |   UCI HAR Dataset",
        st["cover_sub"]
    ))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor(BRAND_COLOR), spaceAfter=20))

    for session, report_text in zip(sessions, reports):
        s = session

        anom_flag = (
            "  [WARNING: ANOMALIES FLAGGED]"
            if s["n_anomalies"] > s["n_samples"] * 0.05 else ""
        )

        # Session header
        story.append(Paragraph(
            f"Session {s['session_id']}  |  "
            f"Dominant: {ACTIVITY_NAMES[s['dominant_activity']]}  |  "
            f"Active {s['active_pct']}%  /  Sedentary {s['sedentary_pct']}%"
            f"{anom_flag}",
            st["session_header"]
        ))
        story.append(HRFlowable(width="100%", thickness=0.8,
                                 color=colors.HexColor(ACCENT_COLOR), spaceAfter=8))

        # Table 1: Session metrics
        story.append(Paragraph("Session Metrics Summary", st["caption"]))
        story.append(_summary_table(s, st))
        story.append(Spacer(1, 10))

        # Table 2: Activity breakdown
        story.append(Paragraph("Activity-by-Activity Breakdown", st["caption"]))
        story.append(_activity_breakdown_table(s, st))
        story.append(Spacer(1, 16))

        # LLM Narrative
        story.append(Paragraph("Detailed Activity Narrative", st["caption"]))
        story.append(HRFlowable(width="100%", thickness=0.4,
                                 color=colors.HexColor("#cccccc"), spaceAfter=8))

        sections = parse_sections(report_text)

        if sections:
            for heading, body in sections:
                # Clean heading of numbering like "1. " if present
                clean_heading = re.sub(r"^\d+[\.\)]\s*", "", heading).strip()
                story.append(Paragraph(clean_heading, st["section_heading"]))

                # Body: split on double newlines into sub-paragraphs
                sub_paras = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
                if not sub_paras:
                    sub_paras = [body.strip()]

                for sp in sub_paras:
                    # Further clean single-line breaks within a paragraph
                    sp_clean = " ".join(line.strip() for line in sp.split("\n") if line.strip())
                    if sp_clean:
                        story.append(safe_para(sp_clean, st["body"]))

                story.append(Spacer(1, 6))
        else:
            # Fallback: dump all text line by line
            for line in report_text.split("\n"):
                line = line.strip()
                if line:
                    story.append(safe_para(line, st["body"]))

        story.append(Spacer(1, 12))
        story.append(HRFlowable(width="100%", thickness=1.2,
                                 color=colors.HexColor(BRAND_COLOR), spaceAfter=20))

    doc.build(story)
    print(f"[INFO] PDF saved -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
    # X_TEST_PATH   = os.path.join(BASE_DIR, "UCI HAR Dataset", "test", "X_test.txt")
    # Y_TEST_PATH   = os.path.join(BASE_DIR, "UCI HAR Dataset", "test", "y_test.txt")
    # CLASSIFIER_H5 = os.path.join(BASE_DIR, "classifier_model.h5")
    # AUTOENC_H5    = os.path.join(BASE_DIR, "autoencoder_model.h5")
    # PDF_OUTPUT    = os.path.join(BASE_DIR, "daily_activity_report.pdf")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # go outside dl_model__llm folder
    PROJECT_ROOT = os.path.dirname(BASE_DIR)

    DATASET_DIR = os.path.join(
        PROJECT_ROOT,
        "UCI HAR Dataset"
    )

    X_TEST_PATH = os.path.join(DATASET_DIR, "test", "X_test.txt")
    Y_TEST_PATH = os.path.join(DATASET_DIR, "test", "y_test.txt")

    CLASSIFIER_H5 = os.path.join(BASE_DIR, "classifier_model.h5")
    AUTOENC_H5    = os.path.join(BASE_DIR, "autoencoder_model.h5")

    PDF_OUTPUT = os.path.join(BASE_DIR, "daily_activity_report.pdf")

    # 1. DL inference
    dl_outputs = load_dl_outputs(X_TEST_PATH, Y_TEST_PATH, CLASSIFIER_H5, AUTOENC_H5)

    # 2. Session splitting (every 1000 rows = 1 session)
    sessions = split_into_sessions(dl_outputs, session_size=SESSION_SIZE)
    print(f"[INFO] {len(sessions)} sessions from {dl_outputs['n_samples']} samples.")

    # 3. Load local LLM — swap model_name for a larger model if you have a GPU
    # generator = load_llm("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # generator = load_llm("mistralai/Mistral-7B-Instruct-v0.2")   # richer text
    model, tokenizer = load_llm()

    # 4. Generate one detailed sectioned report per session
    reports = []
    for s in sessions:
        print(f"[INFO] Generating report for session {s['session_id']} ...")
        prompt = build_prompt(s)
        text = generate_report_text(model, tokenizer, prompt)
        reports.append(text)

        secs = parse_sections(text)
        print(f"       -> {len(text.split())} words | {len(secs)} sections parsed.")

    # 5. Save PDF
    save_pdf(sessions, reports, PDF_OUTPUT)
    print(f"\n[DONE] Report -> {PDF_OUTPUT}")


if __name__ == "__main__":
    main()