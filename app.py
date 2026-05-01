# Blood Cancer AI System - UI
import io
import os
import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image as RLImage, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle)

matplotlib.use("Agg")  # non-GUI backend

st.set_page_config(page_title="Blood Cancer AI System",
    page_icon="🩸", layout="centered", initial_sidebar_state="collapsed")

# CONSTANTS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "binary"     : os.path.join(BASE_DIR, "models", "binary_classification_final_pipeline.pkl"),
    "multiclass" : os.path.join(BASE_DIR, "models", "multiclass_final_pipeline.pkl"),
    "regression" : os.path.join(BASE_DIR, "models", "regression_final_pipeline.pkl"),
}

SEVERITY_LABELS = {0: "Low", 1: "Medium", 2: "High"}
SEVERITY_TARGET  = {"Low": 0, "Medium": 1, "High": 2}
CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
          "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh",]
LAB_REFS = {
    "WBC"        : {"unit": "cells/uL", "low": 4_000,   "high": 11_000},
    "Haemoglobin": {"unit": "g/dL",     "low": 12.0,    "high": 17.5},
    "Platelets"  : {"unit": "cells/uL", "low": 150_000, "high": 400_000},
    "RBC"        : {"unit": "M/uL",     "low": 4.2,     "high": 5.9},
}

def init_state():
    # Define default values
    defaults = {
        "predicted": False,
        "cancer_detected": False,
        "severity_label": "",
        "cost_estimated": False,
        "estimated_cost": 0.0,
        "city": CITIES[0],
        "risk_score": 0.0,
        "eng_features": {},
        "snap": {},
        "patient_name": "",
        "theme": "Dark"}

    for key in defaults:
        if key not in st.session_state:
            st.session_state[key] = defaults[key]

init_state()

# THEME CSS
THEME_VARS: dict[str, dict] = {
    "Dark": {
        "--bg-deep"  : "#080e1a",
        "--bg-card"  : "#0d1526",
        "--bg-card2" : "#111d35",
        "--text-main": "#e8eaf2",
        "--text-mute": "#7a8aa8",
        "--border"   : "rgba(0,201,255,0.15)",
        "--input-bg" : "#111d35",
        "--metric-bg": "#111d35",
    },
    "Light": {
        "--bg-deep"  : "#f0f4ff",
        "--bg-card"  : "#ffffff",
        "--bg-card2" : "#eef2fb",
        "--text-main": "#0d1a3a",
        "--text-mute": "#556080",
        "--border"   : "rgba(0,100,200,0.18)",
        "--input-bg" : "#f5f8ff",
        "--metric-bg": "#eef2fb",
    },
    "System": {},
}

SHARED_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --accent  : #00c9ff;
    --accent2 : #0077ff;
    --success : #00e5a0;
    --danger  : #ff4d6d;
    --warn    : #ffb347;
    --radius  : 16px;
}

html, body, .stApp {
    background : var(--bg-deep) !important;
    color      : var(--text-main);
    font-family: 'DM Sans', sans-serif;
}

/*  Only hide footer + "Made with Streamlit" badge — NOT the header.
   Hiding `header` also hides the sidebar toggle on some Streamlit versions. */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

.block-container { padding: 2rem 3rem !important; max-width: 1280px; }

/* ── Hero ── */
.hero { text-align: center; padding: 2.5rem 1rem 1.5rem; }
.hero-badge {
    display: inline-block;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    color: #fff; font-size: .68rem; font-weight: 600;
    letter-spacing: .18em; text-transform: uppercase;
    padding: .3rem 1rem; border-radius: 50px; margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(1.8rem, 4.5vw, 3rem);
    color: var(--text-main); margin: 0 0 .4rem; line-height: 1.1;
}
.hero h1 span { color: var(--accent); }
.hero p { color: var(--text-mute); font-size: .95rem; max-width: 540px; margin: 0 auto; }

/* ── Step bar ── */
.step-bar {
    display: flex; justify-content: center;
    gap: 0; margin: 1.5rem auto 0; max-width: 620px;
}
.step-item {
    display: flex; flex-direction: column; align-items: center;
    flex: 1; position: relative;
}
.step-item:not(:last-child)::after {
    content: ''; position: absolute; top: 14px; left: 50%;
    width: 100%; height: 2px; background: var(--border);
}
.step-item.done::after   { background: var(--success); }
.step-item.active::after { background: var(--accent); }
.step-dot {
    width: 28px; height: 28px; border-radius: 50%;
    border: 2px solid var(--border); background: var(--bg-card2);
    display: flex; align-items: center; justify-content: center;
    font-size: .7rem; font-weight: 600; color: var(--text-mute);
    position: relative; z-index: 1;
}
.step-item.done   .step-dot { border-color:var(--success); color:var(--success); background:rgba(0,229,160,.1); }
.step-item.active .step-dot { border-color:var(--accent);  color:var(--accent);  background:rgba(0,201,255,.1); }
.step-label { font-size: .62rem; color: var(--text-mute); margin-top: .35rem; text-align: center; }
.step-item.done   .step-label { color: var(--success); }
.step-item.active .step-label { color: var(--accent); }

/* ── Section title ── */
.section-title {
    font-family: 'DM Serif Display', serif; font-size: 1.15rem;
    color: var(--accent); border-bottom: 1px solid var(--border);
    padding-bottom: .5rem; margin-bottom: 1.25rem;
}

/* ── Glass card ── */
.glass-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.75rem; margin-bottom: 1.5rem;
}

/* ── Patient banner ── */
.patient-banner {
    background: linear-gradient(90deg, rgba(0,119,255,.12), rgba(0,201,255,.08));
    border: 1px solid var(--accent2); border-radius: 12px;
    padding: .75rem 1.25rem; margin-bottom: 1.25rem;
    font-size: .9rem; color: var(--text-main);
    display: flex; align-items: center; gap: .75rem;
}
.patient-banner strong { color: var(--accent); }

/* ── Gauge grid ── */
.gauge-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 1rem; margin-bottom: 1.5rem;
}
.gauge-card {
    background: var(--bg-card2); border: 1px solid var(--border);
    border-radius: 12px; padding: 1rem; text-align: center;
}
.gauge-label { font-size: .68rem; color: var(--text-mute); text-transform: uppercase; letter-spacing: .1em; }
.gauge-value { font-size: 1.55rem; font-weight: 600; margin: .2rem 0; }
.gauge-unit  { font-size: .68rem; color: var(--text-mute); }
.gauge-bar-bg { background: rgba(128,128,128,.15); border-radius: 4px; height: 4px; margin-top: .5rem; overflow: hidden; }
.gauge-bar-fill { height: 4px; border-radius: 4px; }
.normal   { color: var(--success); }
.abnormal { color: var(--danger); }
.gauge-ref { font-size: .6rem; color: var(--text-mute); margin-top: .3rem; display: block; }

/* ── Result banners ── */
.result-banner {
    border-radius: var(--radius); padding: 1.4rem 2rem;
    display: flex; align-items: center; gap: 1.25rem;
    margin-bottom: 1.5rem; border: 1px solid;
}
.result-ok  { background: rgba(0,229,160,.07); border-color: var(--success); color: var(--success); }
.result-bad { background: rgba(255,77,109,.07); border-color: var(--danger);  color: var(--danger); }
.result-icon  { font-size: 2.4rem; }
.result-title { font-size: 1.25rem; font-weight: 600; margin: 0; }
.result-sub   { font-size: .85rem; color: var(--text-mute); margin: .2rem 0 0; }

/* ── Severity pill ── */
.severity-pill {
    display: inline-block; padding: .3rem 1.1rem;
    border-radius: 50px; font-weight: 600; font-size: .88rem; letter-spacing: .04em;
}
.sev-low    { background: rgba(0,229,160,.12); color: var(--success); border: 1px solid var(--success); }
.sev-medium { background: rgba(255,179,71,.12); color: var(--warn);   border: 1px solid var(--warn); }
.sev-high   { background: rgba(255,77,109,.12); color: var(--danger); border: 1px solid var(--danger); }

/* ── Cost display ── */
.cost-display {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2rem, 5vw, 3rem); color: var(--accent);
    text-align: center; padding: 1rem 0 .25rem;
}
.cost-note { text-align: center; font-size: .8rem; color: var(--text-mute); margin-bottom: 1rem; }

/* ── Alert box ── */
.alert-box {
    border-radius: 10px; padding: .9rem 1.2rem;
    font-size: .84rem; margin-bottom: 1rem; border: 1px solid;
}
.alert-warn { background: rgba(255,179,71,.08); border-color: var(--warn); color: var(--warn); }

/* ── Streamlit widget overrides ── */
.stNumberInput input,
div[data-baseweb="select"] > div,
.stTextInput input {
    background: var(--input-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-main) !important;
}
label {
    color: var(--text-mute) !important; font-size: .78rem !important;
    font-weight: 500 !important; text-transform: uppercase; letter-spacing: .05em;
}
[data-testid="stMetric"] {
    background: var(--metric-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important; padding: .75rem 1rem !important;
}
[data-testid="stMetricLabel"] { color: var(--text-mute) !important; }
[data-testid="stMetricValue"] { color: var(--text-main) !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: .95rem !important; padding: .65rem 2rem !important;
    width: 100%; letter-spacing: .04em; transition: opacity .2s, transform .15s;
}
.stButton > button:hover  { opacity: .86; transform: translateY(-1px); }
.stButton > button:active { transform: scale(.98); }

.stDownloadButton > button {
    background: transparent !important; color: var(--accent) !important;
    border: 1px solid var(--accent) !important; border-radius: 10px !important;
    font-weight: 600 !important; width: 100%; transition: background .2s;
}
.stDownloadButton > button:hover { background: rgba(0,201,255,.1) !important; }

hr { border-color: var(--border) !important; margin: 1.5rem 0; }
.footer { text-align: center; color: var(--text-mute); font-size: .75rem; padding: 2rem 0 1rem; }
"""

# This function applies a theme (colors/styles) 
def apply_theme(theme):
    css_variables = ""
    if theme != "System" and theme in THEME_VARS:
        css_list = []
        for key, value in THEME_VARS[theme].items():
            css_list.append(f"{key}: {value};")
        css_text = "\n".join(css_list)
        css_variables = f":root {{\n{css_text}\n}}"
    st.markdown(f"<style>\n{css_variables}\n{SHARED_CSS}\n</style>",
                unsafe_allow_html=True)

# SIDEBAR
def render_sidebar():
    st.sidebar.title("🩸 Blood Cancer AI")
    st.sidebar.caption("Clinical Decision Support System")
    st.sidebar.divider()
    st.sidebar.subheader("🎨 Choose Theme Theme")
    theme = st.sidebar.radio(
        "Theme",
        options=["Dark", "Light", "System"],
        index=["Dark", "Light", "System"].index(st.session_state["theme"]),
        horizontal=True, label_visibility="collapsed")
    
    if theme != st.session_state["theme"]:
        st.session_state["theme"] = theme
        st.rerun()
    st.sidebar.divider()

    st.sidebar.subheader("📋 How to Use")
    steps = [
        ("1️⃣ Patient Details",
         "Enter the patient's **name**, age, gender, and blood-test values "
         "(WBC, RBC, Haemoglobin, Platelets). Then select any symptoms present."),
        ("2️⃣ Live Lab Gauges",
         "Four cards update as you type. "
         "🟢 Green = normal range. 🔴 Red = out of range."),
        ("3️⃣ Run Prediction",
         "Click **Run Diagnostic Prediction**. The binary model decides "
         "Cancer / No Cancer; if cancer is found the multiclass model "
         "classifies severity automatically."),
        ("4️⃣ Review Result",
         "✅ Green banner = no cancer detected (follow-up in 6–12 months). "
         "⚠️ Red banner = malignancy markers found (oncology referral)."),
        ("5️⃣ Select City",
         "If cancer is detected, choose the patient's city. "
         "Treatment costs vary significantly by location."),
        ("6️⃣ Estimate Cost",
         "Click **Estimate Treatment Cost**. The gradient-boosting regressor "
         "predicts total cost in ₹ based on profile, severity, and city."),
        ("7️⃣ Download Report",
         "Click **Download PDF Report** for a complete medical summary "
         "including patient name, diagnosis, severity, cost, and chart."),]
    
    for title, body in steps:
        st.sidebar.markdown(f"**{title}**  \n{body}")
    st.sidebar.divider()

    st.sidebar.subheader("🤖 Models Used")
    st.sidebar.markdown(
        "- **Binary Model** — Cancer vs No Cancer  \n"
        "- **Multi-class Model** — Low / Medium / High severity  \n"
        "- **Regression Model** — Treatment cost (₹)")
    st.sidebar.divider()
 
    st.sidebar.warning(
        "**⚕️ Clinical Disclaimer**  \n"
        "This tool is for decision **support** only. It does **not** replace "
        "professional medical diagnosis. Always confirm results with a "
        "qualified haematologist or oncologist.")

# MODEL LOADING 
@st.cache_resource(show_spinner=False)
def load_models():
    models = {}

    for name in MODEL_PATHS:
        path = MODEL_PATHS[name]
        # Check if file exists
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                models[name] = model
            except Exception as e:
                st.warning(f"Error loading {name} model: {e}")
                models[name] = None
        else:
            st.warning(f"{name} model file not found.")
            models[name] = None

    return models

def engineer_features(gender, wbc, hb, platelets, fever, fatigue, weight_loss):

    if gender == "Male":
        gender_num = 1
    else:
        gender_num = 0

    symptom_count = fever + fatigue + weight_loss

    if wbc > 11000 or wbc < 4000:
        wbc_abnormal = 1
    else:
        wbc_abnormal = 0

    if platelets > 0:
        wbc_platelet_ratio = wbc / platelets
    else:
        wbc_platelet_ratio = 0

    if gender_num == 1:   # Male
        anemia_flag = 1 if hb < 13.5 else 0
    else:                 # Female
        anemia_flag = 1 if hb < 12.0 else 0

    if platelets < 100000:
        thrombocytopenia_flag = 1
    else:
        thrombocytopenia_flag = 0

    wbc_part = 0.40 * (wbc / 130000)
    hb_part = 0.35 * (1 - hb / 20)
    platelets_part = 0.25 * (1 - platelets / 450000)

    clinical_risk_score = wbc_part + hb_part + platelets_part

    # Return all calculated values
    return {
        "symptom_count": symptom_count,
        "wbc_abnormal": wbc_abnormal,
        "wbc_platelet_ratio": wbc_platelet_ratio,
        "anemia_flag": anemia_flag,
        "thrombocytopenia_flag": thrombocytopenia_flag,
        "clinical_risk_score": clinical_risk_score}

# DATAFRAME BUILDER  
def build_input_df(
    age        : int,
    gender     : str,
    wbc        : float,
    rbc        : float,
    hb         : float,
    platelets  : float,
    fever      : str,
    fatigue    : str,
    weight_loss: str,
    city       : str | None = None,
    severity   : str | None = None,
) -> pd.DataFrame:
    fever_int       = 1 if fever       == "Yes" else 0
    fatigue_int     = 1 if fatigue     == "Yes" else 0
    weight_loss_int = 1 if weight_loss == "Yes" else 0

    eng = engineer_features(
        gender, wbc, hb, platelets,
        fever_int, fatigue_int, weight_loss_int,)

    row: dict = {
        "age"        : age,
        "gender"     : gender.lower(),
        "wbc"        : wbc,
        "rbc"        : rbc,
        "hemoglobin" : hb,
        "platelets"  : platelets,
        "fever"      : fever_int,
        "fatigue"    : fatigue_int,
        "weight_loss": weight_loss_int,
        **eng,
    }

    if city is not None:
        row["city"] = city.lower()

    if severity is not None:
        row["severity_level"]  = severity.lower()
        row["severity_target"] = SEVERITY_TARGET.get(severity, 0)

    return pd.DataFrame([row])

# UI HELPERS
def gauge_card_html(label: str, value: float, ref: dict) -> str:
    """Return HTML for one coloured lab-value gauge card."""
    low, high = ref["low"], ref["high"]
    pct = min(max((value - low) / (high - low + 1e-9), 0), 1) * 100
    if value < low:
        css, bar, arrow = "abnormal", "#ff4d6d", "↓ Low"
    elif value > high:
        css, bar, arrow = "abnormal", "#ff4d6d", "↑ High"
    else:
        css, bar, arrow = "normal",   "#00e5a0", "✓ Normal"
    display = f"{value:,.0f}" if value >= 1_000 else f"{value:.1f}"
    return (
        f'<div class="gauge-card">'
        f'  <div class="gauge-label">{label}</div>'
        f'  <div class="gauge-value {css}">{display}</div>'
        f'  <div class="gauge-unit">{ref["unit"]} &nbsp;·&nbsp; {arrow}</div>'
        f'  <div class="gauge-bar-bg">'
        f'    <div class="gauge-bar-fill" style="width:{pct:.0f}%;background:{bar}"></div>'
        f'  </div>'
        f'  <span class="gauge-ref">Ref: {ref["low"]:,} – {ref["high"]:,}</span>'
        f'</div>')


def step_bar_html(current_step: int) -> str:
    """Return HTML for the 3-step progress indicator."""
    steps = ["Patient Input", "Diagnosis", "Cost & Report"]
    items = []
    for i, label in enumerate(steps, start=1):
        if i < current_step:
            css, dot = "done",   "✓"
        elif i == current_step:
            css, dot = "active", str(i)
        else:
            css, dot = "",       str(i)
        items.append(
            f'<div class="step-item {css}">'
            f'  <div class="step-dot">{dot}</div>'
            f'  <div class="step-label">{label}</div>'
            f'</div>')
        
    return '<div class="step-bar">' + "".join(items) + "</div>"

# COST BENCHMARK CHART  — returns PNG bytes, no disk writes
def make_cost_chart(cost: float, theme: str = "Dark") -> bytes:
    bg         = "#0d1526" if theme == "Dark" else ("#ffffff" if theme == "Light" else "#f8f9fa")
    text_color = "#e8eaf2" if theme == "Dark" else "#0d1a3a"
    grid_color = "#1e2d4a" if theme == "Dark" else "#d0d8f0"
    muted      = "#7a8aa8" if theme == "Dark" else "#556080"

    categories = ["Low\n₹50k–1L", "Mid\n₹1L–3L", "High\n₹3L–6L", "Critical\n₹6L+"]
    benchmarks = [75_000, 200_000, 450_000, 700_000]
    bar_colors = ["#00e5a0", "#00c9ff", "#ffb347", "#ff4d6d"]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    bars = ax.bar(categories, benchmarks, color=bar_colors, alpha=0.30, width=0.55, zorder=2)
    for i, bm in enumerate(benchmarks):
        if cost <= bm:
            bars[i].set_alpha(0.60)
            break

    ax.axhline(cost, color="#00c9ff", linewidth=2.5, linestyle="--",
               label=f"Estimate  ₹{cost:,.0f}", zorder=3)
    ax.set_ylabel("Treatment Cost (₹)", color=muted, fontsize=9)
    ax.set_title("Your Estimate vs Benchmark Ranges", color=text_color, fontsize=11, pad=12)
    ax.tick_params(colors=muted, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(grid_color)
    ax.grid(axis="y", color=grid_color, linewidth=0.8, zorder=1)
    ax.legend(fontsize=8, labelcolor=text_color, facecolor=bg, edgecolor=grid_color)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# PDF REPORT  — fully in-memory, returns raw bytes
def generate_pdf(
    patient_name: str,       
    snap        : dict,
    severity    : str,
    cost        : float,
    chart_bytes : bytes,
    cancer_found: bool = True,) -> bytes:
    buf    = io.BytesIO()
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "RT", parent=styles["Title"],
        fontSize=18, textColor=colors.HexColor("#0077ff"), spaceAfter=2)
    sub_style = ParagraphStyle(
        "Sub", parent=styles["Normal"],
        fontSize=8.5, textColor=colors.HexColor("#7a8aa8"), spaceAfter=4)
    
    patient_style = ParagraphStyle(
        "PS", parent=styles["Normal"],
        fontSize=13, textColor=colors.HexColor("#0d1a3a"),
        spaceBefore=6, spaceAfter=8)
    
    heading_style = ParagraphStyle(
        "SH", parent=styles["Heading2"],
        fontSize=11, textColor=colors.HexColor("#0077ff"), spaceBefore=10)

    doc   = SimpleDocTemplate(buf, leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    elems = []

    # Optional logo — place logo.png next to app.py
    logo_path = os.path.join(BASE_DIR, "logo.png")
    if os.path.exists(logo_path):
        elems += [RLImage(logo_path, width=3*cm, height=1.5*cm), Spacer(1, 6)]

    # Report header
    elems += [
        Paragraph("Blood Cancer AI — Diagnostic Report", title_style),
        Paragraph("Confidential  ·  Generated by Blood Cancer AI System", sub_style),]

    # Patient name — prominent, right after the header (FIX 2)
    display_name = patient_name.strip() if patient_name.strip() else "N/A"
    elems += [Paragraph(f"<b>Patient:</b> {display_name}", patient_style), Spacer(1, 6)]

    # Patient data table
    elems.append(Paragraph("Patient Information & Lab Values", heading_style))
    elems.append(Spacer(1, 5))
    table_data = [["Field", "Value"]] + [[str(k), str(v)] for k, v in snap.items()]
    tbl = Table(table_data, colWidths=[6*cm, 9*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  colors.HexColor("#0077ff")),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f4ff"),
                                              colors.HexColor("#ffffff")]),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d8f0")),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 8),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
    ]))
    elems += [tbl, Spacer(1, 12)]

    # Diagnosis summary
    elems.append(Paragraph("Diagnosis Summary", heading_style))
    elems.append(Spacer(1, 5))

    if cancer_found:
        sev_color = {"Low": "#00b37d", "Medium": "#ff8c00", "High": "#e30022"}.get(severity, "#333")
        diag_html = (
            f"<b>Cancer Detected:</b> Yes<br/>"
            f"<b>Severity Level:</b> <font color='{sev_color}'>{severity}</font><br/>"
            f"<b>Estimated Treatment Cost:</b> ₹{cost:,.0f}<br/><br/>"
            "<i>Disclaimer: This report is AI-generated and is intended to assist "
            "clinicians, not replace professional medical judgement.</i>")
    else:
        # clean negative-result block for "no cancer" PDF
        diag_html = (
            "<b>Cancer Detected:</b> No<br/>"
            "All haematological indicators appear within clinically acceptable ranges.<br/>"
            "<b>Recommendation:</b> Routine follow-up in 6–12 months.<br/><br/>"
            "<i>Disclaimer: This report is AI-generated and is intended to assist "
            "clinicians, not replace professional medical judgement.</i>")

    elems.append(Paragraph(diag_html, styles["Normal"]))
    elems.append(Spacer(1, 12))

    # Benchmark chart (only appended when bytes are provided)
    if chart_bytes:
        elems.append(Paragraph("Cost Benchmark Chart", heading_style))
        elems.append(Spacer(1, 5))
        elems.append(RLImage(io.BytesIO(chart_bytes), width=13*cm, height=7*cm))

    doc.build(elems)
    return buf.getvalue()

# STATE RESET HELPER
def _reset_state() -> None:
    for key in ["predicted", "cancer_detected", "cost_estimated"]:
        st.session_state[key] = False
    st.session_state["severity_label"] = ""
    st.session_state["estimated_cost"] = 0.0
    st.session_state["risk_score"]     = 0.0
    st.session_state["eng_features"]   = {}
    st.session_state["snap"]           = {}

# MAIN APPLICATION
def main() -> None:
    theme = st.session_state["theme"]
    apply_theme(theme)
    render_sidebar()          # sidebar always rendered first
    models = load_models()

    # Progress step
    if not st.session_state["predicted"]:
        current_step = 1
    elif st.session_state["cost_estimated"]:
        current_step = 3
    else:
        current_step = 2

    # Hero 
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-badge"> <span style="color:white;">&#9877;</span> AI-Powered Clinical Diagnostics</div>
            <h1>Blood Cancer <span>AI</span> System</h1>
            <p>Enter patient haematology results and symptoms for an instant
               AI-powered diagnosis, severity classification, and treatment cost estimate.</p>
            {step_bar_html(current_step)}
        </div>
        """,
        unsafe_allow_html=True)

    # Missing model files warning
    missing = [k for k, m in models.items() if m is None]
    if missing:
        st.markdown(
            f'<div class="alert-box alert-warn">⚠️ Missing model files: '
            f'<b>{", ".join(missing)}</b>. '
            f'Place the .pkl files in the <code>models/</code> folder.</div>',
            unsafe_allow_html=True)

    # SECTION 1 — Patient Clinical Data
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧑‍⚕️ Patient Clinical Data</div>', unsafe_allow_html=True)

    # Patient name — full-width, first field
    patient_name = st.text_input(
        "Patient Name", value=st.session_state["patient_name"],
        placeholder="e.g. Vijay Kumar",
        help="Appears in the results and the downloadable PDF report")
    
    st.session_state["patient_name"] = patient_name

    if patient_name.strip():
        st.markdown(
            f'<div class="patient-banner">'
            f'👤 Patient: <strong>{patient_name.strip()}</strong>'
            f'</div>',
            unsafe_allow_html=True)

    # Three-column input grid
    col1, col2, col3 = st.columns(3)
    with col1:
        age    = st.number_input("Age (years)",      min_value=1,        max_value=100,     value=35,      step=1)
        gender = st.selectbox("Gender",              ["Male", "Female"])
        wbc    = st.number_input("WBC (cells/uL)",   min_value=2_000,    max_value=130_000, value=7_000,   step=500)
    with col2:
        rbc       = st.number_input("RBC (M/uL)",           min_value=1.5,   max_value=7.0,     value=4.5,     step=0.1, format="%.1f")
        hb        = st.number_input("Haemoglobin (g/dL)",   min_value=3.0,   max_value=20.0,    value=13.5,    step=0.1, format="%.1f")
        platelets = st.number_input("Platelets (cells/uL)", min_value=5_000, max_value=450_000, value=250_000, step=1_000)
    with col3:
        fever       = st.selectbox("Fever",       ["No", "Yes"])
        fatigue     = st.selectbox("Fatigue",     ["No", "Yes"])
        weight_loss = st.selectbox("Weight Loss", ["No", "Yes"])

    st.markdown('</div>', unsafe_allow_html=True)

    # Live gauge cards
    st.markdown(
        '<div class="gauge-grid">'
        + gauge_card_html("WBC",         wbc,       LAB_REFS["WBC"])
        + gauge_card_html("Haemoglobin", hb,        LAB_REFS["Haemoglobin"])
        + gauge_card_html("Platelets",   platelets, LAB_REFS["Platelets"])
        + gauge_card_html("RBC",         rbc,       LAB_REFS["RBC"])
        + '</div>',
        unsafe_allow_html=True)

    # Predict button
    _, btn_col, _ = st.columns([2, 2, 2])
    with btn_col:
        predict_clicked = st.button(
            "🔬 Run Diagnostic Prediction",
            disabled=models["binary"] is None)

    # SECTION 2 — Run binary + multiclass inference
    if predict_clicked:
        with st.spinner("Running diagnostic models…"):
            df_base     = build_input_df(age, gender, wbc, rbc, hb, platelets,
                                         fever, fatigue, weight_loss)
            cancer_pred = int(models["binary"].predict(df_base)[0])

            severity_label = ""
            if cancer_pred == 1 and models["multiclass"] is not None:
                severity_code  = int(models["multiclass"].predict(df_base)[0])
                severity_label = SEVERITY_LABELS.get(severity_code, "Unknown")

            eng = engineer_features(
                gender, wbc, hb, platelets,
                1 if fever       == "Yes" else 0,
                1 if fatigue     == "Yes" else 0,
                1 if weight_loss == "Yes" else 0,)

            st.session_state.update({
                "predicted"      : True,
                "cancer_detected": cancer_pred == 1,
                "severity_label" : severity_label,
                "cost_estimated" : False,
                "estimated_cost" : 0.0,
                "risk_score"     : eng["clinical_risk_score"],
                "eng_features"   : eng,
                "snap"           : {
                    "Age"        : age,
                    "Gender"     : gender,
                    "WBC"        : f"{wbc:,}",
                    "RBC"        : rbc,
                    "Haemoglobin": hb,
                    "Platelets"  : f"{platelets:,}",
                    "Fever"      : fever,
                    "Fatigue"    : fatigue,
                    "Weight Loss": weight_loss,
                },})
        st.rerun()

    if not st.session_state["predicted"]:
        return

    st.markdown("<hr>", unsafe_allow_html=True)

    # SECTION 3 — Diagnosis result
    name_display = st.session_state["patient_name"].strip() or "the patient"

    # No cancer
    if not st.session_state["cancer_detected"]:
        st.markdown(
            f"""
            <div class="result-banner result-ok">
                <div class="result-icon">✅</div>
                <div>
                    <p class="result-title">No Cancer Detected — {name_display}</p>
                    <p class="result-sub">All haematological indicators appear within clinically
                       acceptable ranges. Routine follow-up recommended in 6–12 months.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True)

        pdf_bytes = generate_pdf(
            patient_name = st.session_state["patient_name"],
            snap         = st.session_state["snap"],
            severity     = "",
            cost         = 0.0,
            chart_bytes  = b"",
            cancer_found = False)
        
        _, dl_col, ra_col = st.columns([1, 2, 2])
        with dl_col:
            file_name = f"{patient_name}_blood_cancer_report.pdf"
            st.download_button(
                label     = "⬇️ Download Report (PDF)",
                data      = pdf_bytes,
                file_name = file_name,
                mime      = "application/pdf")
            
        with ra_col:
            if st.button("↩ New Patient"):
                _reset_state()   
                st.rerun()
        return

    # Cancer detected 
    severity   = st.session_state["severity_label"]
    risk_score = st.session_state["risk_score"]
    eng        = st.session_state["eng_features"]
    sev_css    = {"Low": "sev-low", "Medium": "sev-medium", "High": "sev-high"}.get(severity, "sev-low")

    st.markdown(
        f"""
        <div class="result-banner result-bad">
            <div class="result-icon">⚠️</div>
            <div>
                <p class="result-title">Cancer Indicators Detected — {name_display}</p>
                <p class="result-sub">Haematological markers suggest possible malignancy.
                   Immediate clinical review and oncology referral recommended.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Severity Level", severity)
    m2.metric("Clinical Risk",  f"{risk_score:.3f}")
    m3.metric("Symptom Count",  eng.get("symptom_count", 0))
    m4.metric("Anemia Flag",    "Yes" if eng.get("anemia_flag") else "No")

    st.markdown(
        f'<p style="margin:.75rem 0 1.5rem;">Severity: '
        f'<span class="severity-pill {sev_css}">{severity}</span></p>',
        unsafe_allow_html=True)

    with st.expander("📊 Engineered Clinical Features", expanded=False):
        feat_df = pd.DataFrame([{
            "Feature": k.replace("_", " ").title(),
            "Value"  : f"{v:.4f}" if isinstance(v, float) else str(v),
        } for k, v in eng.items()])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # SECTION 4 - Treatment Cost Estimation
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">💰 Treatment Cost Estimation</div>',
        unsafe_allow_html=True)

    if models["regression"] is None:
        st.markdown(
            '<div class="alert-box alert-warn">Regression model not found — '
            'cost estimation is unavailable.</div>',
            unsafe_allow_html=True)
    else:
        city_idx = CITIES.index(st.session_state["city"]) \
                   if st.session_state["city"] in CITIES else 0
        city = st.selectbox("Select City / Hospital Location", CITIES, index=city_idx)
        st.session_state["city"] = city

        _, cbtn_col, _ = st.columns([2, 2, 2])
        with cbtn_col:
            estimate_clicked = st.button("💳 Estimate Treatment Cost")

        if estimate_clicked:
            with st.spinner("Running cost estimation model…"):
                df_reg = build_input_df(
                    age, gender, wbc, rbc, hb, platelets,
                    fever, fatigue, weight_loss,
                    city=city, severity=severity)
                
                raw_cost = float(models["regression"].predict(df_reg)[0])
                st.session_state["estimated_cost"] = max(0.0, raw_cost)
                st.session_state["cost_estimated"]  = True
            st.rerun()

        if st.session_state["cost_estimated"]:
            cost = st.session_state["estimated_cost"]

            st.markdown(
                f'<div class="cost-display">₹ {cost:,.0f}</div>'
                f'<p class="cost-note">Estimated total treatment cost in {city} '
                f'for <b>{name_display}</b> — {severity.lower()}-severity case</p>',
                unsafe_allow_html=True)

            chart_bytes = make_cost_chart(cost, theme)
            st.image(chart_bytes, use_column_width=True)

            if cost < 100_000:
                band_msg = "🟢 Low cost range — standard outpatient/day-care protocols likely."
            elif cost < 300_000:
                band_msg = "🟡 Mid cost range — hospitalisation and combination therapy expected."
            elif cost < 600_000:
                band_msg = "🟠 High cost range — intensive inpatient treatment or transplant may be involved."
            else:
                band_msg = "🔴 Critical cost range — complex or advanced-stage treatment anticipated."
            st.info(band_msg)

            snap_full = {
                **st.session_state["snap"],
                "City"          : city,
                "Diagnosis"     : "Cancer Detected",
                "Severity"      : severity,
                "Risk Score"    : f"{risk_score:.3f}",
                "Estimated Cost": f"₹{cost:,.0f}"}

            pdf_bytes = generate_pdf(
                patient_name = st.session_state["patient_name"],
                snap         = snap_full,
                severity     = severity,
                cost         = cost,
                chart_bytes  = chart_bytes,
                cancer_found = True)

            st.markdown("<br>", unsafe_allow_html=True)
            _, dl_col, _ = st.columns([1, 3, 1])
            with dl_col:
                file_name = f"{patient_name}_blood_cancer_report.pdf"
                st.download_button(
                    label     = "⬇️ Download Full Medical Report (PDF)",
                    data      = pdf_bytes,
                    file_name = file_name,
                    mime      = "application/pdf")

    st.markdown('</div>', unsafe_allow_html=True)

    # Reset button
    st.markdown("<br>", unsafe_allow_html=True)
    _, rst_col, _ = st.columns([2, 2, 2])
    with rst_col:
        if st.button("↩ Start New Consultation"):
            _reset_state()  
            st.rerun()

    # Footer
    st.markdown(
        """
        <hr>
        <div class="footer">
            Blood Cancer AI System &nbsp;·&nbsp;
            For clinical decision support only &nbsp;·&nbsp;
            Not a substitute for professional medical advice
        </div>
        """,
        unsafe_allow_html=True)

# ENTRY POINT
if __name__ == "__main__":
    main()