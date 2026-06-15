import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdfplumber, re, joblib, os, io
from datetime import date, datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Risk Stratification",
    page_icon="🏥",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ════════════════════════════════════════════════════════════
   HOSPITAL-THEMED BACKGROUND
   ════════════════════════════════════════════════════════════ */

/* Main background: soft medical blue + subtle grid cross pattern */
.stApp {
    background-color: #e8f4fd;
    background-image:
        radial-gradient(ellipse at 8%  8%,  rgba(0,119,182,0.18) 0%, transparent 50%),
        radial-gradient(ellipse at 92% 92%, rgba(0,180,216,0.14) 0%, transparent 50%),
        radial-gradient(ellipse at 85% 10%, rgba(2,62,138,0.08)  0%, transparent 40%),
        repeating-linear-gradient(
            0deg,  transparent, transparent 28px,
            rgba(0,119,182,0.055) 28px, rgba(0,119,182,0.055) 32px),
        repeating-linear-gradient(
            90deg, transparent, transparent 28px,
            rgba(0,119,182,0.055) 28px, rgba(0,119,182,0.055) 32px);
}

/* Sidebar: deep navy-to-cyan medical gradient */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #023e8a 0%, #0077b6 65%, #00b4d8 100%) !important;
    border-right: 3px solid #90e0ef;
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] img { animation: hb-pulse 2s ease-in-out infinite; }
@keyframes hb-pulse {
    0%, 100% { transform: scale(1);     opacity: 1;    }
    50%       { transform: scale(1.08); opacity: 0.85; }
}

/* Main content: frosted-glass card */
.block-container {
    background: rgba(255,255,255,0.55) !important;
    backdrop-filter: blur(4px);
    border-radius: 16px;
    border: 1px solid rgba(0,119,182,0.12);
    box-shadow: 0 4px 24px rgba(0,119,182,0.08);
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

/* Tabs: pill style */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.70);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 6px 18px;
    font-weight: 600;
    color: #023e8a !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #023e8a, #0077b6) !important;
    color: #ffffff !important;
}

/* Metric cards: frosted glass */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.82) !important;
    backdrop-filter: blur(6px);
    border: 1px solid rgba(0,119,182,0.18) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 10px rgba(0,119,182,0.10);
}
[data-testid="stMetricLabel"] { color: #023e8a !important; font-weight: 600; }
[data-testid="stMetricValue"] { color: #0077b6 !important; }

/* Buttons: navy gradient */
.stButton > button {
    background: linear-gradient(90deg, #023e8a, #0077b6) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 3px 10px rgba(0,119,182,0.28);
    transition: transform 0.15s, box-shadow 0.15s;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(0,119,182,0.42) !important;
}

/* Headings */
h1, h2 { color: #023e8a !important; }
h3      { color: #0077b6 !important; }

/* Divider */
hr { border-color: rgba(0,119,182,0.22) !important; }

/* DataFrames */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.88) !important;
    border-radius: 10px;
    border: 1px solid rgba(0,119,182,0.14);
}

/* Inputs / selects */
[data-testid="stNumberInput"] > div,
[data-testid="stSelectbox"]   > div {
    background: rgba(255,255,255,0.80) !important;
    border-radius: 8px;
}

/* ════════════════════════════════════════════════════════════
   ORIGINAL COMPONENT STYLES (unchanged)
   ════════════════════════════════════════════════════════════ */
.kpi{background:white;border:1px solid #e8eaf0;border-radius:12px;
     padding:.9rem 1.1rem;text-align:center;margin-bottom:.5rem}
.kpi-num{font-size:2rem;font-weight:700;margin:4px 0}
.kpi-lbl{font-size:.72rem;color:#999;text-transform:uppercase;letter-spacing:.06em}
.sec{font-size:.72rem;font-weight:600;text-transform:uppercase;
     letter-spacing:.08em;color:#aaa;border-bottom:1px solid #eee;
     padding-bottom:4px;margin-bottom:10px;margin-top:1rem}
.rc{border-radius:10px;padding:.8rem 1rem;margin-bottom:.45rem;
    border-left:4px solid;font-size:.88rem}
.rcR{background:#fff5f5;border-color:#E24B4A;color:#4a1010}
.rcA{background:#fffbf0;border-color:#BA7517;color:#4a2e00}
.rcG{background:#f4fbec;border-color:#639922;color:#1a3300}
.rcB{background:#f0f7ff;border-color:#378ADD;color:#0a2a50}
.badge{display:inline-block;padding:2px 9px;border-radius:99px;
       font-size:.72rem;font-weight:600}
.red{background:#FCEBEB;color:#A32D2D}
.grn{background:#EAF3DE;color:#3B6D11}
.amb{background:#FAEEDA;color:#854F0B}
.blu{background:#E8F2FC;color:#1a5fa8}
.lab-row{display:flex;justify-content:space-between;align-items:center;
         padding:8px 12px;border-radius:8px;margin-bottom:5px;background:white;
         border:1px solid #e8eaf0}
.lab-name{font-weight:600;font-size:.88rem}
.lab-val{font-size:1rem;font-weight:700}
.lab-range{font-size:.75rem;color:#999}
</style>
""", unsafe_allow_html=True)

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Risk_model1.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

model = load_model()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_risk(age, cost, abn):
    if model:
        X = pd.DataFrame([[age, cost, abn]],
                         columns=["Age", "TreatmentCost", "AbnormalLabCount"])
        return float(model.predict_proba(X)[0][1])
    return sigmoid(0.12726 * age + 0.000179 * cost + 0.32716 * abn - 10.8954)

def risk_label(p):
    if p < 0.30: return "Low",      "#3B6D11", "🟢"
    if p < 0.60: return "Moderate", "#BA7517", "🟡"
    return              "High",     "#A32D2D", "🔴"

# ── Constants ──────────────────────────────────────────────────────────────────
DIAGNOSES = [
    "Hypertension", "Diabetes", "Heart Disease", "Asthma", "Stroke",
    "COPD", "Cancer", "Arthritis", "Kidney Disease", "Liver Disease",
]

LAB_RANGES = {
    "M": {
        "Blood Pressure": (90, 140), "Blood Sugar":   (70, 140),
        "Cholesterol":    (0,  200), "Creatinine":    (0.6, 1.2),
        "Hemoglobin":     (13, 17),  "Vitamin D":     (20, 50),
    },
    "F": {
        "Blood Pressure": (90, 140), "Blood Sugar":   (70, 140),
        "Cholesterol":    (0,  200), "Creatinine":    (0.6, 1.2),
        "Hemoglobin":     (12, 15.5),"Vitamin D":     (20, 50),
    },
}

LAB_UNITS = {
    "Blood Pressure": "mmHg", "Blood Sugar": "mg/dL", "Cholesterol": "mg/dL",
    "Creatinine": "mg/dL",    "Hemoglobin":  "g/dL",  "Vitamin D":   "ng/mL",
}

LAB_ICONS = {
    "Blood Pressure": "🩺", "Blood Sugar": "🍬", "Cholesterol": "❤️",
    "Creatinine":     "🫘", "Hemoglobin":  "🩸", "Vitamin D":   "☀️",
}

MEDICATIONS = {
    "Hypertension":   ["Amlodipine 5 mg OD", "Losartan 50 mg OD", "Hydrochlorothiazide 12.5 mg OD"],
    "Diabetes":       ["Metformin 500 mg BD", "Glipizide 5 mg OD", "Insulin Glargine (if HbA1c > 9%)"],
    "Heart Disease":  ["Aspirin 75 mg OD", "Atorvastatin 40 mg HS", "Carvedilol 6.25 mg BD"],
    "Asthma":         ["Salbutamol inhaler PRN", "Budesonide inhaler BD", "Montelukast 10 mg HS"],
    "Stroke":         ["Clopidogrel 75 mg OD", "Atorvastatin 80 mg HS", "Ramipril 5 mg OD"],
    "COPD":           ["Tiotropium inhaler OD", "Salmeterol inhaler BD", "Prednisolone 30 mg (acute)"],
    "Cancer":         ["Ondansetron 8 mg TDS", "Dexamethasone 4 mg BD", "Morphine SR (pain)"],
    "Arthritis":      ["Naproxen 500 mg BD (with food)", "Methotrexate 7.5 mg weekly", "Folic acid 5 mg weekly"],
    "Kidney Disease": ["Erythropoietin (if Hb<10)", "Calcium carbonate 500 mg TDS", "Furosemide 40 mg OD"],
    "Liver Disease":  ["Lactulose 30 ml TDS", "Spironolactone 50 mg OD", "Propranolol 20 mg BD"],
}

LAB_RECS = {
    "Blood Pressure": {
        "high": ("🩺 Hypertension",    "Antihypertensive review, low-salt diet, BP monitoring every 4 hrs.", "R"),
        "low":  ("🩺 Hypotension",     "IV fluid bolus, sepsis screen, orthostatic BP check.", "A"),
    },
    "Blood Sugar": {
        "high": ("🍬 Hyperglycemia",   "Insulin sliding scale, dietitian consult, glucose check every 6 hrs.", "R"),
        "low":  ("🍬 Hypoglycemia",    "15g oral glucose immediately, recheck in 15 min, review medications.", "R"),
    },
    "Cholesterol": {
        "high": ("❤️ Dyslipidemia",    "Start/intensify statin therapy, low-fat diet counselling, daily exercise.", "A"),
    },
    "Creatinine": {
        "high": ("🫘 Renal impairment","Nephrology consult, adjust renally-cleared meds, monitor GFR & urine output.", "R"),
        "low":  ("🫘 Low creatinine",  "Nutritional assessment, consider muscle-wasting workup.", "A"),
    },
    "Hemoglobin": {
        "low":  ("🩸 Anemia",          "Iron studies, B12/folate; consider IV iron or EPO if renal cause.", "A"),
        "high": ("🩸 Polycythemia",    "Hematology referral, assess for dehydration, phlebotomy if needed.", "A"),
    },
    "Vitamin D": {
        "low":  ("☀️ Vitamin D deficiency", "Cholecalciferol 60,000 IU weekly × 8 wks, then maintenance.", "A"),
        "high": ("☀️ Vitamin D toxicity",   "Stop supplement, check serum calcium, increase fluids.", "R"),
    },
}

DIAG_OUTCOME = {
    "Hypertension":  {"Complicated": 11, "Deceased": 12, "Recovered": 5},
    "Diabetes":      {"Complicated": 9,  "Deceased": 3,  "Recovered": 5},
    "Heart Disease": {"Complicated": 9,  "Deceased": 6,  "Recovered": 9},
    "Asthma":        {"Complicated": 6,  "Deceased": 3,  "Recovered": 8},
    "Stroke":        {"Complicated": 7,  "Deceased": 9,  "Recovered": 7},
    "COPD":          {"Complicated": 6,  "Deceased": 10, "Recovered": 4},
    "Cancer":        {"Complicated": 5,  "Deceased": 8,  "Recovered": 5},
    "Arthritis":     {"Complicated": 5,  "Deceased": 8,  "Recovered": 4},
    "Kidney Disease":{"Complicated": 3,  "Deceased": 4,  "Recovered": 3},
    "Liver Disease": {"Complicated": 7,  "Deceased": 6,  "Recovered": 13},
}

AVG_COST = {
    "Arthritis": 6679, "Kidney Disease": 6329, "Stroke": 6267,
    "Heart Disease": 6237, "Cancer": 6234, "Diabetes": 6153,
    "Asthma": 6091, "Liver Disease": 5976, "Hypertension": 5854, "COPD": 5533,
}

# ── PDF lab extraction ─────────────────────────────────────────────────────────
LAB_PATTERNS = {
    "Blood Pressure": [
        r"blood\s*pressure[\s:=\-]*(\d{2,3})(?:\s*/\s*\d{2,3})?",
        r"\bbp[\s:=\-]*(\d{2,3})(?:\s*/\s*\d{2,3})?",
        r"systolic[\s:=\-]*(\d{2,3})",
    ],
    "Blood Sugar": [
        r"blood\s*sugar[\s:=\-]*(\d{2,3}(?:\.\d+)?)",
        r"glucose[\s:=\-]*(\d{2,3}(?:\.\d+)?)",
        r"\bfbs[\s:=\-]*(\d{2,3}(?:\.\d+)?)",
        r"\brbs[\s:=\-]*(\d{2,3}(?:\.\d+)?)",
    ],
    "Cholesterol": [
        r"total\s*cholesterol[\s:=\-]*(\d{2,3}(?:\.\d+)?)",
        r"cholesterol[\s:=\-]*(\d{2,3}(?:\.\d+)?)",
    ],
    "Creatinine": [
        r"(?:s\.?\s*|serum\s*)?creatinine[\s:=\-]*(\d{1,2}(?:\.\d+)?)",
    ],
    "Hemoglobin": [
        r"h(?:a?e)?moglobin[\s:=\-]*(\d{1,2}(?:\.\d+)?)",
        r"\bhb[\s:=\-]*(\d{1,2}(?:\.\d+)?)",
        r"\bhgb[\s:=\-]*(\d{1,2}(?:\.\d+)?)",
    ],
    "Vitamin D": [
        r"vitamin\s*d[\s:=\-]*(\d{1,3}(?:\.\d+)?)",
        r"25[\s\-]?oh[\s\-]?d[\s:=\-]*(\d{1,3}(?:\.\d+)?)",
        r"vit\.?\s*d[\s:=\-]*(\d{1,3}(?:\.\d+)?)",
    ],
}

VALID_RANGES = {
    "Blood Pressure": (60, 250), "Blood Sugar":  (40, 600),
    "Cholesterol":    (50, 500), "Creatinine":   (0.1, 15),
    "Hemoglobin":     (3,  25),  "Vitamin D":    (1, 150),
}

def extract_pdf_text(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
            for table in (page.extract_tables() or []):
                for row in table:
                    if row:
                        text += " | ".join(str(c) if c else "" for c in row) + "\n"
    return text

def parse_labs(text):
    text_low = text.lower()
    found, context = {}, {}
    for lab, patterns in LAB_PATTERNS.items():
        lo, hi = VALID_RANGES[lab]
        for pat in patterns:
            for match in re.finditer(pat, text_low):
                try:
                    val = float(match.group(1))
                    if lo <= val <= hi:
                        found[lab] = round(val, 2)
                        start = max(0, match.start() - 30)
                        end   = min(len(text_low), match.end() + 30)
                        context[lab] = text_low[start:end].strip()
                        break
                except (ValueError, IndexError):
                    continue
            if lab in found:
                break
    return found, context

def check_labs(vals, gender="M"):
    r = LAB_RANGES[gender[0].upper()]
    abn, norm = [], []
    for lab, val in vals.items():
        lo, hi = r[lab]
        if val < lo:   abn.append((lab, val, lo, hi, "low"))
        elif val > hi: abn.append((lab, val, lo, hi, "high"))
        else:          norm.append((lab, val, lo, hi))
    return abn, norm

# ── Session state ──────────────────────────────────────────────────────────────
if "registry"     not in st.session_state: st.session_state.registry     = []
if "risk_history" not in st.session_state: st.session_state.risk_history = []

# ── Helpers ────────────────────────────────────────────────────────────────────
def gauge_svg(pct, color):
    dash = round(2.827 * min(pct, 100))
    return f"""<div style="text-align:center;padding:.4rem 0">
<svg width="200" height="110" viewBox="0 0 200 110">
  <path d="M15,105 A85,85 0 0,1 185,105" fill="none" stroke="#eee"
        stroke-width="16" stroke-linecap="round"/>
  <path d="M15,105 A85,85 0 0,1 185,105" fill="none" stroke="{color}"
        stroke-width="16" stroke-linecap="round"
        stroke-dasharray="{dash},267"/>
  <text x="100" y="98"  text-anchor="middle" font-size="26"
        font-weight="700" fill="{color}">{pct:.0f}%</text>
  <text x="100" y="112" text-anchor="middle" font-size="10"
        fill="#aaa">readmission probability</text>
</svg></div>"""

def waterfall_chart(age, cost, abn_cnt, prob):
    items = [
        ("Age", 0.12726 * age),
        ("Treatment Cost", 0.000179 * cost),
        ("Abnormal Labs", 0.32716 * abn_cnt),
        ("Base Risk", -10.8954)
    ]
    vals = [v for _, v in items]
    labels = [l for l, _ in items] + ["Risk Score"]
    bottoms = []
    running = 0
    for v in vals:
        bottoms.append(min(running, running + v))
        running += v
    colors = ["#4E79A7" if v >= 0 else "#E15759" for v in vals]
    final_color = ("#D62728" if prob >= 0.60 else "#F28E2B" if prob >= 0.30 else "#59A14F")
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (b, v, c) in enumerate(zip(bottoms, vals, colors)):
        ax.bar(i, abs(v), bottom=b, color=c, width=0.7, edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(i, b + abs(v) + 0.15, f"{v:+.2f}", ha="center", fontsize=10, fontweight="bold", color=c)
    final_score = sum(vals)
    ax.bar(len(vals), abs(final_score), bottom=min(final_score, 0), color=final_color,
           width=0.7, edgecolor="white", linewidth=1.5, zorder=3)
    ax.text(len(vals), abs(final_score) + min(final_score, 0) + 0.3, f"{final_score:.2f}",
            ha="center", fontsize=11, fontweight="bold", color=final_color)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=11, rotation=15)
    ax.axhline(y=0, color="#BDBDBD", linewidth=1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylabel("Log-Odds Contribution", fontsize=11, fontweight="bold")
    ax.set_title("Risk Score Breakdown", fontsize=16, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_facecolor("#FAFAFA"); fig.patch.set_facecolor("white")
    ax.margins(x=0.12); plt.tight_layout()
    return fig

def risk_trend_chart(history):
    if len(history) < 2:
        return None
    risks = [h["risk_pct"] for h in history]
    dates = [h["datetime"].strftime("%d %b\n%H:%M") for h in history]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.fill_between(range(len(risks)), risks, alpha=0.12, color="#E24B4A")
    ax.plot(range(len(risks)), risks, color="#E24B4A", linewidth=2.2,
            marker="o", markersize=7, zorder=5)
    for i, r in enumerate(risks):
        ax.annotate(f"{r:.0f}%", (i, r), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, fontweight="700", color="#A32D2D")
    ax.axhspan(0,  30,  alpha=0.05, color="#3B6D11")
    ax.axhspan(30, 60,  alpha=0.05, color="#BA7517")
    ax.axhspan(60, 100, alpha=0.05, color="#E24B4A")
    ax.set_xticks(range(len(dates))); ax.set_xticklabels(dates, fontsize=7)
    ax.set_ylabel("Risk %", fontsize=8); ax.set_ylim(0, 100)
    ax.set_title("Patient risk trend over time", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#fafafa"); fig.patch.set_facecolor("#fafafa")
    plt.tight_layout()
    return fig

def population_percentile(age, cost, abn):
    np.random.seed(42)
    pop = [predict_risk(a, c, n) for a, c, n in zip(
        np.random.randint(30, 90, 200),
        np.random.randint(2054, 9973, 200),
        np.random.randint(0, 7, 200),
    )]
    pat    = predict_risk(age, cost, abn)
    pctile = round(sum(1 for r in pop if r < pat) / 200 * 100)
    return pctile

# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🏥 Healthcare Risk Stratification System")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔬 Prediction",
    "📄 Lab Report Analyzer",
    "👥 Patient Registry",
    "📊 Dashboard",
    "📁 Batch Prediction",
    "🔮 What-If Simulator",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    L, R = st.columns([1.1, 1], gap="large")

    with L:
        st.markdown('<div class="sec">Patient information</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        age    = c1.number_input("Age",             1,   120,  58)
        gender = c2.selectbox("Gender",             ["Male", "Female"])
        diag   = c3.selectbox("Diagnosis",          DIAGNOSES)

        c4, c5 = st.columns(2)
        cost = c4.number_input("Treatment cost (₹)", 0, 500000, 6500, step=500)
        los  = c5.number_input("Length of stay (days)", 1, 365, 5)

        st.markdown('<div class="sec">Lab values</div>', unsafe_allow_html=True)
        c6,  c7  = st.columns(2)
        bp   = c6.number_input("Blood Pressure (mmHg)", 60,  250,  145)
        bs   = c7.number_input("Blood Sugar (mg/dL)",   40,  600,  165)
        c8,  c9  = st.columns(2)
        chol = c8.number_input("Cholesterol (mg/dL)",   50,  500,  215)
        cr   = c9.number_input("Creatinine (mg/dL)",    0.1, 15.0, 1.5, step=0.1)
        c10, c11 = st.columns(2)
        hgb  = c10.number_input("Hemoglobin (g/dL)",   3.0, 25.0, 11.0, step=0.1)
        vitd = c11.number_input("Vitamin D (ng/mL)",    1,   150,  14)

        save_cb  = st.checkbox("💾 Save to patient registry after prediction")
        pat_name = st.text_input("Patient name", "Patient") if save_cb else ""
        run      = st.button("🧠 Calculate risk score", use_container_width=True, type="primary")

    with R:
        if run:
            lab_vals = {
                "Blood Pressure": bp,   "Blood Sugar": bs,
                "Cholesterol":    chol, "Creatinine":  cr,
                "Hemoglobin":     hgb,  "Vitamin D":   vitd,
            }
            abn, norm   = check_labs(lab_vals, gender)
            abn_cnt     = len(abn)
            prob        = predict_risk(age, cost, abn_cnt)
            pct         = prob * 100
            lvl, col, ico = risk_label(prob)

            st.markdown(gauge_svg(pct, col), unsafe_allow_html=True)
            st.markdown(
                f"<h3 style='text-align:center;color:{col};margin-top:-6px'>"
                f"{ico} {lvl} Risk</h3>", unsafe_allow_html=True)

            k1, k2, k3 = st.columns(3)
            k1.metric("Abnormal labs", f"{abn_cnt}/6")
            k2.metric("Age",           f"{age} yrs")
            k3.metric("Cost",          f"₹{cost:,}")

            pctile = population_percentile(age, cost, abn_cnt)
            st.markdown(
                f'<div class="rc rcA" style="margin-top:.5rem">📊 '
                f'<b>Population rank:</b> Top <b>{100 - pctile}%</b> '
                f'highest-risk in reference dataset.</div>',
                unsafe_allow_html=True)

            st.pyplot(waterfall_chart(age, cost, abn_cnt, prob))
            plt.close()

            st.markdown('<div class="sec">Lab results & recommendations</div>', unsafe_allow_html=True)
            for lab, val, lo, hi, direction in abn:
                rec = LAB_RECS.get(lab, {}).get(direction)
                css  = "rcR" if rec and rec[2] == "R" else "rcA"
                body = (f"<b>{rec[0]}</b><br><span style='font-size:.82rem'>{rec[1]}</span>"
                        if rec else f"<b>{lab}</b>: {val} (range {lo}–{hi})")
                st.markdown(f'<div class="rc {css}">⚠️ {body}</div>', unsafe_allow_html=True)
            for lab, val, lo, hi in norm:
                st.markdown(
                    f'<div class="rc rcG">✓ <b>{lab}</b>: {val} '
                    f'<span class="badge grn">Normal</span></div>',
                    unsafe_allow_html=True)

            st.markdown(
                f'<div class="sec">Suggested medications — {diag}</div>',
                unsafe_allow_html=True)
            for med in MEDICATIONS.get(diag, []):
                st.markdown(f'<div class="rc rcA">💊 {med}</div>', unsafe_allow_html=True)
            st.caption("⚕️ Reference only. Always verify with prescribing physician.")

            st.divider()
            if prob >= 0.6:
                st.error("**High risk** — Intensive monitoring and discharge planning required.", icon="🚨")
            elif prob >= 0.3:
                st.warning("**Moderate risk** — Follow-up within 7 days.", icon="⚠️")
            else:
                st.success("**Low risk** — Standard discharge protocol.", icon="✅")

            if save_cb and pat_name:
                rec_entry = {
                    "id": len(st.session_state.registry) + 1,
                    "name": pat_name, "age": age, "gender": gender,
                    "diagnosis": diag, "cost": cost, "los": los,
                    "bp": bp, "bs": bs, "chol": chol, "cr": cr,
                    "hgb": hgb, "vitd": vitd, "abn_cnt": abn_cnt,
                    "risk_pct": round(pct, 1), "risk_level": lvl,
                    "datetime": datetime.now(),
                }
                st.session_state.registry.append(rec_entry)
                st.session_state.risk_history.append({
                    "datetime": datetime.now(), "risk_pct": round(pct, 1),
                    "label": lvl, "name": pat_name,
                })
                st.success(f"✅ {pat_name} saved to registry!")

            report = f"""HEALTHCARE RISK STRATIFICATION REPORT
======================================
Date/Time    : {datetime.now().strftime('%Y-%m-%d %H:%M')}
Patient      : {pat_name or 'N/A'}
Diagnosis    : {diag} | Gender: {gender} | Age: {age}
Risk Level   : {lvl} | Probability: {pct:.1f}%
Abnormal Labs: {abn_cnt}/6 | Cost: ₹{cost:,} | LOS: {los} days

LAB RESULTS
-----------
Blood Pressure : {bp} mmHg
Blood Sugar    : {bs} mg/dL
Cholesterol    : {chol} mg/dL
Creatinine     : {cr} mg/dL
Hemoglobin     : {hgb} g/dL
Vitamin D      : {vitd} ng/mL

MEDICATIONS ({diag})
--------------------
{chr(10).join('  • ' + m for m in MEDICATIONS.get(diag, []))}

DISCHARGE NOTE
--------------
{'High risk — intensive monitoring required.' if prob >= 0.6 else
 'Moderate risk — 7-day follow-up.' if prob >= 0.3 else
 'Low risk — standard discharge.'}

For clinical decision support only.
"""
            st.download_button("📄 Download report", report,
                               f"risk_report_{date.today()}.txt",
                               "text/plain", use_container_width=True)
        else:
            st.info("Complete the form and click **Calculate risk score**.", icon="ℹ️")
            st.markdown("""
**Model features & coefficients**

| Feature | Coefficient |
|---------|------------|
| Age | 0.127 |
| Treatment Cost | 0.000179 |
| Abnormal Lab Count | 0.327 |
| Intercept | −10.895 |

**Risk thresholds**

| Probability | Level |
|-------------|-------|
| < 30% | 🟢 Low |
| 30–60% | 🟡 Moderate |
| > 60% | 🔴 High |
""")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Lab Report Analyzer
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📄 Lab Report Analyzer")
    st.caption(
        "Upload a patient's lab report PDF — values are extracted automatically, "
        "checked against normal ranges, and fed into the risk model.")

    la, lb = st.columns([1.1, 1], gap="large")

    with la:
        st.markdown('<div class="sec">Upload & patient info</div>', unsafe_allow_html=True)
        pdf_file   = st.file_uploader("Upload lab report (PDF)", type=["pdf"])
        la1, la2   = st.columns(2)
        pdf_age    = la1.number_input("Patient age", 1, 120, 55, key="pdf_age")
        pdf_gender = la2.selectbox("Gender", ["Male", "Female"], key="pdf_gender")
        la3, la4   = st.columns(2)
        pdf_diag   = la3.selectbox("Diagnosis", DIAGNOSES, key="pdf_diag")
        pdf_cost   = la4.number_input("Treatment cost (₹)", 0, 500000, 6000,
                                       step=500, key="pdf_cost")

        st.markdown('<div class="sec">Manual override (optional)</div>', unsafe_allow_html=True)
        st.caption("If any value was not detected from the PDF, enter it manually below.")
        ov1, ov2 = st.columns(2)
        ov_bp    = ov1.number_input("Blood Pressure (0 = skip)",  0,   250,  0,   key="ov_bp")
        ov_bs    = ov2.number_input("Blood Sugar (0 = skip)",     0,   600,  0,   key="ov_bs")
        ov3, ov4 = st.columns(2)
        ov_chol  = ov3.number_input("Cholesterol (0 = skip)",     0,   500,  0,   key="ov_chol")
        ov_cr    = ov4.number_input("Creatinine (0 = skip)",      0.0, 15.0, 0.0, step=0.1, key="ov_cr")
        ov5, ov6 = st.columns(2)
        ov_hgb   = ov5.number_input("Hemoglobin (0 = skip)",      0.0, 25.0, 0.0, step=0.1, key="ov_hgb")
        ov_vitd  = ov6.number_input("Vitamin D (0 = skip)",       0,   150,  0,   key="ov_vitd")

        analyze_btn = st.button("🔍 Analyze lab report", use_container_width=True,
                                type="primary", disabled=(pdf_file is None))

    with lb:
        if pdf_file and analyze_btn:
            with st.spinner("Reading PDF and extracting lab values…"):
                try:
                    raw_text            = extract_pdf_text(pdf_file)
                    found_labs, contexts = parse_labs(raw_text)
                except Exception as e:
                    st.error(f"Could not read PDF: {e}")
                    raw_text             = ""
                    found_labs, contexts = {}, {}

            overrides = {
                "Blood Pressure": ov_bp,  "Blood Sugar": ov_bs,
                "Cholesterol":    ov_chol,"Creatinine":  ov_cr,
                "Hemoglobin":     ov_hgb, "Vitamin D":   ov_vitd,
            }
            for lab, val in overrides.items():
                if val > 0 and lab not in found_labs:
                    found_labs[lab] = val

            all_labs = list(LAB_RANGES["M"].keys())
            detected = list(found_labs.keys())
            missed   = [l for l in all_labs if l not in found_labs]

            st.markdown('<div class="sec">Extraction summary</div>', unsafe_allow_html=True)
            es1, es2 = st.columns(2)
            es1.metric("Labs detected", len(detected))
            es2.metric("Labs missing",  len(missed))
            if missed:
                st.warning(
                    f"Could not extract: **{', '.join(missed)}**. "
                    f"Enter them manually in the override panel.", icon="⚠️")

            if not found_labs:
                st.error(
                    "No lab values could be extracted from this PDF. "
                    "The document may be scanned/image-based or use unsupported formatting. "
                    "Please use the manual override fields.", icon="🚨")
            else:
                st.markdown('<div class="sec">Extracted lab values</div>', unsafe_allow_html=True)
                ranges = LAB_RANGES["F" if pdf_gender == "Female" else "M"]
                for lab in all_labs:
                    if lab not in found_labs:
                        continue
                    val       = found_labs[lab]
                    lo, hi    = ranges[lab]
                    unit      = LAB_UNITS[lab]
                    icon      = LAB_ICONS[lab]
                    ok        = lo <= val <= hi
                    bar_color = "#3B6D11" if ok else ("#BA7517" if val < lo else "#A32D2D")
                    badge     = ('<span class="badge grn">✓ Normal</span>' if ok else
                                 '<span class="badge amb">↓ Below range</span>' if val < lo else
                                 '<span class="badge red">↑ Above range</span>')
                    border    = "#d4edda" if ok else ("#ffe8c0" if val < lo else "#ffc9c9")
                    clamp     = max(lo * 0.7, min(val, hi * 1.3))
                    rang      = hi * 1.3 - lo * 0.7
                    fill      = round((clamp - lo * 0.7) / rang * 100)
                    st.markdown(f"""
<div style="background:white;border:1px solid {border};border-radius:10px;
            padding:10px 14px;margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
    <span style="font-weight:600;font-size:.9rem">{icon} {lab}</span>
    <span>{badge}</span>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:baseline">
    <span style="font-size:1.5rem;font-weight:700;color:{bar_color}">{val}
      <span style="font-size:.8rem;color:#999">{unit}</span></span>
    <span style="font-size:.75rem;color:#aaa">Normal: {lo}–{hi} {unit}</span>
  </div>
  <div style="background:#f0f0f0;border-radius:99px;height:5px;margin-top:6px">
    <div style="width:{fill}%;background:{bar_color};height:5px;border-radius:99px"></div>
  </div>
</div>""", unsafe_allow_html=True)

                st.markdown('<div class="sec">Risk prediction from lab report</div>', unsafe_allow_html=True)
                abn_r, norm_r = check_labs(found_labs, pdf_gender)
                abn_cnt_r     = len(abn_r)
                prob_r        = predict_risk(pdf_age, pdf_cost, abn_cnt_r)
                pct_r         = prob_r * 100
                lvl_r, col_r, ico_r = risk_label(prob_r)

                st.markdown(gauge_svg(pct_r, col_r), unsafe_allow_html=True)
                st.markdown(
                    f"<h3 style='text-align:center;color:{col_r};margin-top:-6px'>"
                    f"{ico_r} {lvl_r} Risk — {pct_r:.1f}%</h3>",
                    unsafe_allow_html=True)

                if prob_r >= 0.6:
                    st.error("High readmission risk — intensive follow-up required.", icon="🚨")
                elif prob_r >= 0.3:
                    st.warning("Moderate risk — follow-up within 7 days.", icon="⚠️")
                else:
                    st.success("Low risk — standard discharge protocol.", icon="✅")

                if abn_r:
                    st.markdown('<div class="sec">Clinical recommendations</div>', unsafe_allow_html=True)
                    for lab, val, lo, hi, direction in abn_r:
                        rec = LAB_RECS.get(lab, {}).get(direction)
                        if rec:
                            css = "rcR" if rec[2] == "R" else "rcA"
                            st.markdown(
                                f'<div class="rc {css}">⚠️ <b>{rec[0]}</b><br>'
                                f'<span style="font-size:.82rem">{rec[1]}</span></div>',
                                unsafe_allow_html=True)

                with st.expander("📋 View extracted PDF text"):
                    st.text(raw_text[:3000] + ("…" if len(raw_text) > 3000 else ""))

                lab_report_out = f"""LAB REPORT ANALYSIS
===================
Date      : {datetime.now().strftime('%Y-%m-%d %H:%M')}
Age       : {pdf_age} | Gender: {pdf_gender}
Diagnosis : {pdf_diag}

DETECTED LAB VALUES
-------------------
{chr(10).join(f'  {l}: {v} {LAB_UNITS[l]}' for l, v in found_labs.items())}

RISK SCORE
----------
Risk Level        : {lvl_r}
Readmission Prob. : {pct_r:.1f}%
Abnormal Labs     : {abn_cnt_r}/{len(found_labs)}

ABNORMAL FINDINGS
-----------------
{chr(10).join(f'  {l}: {v} (Normal: {lo}–{hi})' for l, v, lo, hi, _ in abn_r) or '  None'}
"""
                st.download_button("📥 Download lab analysis", lab_report_out,
                                   f"lab_analysis_{date.today()}.txt",
                                   "text/plain", use_container_width=True)
        elif pdf_file is None:
            st.markdown("""
**How it works**

1. **Upload** any patient lab report as a PDF
2. Values for Blood Pressure, Blood Sugar, Cholesterol, Creatinine, Hemoglobin,
   and Vitamin D are **extracted automatically**
3. Each value is **checked against clinical normal ranges** with a visual indicator
4. The risk model runs **instantly** and shows readmission probability
5. Clinical recommendations are generated for all abnormal values
6. Download the full analysis as a report

**Supported formats:** Text-based PDFs (hospital discharge summaries, pathology reports, outpatient lab results)

> If a value is missed (scanned PDF or unusual formatting),
> use the **Manual Override** panel on the left.
""")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Patient Registry
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 👥 Patient registry")
    reg = st.session_state.registry

    if not reg:
        st.info("No patients saved yet. Use the **Prediction** tab and tick 'Save to registry'.", icon="ℹ️")
    else:
        search   = st.text_input("🔍 Search by name or diagnosis", "")
        filtered = ([r for r in reg if search.lower() in r["name"].lower()
                     or search.lower() in r["diagnosis"].lower()]
                    if search else reg)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total", len(reg))
        k2.error(  f"🔴 High: {sum(1 for r in reg if r['risk_level'] == 'High')}")
        k3.warning(f"🟡 Moderate: {sum(1 for r in reg if r['risk_level'] == 'Moderate')}")
        k4.success(f"🟢 Low: {sum(1 for r in reg if r['risk_level'] == 'Low')}")
        st.divider()

        for r in reversed(filtered):
            lvl, col, ico = risk_label(r["risk_pct"] / 100)
            with st.expander(
                f"{ico} {r['name']} — {r['diagnosis']} — {r['risk_pct']}% ({lvl})",
                expanded=False,
            ):
                ec1, ec2 = st.columns(2)
                ec1.markdown(f"**Age:** {r['age']} &nbsp; **Gender:** {r['gender']}")
                ec1.markdown(f"**Diagnosis:** {r['diagnosis']}")
                ec1.markdown(f"**LOS:** {r['los']} days &nbsp; **Cost:** ₹{r['cost']:,}")
                ec2.markdown(f"**BP:** {r['bp']} | **BS:** {r['bs']} | **Chol:** {r['chol']}")
                ec2.markdown(f"**Creatinine:** {r['cr']} | **Hgb:** {r['hgb']} | **Vit D:** {r['vitd']}")
                st.caption(f"Abnormal labs: {r['abn_cnt']}/6 · Saved: {r['datetime'].strftime('%d %b %Y %H:%M')}")

        if len(st.session_state.risk_history) >= 2:
            st.divider()
            fig = risk_trend_chart(st.session_state.risk_history)
            if fig:
                st.pyplot(fig); plt.close(fig)

        st.divider()
        df_e = pd.DataFrame([{k: v for k, v in r.items() if k != "datetime"} for r in reg])
        st.download_button("📥 Export registry CSV", df_e.to_csv(index=False),
                           "patient_registry.csv", "text/csv", use_container_width=True)
        if st.button("🗑️ Clear registry"):
            st.session_state.registry     = []
            st.session_state.risk_history = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Population Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📊 Population analytics — 200 patients")

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown('<div class="kpi"><div class="kpi-num">200</div>'
                '<div class="kpi-lbl">Total</div></div>', unsafe_allow_html=True)
    k2.markdown('<div class="kpi"><div class="kpi-num" style="color:#A32D2D">69</div>'
                '<div class="kpi-lbl">Deceased</div></div>', unsafe_allow_html=True)
    k3.markdown('<div class="kpi"><div class="kpi-num" style="color:#854F0B">68</div>'
                '<div class="kpi-lbl">Complicated</div></div>', unsafe_allow_html=True)
    k4.markdown('<div class="kpi"><div class="kpi-num" style="color:#3B6D11">63</div>'
                '<div class="kpi-lbl">Recovered</div></div>', unsafe_allow_html=True)

    st.divider()

    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown("**Outcome distribution**")
        st.bar_chart(
            pd.DataFrame({"Outcome": ["Deceased", "Complicated", "Recovered"],
                          "Count":   [69, 68, 63]}).set_index("Outcome"),
            color=["#E24B4A"])
    with dc2:
        st.markdown("**Gender split**")
        st.bar_chart(
            pd.DataFrame({"Gender": ["Female", "Male"], "Count": [106, 94]}).set_index("Gender"),
            color=["#378ADD"])

    st.markdown("**Outcomes by diagnosis**")
    st.bar_chart(pd.DataFrame(DIAG_OUTCOME).T, color=["#BA7517", "#E24B4A", "#639922"])

    st.markdown("**Average treatment cost by diagnosis (₹)**")
    cost_df = (pd.DataFrame(list(AVG_COST.items()), columns=["Diagnosis", "Avg Cost"])
               .set_index("Diagnosis").sort_values("Avg Cost", ascending=False))
    st.bar_chart(cost_df, color=["#378ADD"])

    st.divider()
    i1, i2, i3 = st.columns(3)
    i1.error("**Hypertension** has the highest mortality (12 deaths).")
    i2.warning("**COPD** 50% mortality — highest proportion deceased.")
    i3.success("**Liver Disease** best recovery rate (13/26 recovered).")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Batch Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 📁 Batch prediction")
    st.markdown(
        "Upload a CSV with columns: `Name, Age, Gender, Diagnosis, TreatmentCost, "
        "BloodPressure, BloodSugar, Cholesterol, Creatinine, Hemoglobin, VitaminD`")

    sample = pd.DataFrame({
        "Name":          ["Alice", "Bob", "Carol", "David"],
        "Age":           [67, 45, 78, 53],
        "Gender":        ["Female", "Male", "Female", "Male"],
        "Diagnosis":     ["Diabetes", "COPD", "Hypertension", "Stroke"],
        "TreatmentCost": [7200, 5100, 8900, 6300],
        "BloodPressure": [155, 120, 170, 138],
        "BloodSugar":    [180,  95, 145, 200],
        "Cholesterol":   [210, 165, 190, 230],
        "Creatinine":    [1.4, 0.9, 0.8, 1.6],
        "Hemoglobin":    [11.5, 15.0, 10.2, 14.1],
        "VitaminD":      [18, 35, 12, 28],
    })
    st.download_button("⬇️ Download sample CSV", sample.to_csv(index=False),
                       "sample_patients.csv", "text/csv")

    uploaded = st.file_uploader("Upload patient CSV", type=["csv"], key="batch_csv")
    if uploaded:
        df      = pd.read_csv(uploaded)
        results = []
        for _, row in df.iterrows():
            g   = row.get("Gender", "Male")
            rng = LAB_RANGES["F" if g == "Female" else "M"]
            labs = {
                "Blood Pressure": row.get("BloodPressure", 120),
                "Blood Sugar":    row.get("BloodSugar",    100),
                "Cholesterol":    row.get("Cholesterol",   180),
                "Creatinine":     row.get("Creatinine",    0.9),
                "Hemoglobin":     row.get("Hemoglobin",    14),
                "Vitamin D":      row.get("VitaminD",      30),
            }
            abn  = sum(1 for l, v in labs.items() if v < rng[l][0] or v > rng[l][1])
            prob = predict_risk(row.get("Age", 50), row.get("TreatmentCost", 5000), abn)
            lvl, _, ico = risk_label(prob)
            results.append({
                "Name":          row.get("Name",      "—"),
                "Age":           row.get("Age",       "—"),
                "Diagnosis":     row.get("Diagnosis", "—"),
                "Abnormal Labs": abn,
                "Risk %":        f"{prob * 100:.1f}%",
                "Risk Level":    f"{ico} {lvl}",
            })

        res_df = pd.DataFrame(results)
        st.dataframe(res_df, use_container_width=True)

        b1, b2, b3 = st.columns(3)
        b1.error(  f"🔴 High: {sum(1 for r in results if 'High'     in r['Risk Level'])}")
        b2.warning(f"🟡 Moderate: {sum(1 for r in results if 'Moderate' in r['Risk Level'])}")
        b3.success(f"🟢 Low: {sum(1 for r in results if 'Low'      in r['Risk Level'])}")

        st.download_button("📥 Download results", res_df.to_csv(index=False),
                           "batch_results.csv", "text/csv", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — What-If Simulator
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### 🔮 What-If risk simulator")
    st.caption("Drag sliders and watch the risk score and waterfall update instantly.")

    w1, w2 = st.columns([1.1, 1], gap="large")

    with w1:
        w_age  = st.slider("Age (years)",           30,  90,   58)
        w_cost = st.slider("Treatment cost (₹)",  2000, 10000, 6000, step=100)
        st.markdown('<div class="sec">Lab values</div>', unsafe_allow_html=True)
        w_bp   = st.slider("Blood Pressure (mmHg)", 60,  250,  138)
        w_bs   = st.slider("Blood Sugar (mg/dL)",   40,  400,  155)
        w_chol = st.slider("Cholesterol (mg/dL)",   50,  400,  205)
        w_cr   = st.slider("Creatinine (mg/dL)",   0.1, 10.0,  1.3, step=0.1)
        w_hgb  = st.slider("Hemoglobin (g/dL)",    3.0, 25.0, 12.5, step=0.5)
        w_vitd = st.slider("Vitamin D (ng/mL)",      1,  100,   16)

    with w2:
        lc = {
            "Blood Pressure": (w_bp,   90,  140),
            "Blood Sugar":    (w_bs,   70,  140),
            "Cholesterol":    (w_chol,  0,  200),
            "Creatinine":     (w_cr,  0.6,  1.2),
            "Hemoglobin":     (w_hgb,  13,   17),
            "Vitamin D":      (w_vitd, 20,   50),
        }
        w_abn  = sum(1 for _, (v, lo, hi) in lc.items() if v < lo or v > hi)
        prob_w = predict_risk(w_age, w_cost, w_abn)
        pct_w  = prob_w * 100
        lvl_w, col_w, ico_w = risk_label(prob_w)

        st.markdown(gauge_svg(pct_w, col_w), unsafe_allow_html=True)
        st.markdown(
            f"<h3 style='text-align:center;color:{col_w};margin-top:-6px'>"
            f"{ico_w} {lvl_w} Risk — {pct_w:.1f}%</h3>",
            unsafe_allow_html=True)

        st.pyplot(waterfall_chart(w_age, w_cost, w_abn, prob_w))
        plt.close()

        st.markdown('<div class="sec">Lab status</div>', unsafe_allow_html=True)
        cols2 = st.columns(2)
        for i, (lab, (val, lo, hi)) in enumerate(lc.items()):
            ok = lo <= val <= hi
            cols2[i % 2].markdown(
                f'{LAB_ICONS[lab]} {lab}: <b>{val}</b> '
                f'<span class="badge {"grn" if ok else "red"}">'
                f'{"✓ Normal" if ok else "⚠ Abnormal"}</span>',
                unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: Logistic Regression · sklearn · "
    "Features: Age, TreatmentCost, AbnormalLabCount · "
    "For clinical decision support only — not a substitute for professional medical judgment."
)
