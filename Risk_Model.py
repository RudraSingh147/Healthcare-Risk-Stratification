import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, io
from datetime import date

st.set_page_config(
    page_title="Healthcare Risk Stratification",
    page_icon="🏥",
    layout="wide",
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container{padding-top:1.5rem;padding-bottom:2rem}
.stTabs [data-baseweb="tab-list"]{gap:8px}
.stTabs [data-baseweb="tab"]{border-radius:8px;padding:6px 20px;font-weight:500}
.kpi-card{background:white;border:1px solid #e8eaf0;border-radius:12px;
          padding:1rem 1.25rem;text-align:center}
.kpi-num{font-size:2rem;font-weight:700;margin:4px 0}
.kpi-lbl{font-size:0.75rem;color:#888;text-transform:uppercase;letter-spacing:.06em}
.sec-hdr{font-size:.72rem;font-weight:600;text-transform:uppercase;
         letter-spacing:.08em;color:#aaa;border-bottom:1px solid #e8eaf0;
         padding-bottom:5px;margin-bottom:12px}
.badge{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.72rem;font-weight:600}
.badge-red{background:#FCEBEB;color:#A32D2D}
.badge-grn{background:#EAF3DE;color:#3B6D11}
.badge-amb{background:#FAEEDA;color:#854F0B}
.rec-card{border-radius:10px;padding:.85rem 1rem;margin-bottom:.5rem;
          border-left:4px solid;font-size:.9rem}
.rec-red{background:#fff5f5;border-color:#E24B4A;color:#4a1010}
.rec-amb{background:#fffbf0;border-color:#BA7517;color:#4a2e00}
.rec-grn{background:#f4fbec;border-color:#639922;color:#1a3300}
.gauge-wrap{text-align:center;padding:1rem 0}
</style>
""", unsafe_allow_html=True)

# ─── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Risk_model1.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

model = load_model()

def sigmoid(x): return 1 / (1 + np.exp(-x))

def predict_risk(age, cost, abnormal_count):
    if model:
        X = np.array([[age, cost, abnormal_count]])
        return float(model.predict_proba(X)[0][1])
    z = 0.12726*age + 0.000179*cost + 0.32716*abnormal_count - 10.8954
    return sigmoid(z)

# ─── Reference data ────────────────────────────────────────────────────────────
LAB_RANGES_M  = {"Blood Pressure":(90,140),"Blood Sugar":(70,140),
                 "Cholesterol":(0,200),"Creatinine":(0.6,1.2),
                 "Hemoglobin":(13,17),"Vitamin D":(20,50)}
LAB_RANGES_F  = {**LAB_RANGES_M, "Hemoglobin":(12,15.5)}

DIAGNOSES = ["Hypertension","Diabetes","Heart Disease","Asthma","Stroke",
             "COPD","Cancer","Arthritis","Kidney Disease","Liver Disease"]

CLINICAL_RECS = {
    "Blood Pressure": {
        "high": ("🩺 Elevated blood pressure detected.",
                 "Consider antihypertensive therapy review, sodium restriction, "
                 "and stress management. Monitor BP every 4 hours.", "red"),
        "low":  ("🩺 Low blood pressure detected.",
                 "Assess for dehydration or sepsis. Increase fluid intake and "
                 "monitor for orthostatic hypotension.", "amb"),
    },
    "Blood Sugar": {
        "high": ("🍬 Hyperglycemia detected.",
                 "Review insulin/oral hypoglycemic dosing. Dietitian consult "
                 "recommended. Monitor glucose every 6 hours.", "red"),
        "low":  ("🍬 Hypoglycemia risk.",
                 "Immediate glucose supplementation. Review meal schedule and "
                 "medication timing.", "red"),
    },
    "Cholesterol": {
        "high": ("❤️ High cholesterol level.",
                 "Consider statin therapy or dose adjustment. Low-fat diet "
                 "counselling and 30-min daily exercise advised.", "amb"),
    },
    "Creatinine": {
        "high": ("🫘 Elevated creatinine — possible renal impairment.",
                 "Nephrology consult recommended. Adjust renally-cleared "
                 "medications. Monitor BUN, GFR, and urine output.", "red"),
        "low":  ("🫘 Low creatinine — possible muscle wasting.",
                 "Nutritional assessment and physiotherapy referral advised.", "amb"),
    },
    "Hemoglobin": {
        "low":  ("🩸 Anemia detected.",
                 "Iron studies, B12, and folate levels recommended. "
                 "Consider iron supplementation or further workup.", "amb"),
        "high": ("🩸 Elevated hemoglobin — possible polycythemia.",
                 "Hematology referral may be needed. Assess for dehydration.", "amb"),
    },
    "Vitamin D": {
        "low":  ("☀️ Vitamin D deficiency.",
                 "Supplement with Vitamin D3. Encourage sunlight exposure "
                 "and dietary sources (fatty fish, dairy).", "amb"),
        "high": ("☀️ Vitamin D toxicity risk.",
                 "Discontinue current supplementation. Check calcium levels. "
                 "Increase fluid intake.", "red"),
    },
}

# ─── Dataset (embedded stats) ──────────────────────────────────────────────────
DIAG_OUTCOME = {
    "Hypertension":  {"Complicated":11,"Deceased":12,"Recovered":5},
    "Diabetes":      {"Complicated":9, "Deceased":3, "Recovered":5},
    "Heart Disease": {"Complicated":9, "Deceased":6, "Recovered":9},
    "Asthma":        {"Complicated":6, "Deceased":3, "Recovered":8},
    "Stroke":        {"Complicated":7, "Deceased":9, "Recovered":7},
    "COPD":          {"Complicated":6, "Deceased":10,"Recovered":4},
    "Cancer":        {"Complicated":5, "Deceased":8, "Recovered":5},
    "Arthritis":     {"Complicated":5, "Deceased":8, "Recovered":4},
    "Kidney Disease":{"Complicated":3, "Deceased":4, "Recovered":3},
    "Liver Disease": {"Complicated":7, "Deceased":6, "Recovered":13},
}
AVG_COST_DIAG = {
    "Arthritis":6679,"Kidney Disease":6329,"Stroke":6267,"Heart Disease":6237,
    "Cancer":6234,"Diabetes":6153,"Asthma":6091,"Liver Disease":5976,
    "Hypertension":5854,"COPD":5533,
}

# ─── Helpers ───────────────────────────────────────────────────────────────────
def check_labs(vals, gender):
    ranges = LAB_RANGES_F if gender == "Female" else LAB_RANGES_M
    abnormal, normal = [], []
    for lab, val in vals.items():
        lo, hi = ranges[lab]
        if val < lo:   abnormal.append((lab, val, lo, hi, "low"))
        elif val > hi: abnormal.append((lab, val, lo, hi, "high"))
        else:          normal.append((lab, val, lo, hi))
    return abnormal, normal

def risk_label(p):
    if p < 0.30: return "Low",      "#639922", "🟢"
    if p < 0.60: return "Moderate", "#BA7517", "🟡"
    return              "High",     "#A32D2D", "🔴"

def gauge_html(pct, color):
    angle = -90 + 180 * pct / 100
    return f"""
<div class="gauge-wrap">
  <svg width="220" height="120" viewBox="0 0 220 120">
    <path d="M20,110 A90,90 0 0,1 200,110" fill="none" stroke="#e8eaf0" stroke-width="18" stroke-linecap="round"/>
    <path d="M20,110 A90,90 0 0,1 200,110" fill="none" stroke="{color}"
          stroke-width="18" stroke-linecap="round"
          stroke-dasharray="{round(2.83*pct)},283" />
    <text x="110" y="105" text-anchor="middle" font-size="28" font-weight="700" fill="{color}">{pct:.0f}%</text>
    <text x="110" y="120" text-anchor="middle" font-size="11" fill="#888">readmission probability</text>
  </svg>
</div>"""

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🏥 Healthcare Risk Stratification")
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Patient Prediction",
    "📊 Population Dashboard",
    "📁 Batch Prediction",
    "🔮 What-If Simulator",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Patient Prediction
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown('<div class="sec-hdr">Patient information</div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        age       = c1.number_input("Age (years)",         1, 120, 55)
        gender    = c2.selectbox("Gender",                 ["Male","Female"])
        diagnosis = c3.selectbox("Diagnosis",              DIAGNOSES)

        c4,c5 = st.columns(2)
        cost = c4.number_input("Treatment cost (₹)", 0, 500000, 5500, step=500)
        los  = c5.number_input("Length of stay (days)", 1, 365, 4)

        st.markdown('<div class="sec-hdr" style="margin-top:1rem">Lab measurements</div>', unsafe_allow_html=True)
        c6,c7   = st.columns(2)
        bp      = c6.number_input("Blood Pressure (mmHg)",  60,  250, 128)
        bs      = c7.number_input("Blood Sugar (mg/dL)",    40,  600, 105)
        c8,c9   = st.columns(2)
        chol    = c8.number_input("Cholesterol (mg/dL)",    50,  500, 185)
        cr      = c9.number_input("Creatinine (mg/dL)",     0.1, 15.0, 0.9, step=0.1)
        c10,c11 = st.columns(2)
        hgb     = c10.number_input("Hemoglobin (g/dL)",     3.0, 25.0, 14.0, step=0.1)
        vitd    = c11.number_input("Vitamin D (ng/mL)",     1,   150,  32)

        run = st.button("🧠  Calculate risk score", use_container_width=True, type="primary")

    with right:
        if run:
            lab_vals = {"Blood Pressure":bp,"Blood Sugar":bs,"Cholesterol":chol,
                        "Creatinine":cr,"Hemoglobin":hgb,"Vitamin D":vitd}
            abnormal, normal = check_labs(lab_vals, gender)
            abn_cnt = len(abnormal)
            prob    = predict_risk(age, cost, abn_cnt)
            pct     = prob * 100
            lvl, col, ico = risk_label(prob)

            # ── Gauge ──────────────────────────────────────────────────────────
            st.markdown(gauge_html(pct, col), unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align:center;color:{col};margin-top:-8px'>{ico} {lvl} Risk</h3>",
                        unsafe_allow_html=True)

            # ── KPIs ───────────────────────────────────────────────────────────
            k1,k2,k3 = st.columns(3)
            k1.metric("Age",            f"{age} yrs")
            k2.metric("Abnormal labs",  f"{abn_cnt} / 6")
            k3.metric("Treatment cost", f"₹{cost:,}")

            st.divider()

            # ── Lab status ─────────────────────────────────────────────────────
            st.markdown('<div class="sec-hdr">Lab status</div>', unsafe_allow_html=True)
            for lab, val, lo, hi, direction in abnormal:
                st.markdown(
                    f'<div class="rec-card rec-red">⚠️ <b>{lab}</b>: {val} '
                    f'&nbsp;<span class="badge badge-red">Abnormal</span>'
                    f'&nbsp;<span style="color:#888;font-size:.8rem">Normal: {lo}–{hi}</span></div>',
                    unsafe_allow_html=True)
            for lab, val, lo, hi in normal:
                st.markdown(
                    f'<div class="rec-card rec-grn">✓ <b>{lab}</b>: {val} '
                    f'&nbsp;<span class="badge badge-grn">Normal</span></div>',
                    unsafe_allow_html=True)

            st.divider()

            # ── Clinical recommendations ───────────────────────────────────────
            if abnormal:
                st.markdown('<div class="sec-hdr">Clinical recommendations</div>', unsafe_allow_html=True)
                for lab, val, lo, hi, direction in abnormal:
                    rec = CLINICAL_RECS.get(lab, {}).get(direction)
                    if rec:
                        title, body, severity = rec
                        css = "rec-red" if severity=="red" else "rec-amb"
                        st.markdown(
                            f'<div class="rec-card {css}"><b>{title}</b><br>'
                            f'<span style="font-size:.85rem">{body}</span></div>',
                            unsafe_allow_html=True)

            st.divider()

            # ── Discharge note ─────────────────────────────────────────────────
            if prob >= 0.6:
                st.error("**High risk** — Intensive follow-up, care coordination, and "
                         "patient education needed before discharge.", icon="🚨")
            elif prob >= 0.3:
                st.warning("**Moderate risk** — Schedule follow-up within 7 days. "
                           "Review medications and lab trends.", icon="⚠️")
            else:
                st.success("**Low risk** — Standard discharge protocol with "
                           "routine follow-up is appropriate.", icon="✅")

            # ── Download report ────────────────────────────────────────────────
            report = f"""HEALTHCARE RISK STRATIFICATION REPORT
======================================
Date        : {date.today()}
Diagnosis   : {diagnosis}
Gender      : {gender}  |  Age: {age} yrs

PREDICTION
----------
Risk Level           : {lvl}
Readmission Prob.    : {pct:.1f}%
Abnormal Lab Count   : {abn_cnt} / 6
Treatment Cost       : ₹{cost:,}
Length of Stay       : {los} days

LAB RESULTS
-----------
Blood Pressure : {bp} mmHg
Blood Sugar    : {bs} mg/dL
Cholesterol    : {chol} mg/dL
Creatinine     : {cr} mg/dL
Hemoglobin     : {hgb} g/dL
Vitamin D      : {vitd} ng/mL

ABNORMAL LABS
-------------
{'None' if not abnormal else chr(10).join([f'  {l}: {v} (Normal: {lo}–{hi})' for l,v,lo,hi,_ in abnormal])}

CLINICAL NOTES
--------------
{'High risk patient — intensive monitoring required.' if prob>=0.6 else
 'Moderate risk — 7-day follow-up recommended.' if prob>=0.3 else
 'Low risk — standard discharge protocol.'}

Generated by Healthcare Risk Stratification System
"""
            st.download_button("📄 Download clinical report", report,
                               file_name=f"risk_report_{date.today()}.txt",
                               mime="text/plain", use_container_width=True)
        else:
            st.info("Fill in patient data and press **Calculate risk score**.", icon="ℹ️")
            st.markdown("""
**Model features**

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

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Population Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Population analytics — 200 patients")

    # ── Top KPIs ───────────────────────────────────────────────────────────────
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown('<div class="kpi-card"><div class="kpi-num">200</div>'
                '<div class="kpi-lbl">Total patients</div></div>', unsafe_allow_html=True)
    k2.markdown('<div class="kpi-card"><div class="kpi-num" style="color:#A32D2D">69</div>'
                '<div class="kpi-lbl">Deceased</div></div>', unsafe_allow_html=True)
    k3.markdown('<div class="kpi-card"><div class="kpi-num" style="color:#854F0B">68</div>'
                '<div class="kpi-lbl">Complicated</div></div>', unsafe_allow_html=True)
    k4.markdown('<div class="kpi-card"><div class="kpi-num" style="color:#3B6D11">63</div>'
                '<div class="kpi-lbl">Recovered</div></div>', unsafe_allow_html=True)

    st.divider()

    # ── Charts row 1 ──────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Outcome distribution**")
        outcome_df = pd.DataFrame({"Outcome":["Deceased","Complicated","Recovered"],
                                   "Count":[69,68,63]})
        st.bar_chart(outcome_df.set_index("Outcome"), color=["#E24B4A"])

    with c2:
        st.markdown("**Gender split**")
        gender_df = pd.DataFrame({"Gender":["Female","Male"],"Count":[106,94]})
        st.bar_chart(gender_df.set_index("Gender"), color=["#378ADD"])

    # ── Diagnosis breakdown ────────────────────────────────────────────────────
    st.markdown("**Outcomes by diagnosis**")
    diag_df = pd.DataFrame(DIAG_OUTCOME).T.reset_index()
    diag_df.columns = ["Diagnosis","Complicated","Deceased","Recovered"]
    diag_df = diag_df.set_index("Diagnosis")
    st.bar_chart(diag_df, color=["#BA7517","#E24B4A","#639922"])

    # ── Avg cost by diagnosis ──────────────────────────────────────────────────
    st.markdown("**Average treatment cost by diagnosis (₹)**")
    cost_df = pd.DataFrame(list(AVG_COST_DIAG.items()), columns=["Diagnosis","Avg Cost"])
    cost_df = cost_df.set_index("Diagnosis").sort_values("Avg Cost", ascending=False)
    st.bar_chart(cost_df, color=["#378ADD"])

    # ── Insights ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Key insights from the dataset**")
    i1,i2,i3 = st.columns(3)
    i1.info("**Hypertension** has the highest mortality (12 deaths) among all diagnoses.")
    i2.warning("**COPD** shows 50% mortality rate — highest proportion of deceased.")
    i3.success("**Liver Disease** has the best recovery rate (13 of 26 patients recovered).")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Batch Prediction
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Batch prediction — upload a CSV")
    st.markdown("Upload a CSV with columns: `Name, Age, Gender, Diagnosis, TreatmentCost, "
                "BloodPressure, BloodSugar, Cholesterol, Creatinine, Hemoglobin, VitaminD`")

    sample = pd.DataFrame({
        "Name":["Alice","Bob","Carol"],
        "Age":[67,45,78],
        "Gender":["Female","Male","Female"],
        "Diagnosis":["Diabetes","COPD","Hypertension"],
        "TreatmentCost":[7200,5100,8900],
        "BloodPressure":[155,120,170],
        "BloodSugar":[180,95,145],
        "Cholesterol":[210,165,190],
        "Creatinine":[1.4,0.9,0.8],
        "Hemoglobin":[11.5,15.0,10.2],
        "VitaminD":[18,35,12],
    })
    st.download_button("⬇️ Download sample CSV", sample.to_csv(index=False),
                       "sample_patients.csv", "text/csv")

    uploaded = st.file_uploader("Upload patient CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown(f"**{len(df)} patients loaded**")

        results = []
        for _, row in df.iterrows():
            g = row.get("Gender","Male")
            ranges = LAB_RANGES_F if g == "Female" else LAB_RANGES_M
            labs = {
                "Blood Pressure": row.get("BloodPressure", 120),
                "Blood Sugar":    row.get("BloodSugar", 100),
                "Cholesterol":    row.get("Cholesterol", 180),
                "Creatinine":     row.get("Creatinine", 0.9),
                "Hemoglobin":     row.get("Hemoglobin", 14),
                "Vitamin D":      row.get("VitaminD", 30),
            }
            abn = sum(1 for lab,val in labs.items()
                      if val < ranges[lab][0] or val > ranges[lab][1])
            prob = predict_risk(row.get("Age",50), row.get("TreatmentCost",5000), abn)
            lvl, _, ico = risk_label(prob)
            results.append({
                "Name":        row.get("Name","—"),
                "Age":         row.get("Age","—"),
                "Diagnosis":   row.get("Diagnosis","—"),
                "Abnormal Labs": abn,
                "Risk %":      f"{prob*100:.1f}%",
                "Risk Level":  f"{ico} {lvl}",
            })

        res_df = pd.DataFrame(results)
        st.dataframe(res_df, use_container_width=True)

        high_risk = sum(1 for r in results if "High" in r["Risk Level"])
        mod_risk  = sum(1 for r in results if "Moderate" in r["Risk Level"])
        low_risk  = sum(1 for r in results if "Low" in r["Risk Level"])

        b1,b2,b3 = st.columns(3)
        b1.error(f"🔴 High risk: **{high_risk}** patients")
        b2.warning(f"🟡 Moderate risk: **{mod_risk}** patients")
        b3.success(f"🟢 Low risk: **{low_risk}** patients")

        st.download_button("📥 Download results CSV",
                           res_df.to_csv(index=False),
                           "batch_risk_results.csv", "text/csv",
                           use_container_width=True)
    else:
        st.markdown("""
**How to use:**
1. Download the sample CSV above
2. Fill in your patient data  
3. Upload it here — predictions appear instantly
""")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — What-If Simulator
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### What-If risk simulator")
    st.caption("Adjust sliders to see how changing patient parameters affects readmission risk in real time.")

    w1, w2 = st.columns([1.1, 1], gap="large")

    with w1:
        st.markdown('<div class="sec-hdr">Adjust parameters</div>', unsafe_allow_html=True)
        w_age  = st.slider("Age (years)",          30, 90, 55)
        w_cost = st.slider("Treatment cost (₹)",   2000, 10000, 5500, step=100)
        w_abn  = st.slider("Abnormal lab count",   0, 6, 2)

        st.markdown('<div class="sec-hdr" style="margin-top:1rem">Individual labs</div>',
                    unsafe_allow_html=True)
        w_bp   = st.slider("Blood Pressure (mmHg)", 60,  250, 120)
        w_bs   = st.slider("Blood Sugar (mg/dL)",   40,  400, 100)
        w_chol = st.slider("Cholesterol (mg/dL)",   50,  400, 180)
        w_cr   = st.slider("Creatinine (mg/dL)",    0.1, 10.0, 0.9, step=0.1)
        w_hgb  = st.slider("Hemoglobin (g/dL)",     3.0, 25.0, 14.0, step=0.5)
        w_vitd = st.slider("Vitamin D (ng/mL)",     1,   100,  30)

    with w2:
        # Recompute abnormal from sliders
        lab_check = {
            "Blood Pressure": (w_bp,  90,  140),
            "Blood Sugar":    (w_bs,  70,  140),
            "Cholesterol":    (w_chol,0,   200),
            "Creatinine":     (w_cr,  0.6, 1.2),
            "Hemoglobin":     (w_hgb, 13,  17),
            "Vitamin D":      (w_vitd,20,  50),
        }
        auto_abn = sum(1 for lab,(val,lo,hi) in lab_check.items()
                       if val<lo or val>hi)

        prob_w  = predict_risk(w_age, w_cost, auto_abn)
        pct_w   = prob_w * 100
        lvl_w, col_w, ico_w = risk_label(prob_w)

        st.markdown(gauge_html(pct_w, col_w), unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center;color:{col_w};margin-top:-8px'>"
                    f"{ico_w} {lvl_w} Risk</h3>", unsafe_allow_html=True)

        st.divider()

        # ── Factor breakdown ───────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">Factor contributions</div>', unsafe_allow_html=True)

        c_age  = 0.12726  * w_age
        c_cost = 0.000179 * w_cost
        c_labs = 0.32716  * auto_abn
        total  = max(c_age + c_cost + c_labs, 0.001)

        st.markdown(f"**Age** — {w_age} yrs")
        st.progress(c_age / total)
        st.markdown(f"**Treatment cost** — ₹{w_cost:,}")
        st.progress(c_cost / total)
        st.markdown(f"**Abnormal labs** — {auto_abn} of 6")
        st.progress(c_labs / total)

        st.divider()

        # ── Lab status pills ───────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">Lab quick check</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for idx, (lab, (val, lo, hi)) in enumerate(lab_check.items()):
            badge_cls = "badge-red" if (val<lo or val>hi) else "badge-grn"
            status    = "⚠ Abnormal" if (val<lo or val>hi) else "✓ Normal"
            cols[idx%2].markdown(
                f'<div style="margin-bottom:6px">{lab}: <b>{val}</b> '
                f'<span class="badge {badge_cls}">{status}</span></div>',
                unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: Logistic Regression · sklearn 1.8.0 · "
    "Features: Age, TreatmentCost, AbnormalLabCount · "
    "For clinical decision support only — not a substitute for professional medical judgment."
)
