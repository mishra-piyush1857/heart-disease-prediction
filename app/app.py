import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="CardioRisk AI",
    page_icon="🫀",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Full width, no padding waste */
    .block-container {
        padding: 2rem 3rem 2rem 3rem;
        max-width: 1100px;
    }

    /* Top nav bar */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 0 1.5rem 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 2rem;
    }
    .topbar-brand {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e3a5f;
        letter-spacing: -0.3px;
    }
    .topbar-brand span {
        color: #2563eb;
    }
    .topbar-badge {
        background: #eff6ff;
        color: #2563eb;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 99px;
        border: 1px solid #bfdbfe;
        letter-spacing: 0.3px;
    }

    /* Page title block */
    .page-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .page-subtitle {
        font-size: 0.92rem;
        color: #64748b;
        margin-bottom: 2rem;
    }

    /* Section label */
    .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #94a3b8;
        margin-bottom: 0.75rem;
        margin-top: 1.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #f1f5f9;
    }

    /* Input label override */
    label {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: #374151 !important;
    }

    /* Predict button */
    div.stButton > button {
        background: linear-gradient(135deg, #1e3a5f, #2563eb);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-size: 0.95rem;
        font-weight: 600;
        width: 100%;
        margin-top: 1.5rem;
        letter-spacing: 0.2px;
        transition: opacity 0.2s;
    }
    div.stButton > button:hover {
        opacity: 0.9;
    }

    /* Result cards */
    .result-high {
        background: #fff5f5;
        border-left: 5px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-top: 1.5rem;
    }
    .result-low {
        background: #f0fdf4;
        border-left: 5px solid #22c55e;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-top: 1.5rem;
    }
    .result-tag {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
    }
    .result-high .result-tag { color: #dc2626; }
    .result-low  .result-tag { color: #16a34a; }

    .result-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0.1rem 0 0.5rem;
    }
    .result-high .result-title { color: #b91c1c; }
    .result-low  .result-title { color: #15803d; }

    .result-desc {
        font-size: 0.88rem;
        color: #475569;
        margin: 0;
    }

    /* Progress bar */
    .prog-wrap {
        margin-top: 1rem;
    }
    .prog-label {
        font-size: 0.78rem;
        color: #64748b;
        margin-bottom: 4px;
    }
    .prog-bg {
        background: #e2e8f0;
        border-radius: 99px;
        height: 8px;
    }

    /* Divider */
    hr { border-color: #f1f5f9; margin: 1.5rem 0; }

    /* Disclaimer */
    .disclaimer {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #f1f5f9;
        line-height: 1.7;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }

    /* Hide Streamlit default chrome */
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
    header    {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('../models/best_model.joblib')

model = load_model()

# ── Top nav ──────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-brand">Cardio<span>Risk</span> AI</div>
    <div class="topbar-badge">ML-Powered · UCI Dataset</div>
</div>
""", unsafe_allow_html=True)

# ── Page title ───────────────────────────────────────────────
st.markdown("""
<p class="page-title">Heart Disease Risk Assessment</p>
<p class="page-subtitle">Fill in the patient's clinical parameters below. The model will predict cardiovascular disease risk based on patterns learned from 920 patient records.</p>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# INPUT FORM
# ══════════════════════════════════════════════

# ── Section 1: Demographics ──────────────────
st.markdown('<p class="section-label">Patient Demographics</p>', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    age = st.number_input("Age (years)", min_value=20, max_value=80, value=50)
with c2:
    sex = st.selectbox("Biological Sex", ["Male", "Female"])
with c3:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

# ── Section 2: Vitals ────────────────────────
st.markdown('<p class="section-label">Vitals & Lab Results</p>', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
with c2:
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
with c3:
    thalch = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)

# ── Section 3: Cardiac ───────────────────────
st.markdown('<p class="section-label">Cardiac Indicators</p>', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    cp = st.selectbox("Chest Pain Type", [
        "Asymptomatic", "Typical Angina", "Atypical Angina", "Non-Anginal"
    ])
with c2:
    restecg = st.selectbox("Resting ECG Result", [
        "Normal", "ST-T Abnormality", "LV Hypertrophy"
    ])
with c3:
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
with c2:
    slope = st.selectbox("Slope of Peak ST Segment", ["Upsloping", "Flat", "Downsloping"])
with c3:
    pass  # intentional empty for alignment

# ── Section 4: Advanced ──────────────────────
st.markdown('<p class="section-label">Advanced Indicators</p>', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    ca = st.number_input("Major Vessels (Fluoroscopy, 0–3)", min_value=0, max_value=3, value=0)
with c2:
    thal = st.selectbox("Thalassemia Type", [
        "Normal", "Fixed Defect", "Reversable Defect"
    ])
with c3:
    pass  # intentional empty

# ── Predict ──────────────────────────────────
predict = st.button("Run Risk Assessment")

if predict:

    # Build feature row
    input_data = {
        'age':      [age],
        'trestbps': [trestbps],
        'chol':     [chol],
        'fbs':      [1 if fbs == "Yes" else 0],
        'thalch':   [thalch],
        'exang':    [1 if exang == "Yes" else 0],
        'oldpeak':  [oldpeak],
        'ca':       [ca],
        'sex_Male': [1 if sex == "Male" else 0],
        'cp_atypical angina': [1 if cp == "Atypical Angina" else 0],
        'cp_non-anginal':     [1 if cp == "Non-Anginal" else 0],
        'cp_typical angina':  [1 if cp == "Typical Angina" else 0],
        'restecg_normal':             [1 if restecg == "Normal" else 0],
        'restecg_st-t abnormality':   [1 if restecg == "ST-T Abnormality" else 0],
        'slope_flat':       [1 if slope == "Flat" else 0],
        'slope_upsloping':  [1 if slope == "Upsloping" else 0],
        'thal_normal':            [1 if thal == "Normal" else 0],
        'thal_reversable defect': [1 if thal == "Reversable Defect" else 0],
    }

    df_input = pd.DataFrame(input_data)

    # Scale numeric cols using training stats
    scale_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    means = {'age': 53.5, 'trestbps': 131.6, 'chol': 199.4,
             'thalch': 137.6, 'oldpeak': 0.88, 'ca': 0.67}
    stds  = {'age': 9.4,  'trestbps': 17.6,  'chol': 110.0,
             'thalch': 25.1,  'oldpeak': 1.07, 'ca': 0.94}

    for col in scale_cols:
        df_input[col] = (df_input[col] - means[col]) / stds[col]

    prob    = model.predict_proba(df_input)[0][1]
    is_high = prob >= 0.5
    pct     = int(prob * 100)
    label   = "High Risk" if is_high else "Low Risk"
    card    = "result-high" if is_high else "result-low"
    bar_col = "#ef4444" if is_high else "#22c55e"
    desc    = ("This patient shows elevated indicators for cardiovascular disease. "
               "Further clinical evaluation is strongly recommended."
               if is_high else
               "Current indicators suggest lower cardiovascular risk. "
               "Routine monitoring and healthy lifestyle practices are advised.")

    st.markdown(f"""
    <div class="{card}">
        <p class="result-tag">Prediction Result</p>
        <p class="result-title">{label}</p>
        <p class="result-desc">{desc}</p>
        <div class="prog-wrap">
            <p class="prog-label">Risk Probability — {pct}%</p>
            <div class="prog-bg">
                <div style="background:{bar_col}; width:{pct}%;
                            height:8px; border-radius:99px;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key indicators summary
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Clinical Summary**")
    m = st.columns(6)
    m[0].metric("Age",          f"{age} yrs")
    m[1].metric("Max HR",       f"{thalch} bpm")
    m[2].metric("ST Depression",f"{oldpeak}")
    m[3].metric("Vessels",      f"{ca}")
    m[4].metric("Cholesterol",  f"{chol} mg/dl")
    m[5].metric("Resting BP",   f"{trestbps} mmHg")

# ── Disclaimer ───────────────────────────────
st.markdown("""
<p class="disclaimer">
This tool is intended for educational and research purposes only. It does not constitute medical advice,
diagnosis, or treatment. Always consult a qualified healthcare professional for clinical decisions.
Model trained on the UCI Heart Disease Dataset · Accuracy: 85% · Recall: 89%
</p>
""", unsafe_allow_html=True)