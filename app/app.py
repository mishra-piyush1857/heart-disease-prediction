import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="CardioCheck",
    page_icon="❤",
    layout="wide"
)

st.markdown("""
<style>
    /* Base */
    [data-testid="stAppViewContainer"] { background: #f1f5f9; }
    [data-testid="stSidebar"] { background: #0f2744 !important; }
    [data-testid="stSidebar"] * { color: white !important; }
    .block-container { padding: 2rem 2.5rem; }

    /* Sidebar text */
    [data-testid="stSidebar"] .stMarkdown p {
        color: #94b8d8 !important;
        font-size: 0.8rem;
        line-height: 1.9;
    }
    [data-testid="stSidebar"] h1 {
        color: white !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        margin-bottom: 0 !important;
    }
    [data-testid="stSidebar"] h3 {
        color: #38bdf8 !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.2rem !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #1a3a5c !important;
        margin: 1rem 0 !important;
    }

    /* Page title */
    .page-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.3rem;
        letter-spacing: -0.4px;
    }
    .page-sub {
        font-size: 0.84rem;
        color: #64748b;
        margin-bottom: 1.8rem;
    }
    .breadcrumb {
        font-size: 0.7rem;
        color: #94a3b8;
        margin-bottom: 0.5rem;
        letter-spacing: 0.3px;
    }

    /* Section headers */
    .section-head {
        display: flex;
        align-items: center;
        gap: 10px;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px 10px 0 0;
        padding: 0.75rem 1.1rem;
        margin-top: 1.2rem;
        margin-bottom: 0;
    }
    .sec-num {
        background: #0f2744;
        color: white;
        font-size: 0.68rem;
        font-weight: 700;
        min-width: 22px;
        height: 22px;
        border-radius: 6px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }
    .sec-title {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.9px;
        text-transform: uppercase;
        color: #1e293b;
    }
    .section-body {
        background: white;
        border: 1px solid #e2e8f0;
        border-top: none;
        border-radius: 0 0 10px 10px;
        padding: 1.1rem 1.1rem 1.3rem;
        margin-bottom: 0;
    }

    /* Input labels */
    label { font-size: 0.79rem !important; color: #475569 !important; font-weight: 500 !important; }

    /* Button */
    div.stButton > button {
        background: #0f2744 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.5rem !important;
        width: 100%;
        margin-top: 1.2rem;
        letter-spacing: 0.2px;
    }
    div.stButton > button:hover { background: #1a3a5c !important; }

    /* Result */
    .res-high {
        background: white;
        border-radius: 12px;
        border: 1px solid #fecaca;
        border-top: 5px solid #ef4444;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
    }
    .res-low {
        background: white;
        border-radius: 12px;
        border: 1px solid #bbf7d0;
        border-top: 5px solid #22c55e;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
    }
    .res-tag {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .res-high .res-tag { color: #dc2626; }
    .res-low  .res-tag { color: #16a34a; }
    .res-score {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -1px;
        line-height: 1.1;
        margin: 0.2rem 0 0.6rem;
    }
    .res-high .res-score { color: #b91c1c; }
    .res-low  .res-score { color: #15803d; }
    .res-desc { font-size: 0.8rem; color: #475569; line-height: 1.6; }
    .bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.7rem;
        color: #94a3b8;
        margin: 0.9rem 0 4px;
    }
    .bar-bg { background: #f1f5f9; border-radius: 99px; height: 6px; }

    /* Summary table */
    .sum-card {
        background: white;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        padding: 1.1rem 1.3rem;
        margin-top: 0.8rem;
    }
    .sum-title {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 0.8rem;
    }
    .sum-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        padding: 0.38rem 0;
        border-bottom: 1px solid #f8fafc;
        color: #334155;
    }
    .sum-row:last-child { border: none; }
    .sum-k { color: #64748b; }
    .sum-v { font-weight: 600; color: #0f172a; }

    /* Empty state */
    .empty-state {
        background: white;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 2.5rem 1.5rem;
        text-align: center;
        color: #cbd5e1;
        margin-top: 0.5rem;
    }
    .empty-icon { font-size: 2.2rem; margin-bottom: 0.8rem; }
    .empty-title { font-size: 0.85rem; font-weight: 600; color: #94a3b8; margin-bottom: 0.4rem; }
    .empty-desc { font-size: 0.75rem; color: #cbd5e1; line-height: 1.6; }

    /* Disclaimer */
    .disclaimer {
        font-size: 0.7rem;
        color: #94a3b8;
        line-height: 1.7;
        margin-top: 1rem;
        padding-top: 0.8rem;
        border-top: 1px solid #e2e8f0;
    }

    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return joblib.load(os.path.join(base, 'models', 'best_model.joblib'))

model = load_model()

# ════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# CardioCheck")
    st.markdown("CARDIOVASCULAR RISK ASSESSMENT")
    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("""
    **→ Risk Assessment**
    Patient History
    Reports
    Guidelines
    """)
    st.markdown("---")
    st.markdown("### Model Information")
    st.markdown("""
    **Algorithm** · SVM Classifier
    **Dataset** · UCI Heart Disease
    **Patients** · 920 records
    **Accuracy** · 85%
    **Sensitivity** · 89%
    **Specificity** · 79%
    """)
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool uses a Support Vector Machine trained on the UCI Heart Disease dataset to assess cardiovascular risk from clinical parameters.
    """)

# ════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════
st.markdown('<div class="breadcrumb">Dashboard / Risk Assessment</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Cardiovascular Risk Assessment</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Enter the patient\'s clinical parameters to generate a cardiovascular disease risk prediction.</div>', unsafe_allow_html=True)

form_col, result_col = st.columns([1.7, 1], gap="large")

with form_col:

    # Section 1
    st.markdown("""
    <div class="section-head">
        <span class="sec-num">1</span>
        <span class="sec-title">Patient Demographics</span>
    </div>
    <div class="section-body">
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: age  = st.number_input("Age (years)", 20, 80, 50)
    with c2: sex  = st.selectbox("Sex", ["Male", "Female"])
    with c3: fbs  = st.selectbox("Fasting Blood Sugar >120 mg/dl", ["No", "Yes"])
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 2
    st.markdown("""
    <div class="section-head">
        <span class="sec-num">2</span>
        <span class="sec-title">Vitals & Lab Results</span>
    </div>
    <div class="section-body">
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: trestbps = st.number_input("Resting BP (mmHg)", 80, 200, 120)
    with c2: chol     = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    with c3: thalch   = st.number_input("Max Heart Rate (bpm)", 60, 220, 150)
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 3
    st.markdown("""
    <div class="section-head">
        <span class="sec-num">3</span>
        <span class="sec-title">Cardiac Indicators</span>
    </div>
    <div class="section-body">
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: cp      = st.selectbox("Chest Pain Type", ["Asymptomatic","Typical Angina","Atypical Angina","Non-Anginal"])
    with c2: restecg = st.selectbox("Resting ECG", ["Normal","ST-T Abnormality","LV Hypertrophy"])
    with c3: exang   = st.selectbox("Exercise Induced Angina", ["No","Yes"])
    c1, c2, c3 = st.columns(3)
    with c1: oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 7.0, 1.0, 0.1)
    with c2: slope   = st.selectbox("ST Slope", ["Upsloping","Flat","Downsloping"])
    with c3: st.write("")
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 4
    st.markdown("""
    <div class="section-head">
        <span class="sec-num">4</span>
        <span class="sec-title">Advanced Indicators</span>
    </div>
    <div class="section-body">
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: ca   = st.number_input("Major Vessels (0–3)", 0, 3, 0)
    with c2: thal = st.selectbox("Thalassemia", ["Normal","Fixed Defect","Reversable Defect"])
    with c3: st.write("")
    st.markdown("</div>", unsafe_allow_html=True)

    predict = st.button("Generate Risk Assessment")

# ── RESULT COLUMN ─────────────────────────────────
with result_col:
    if predict:
        input_data = {
            'age': [age], 'trestbps': [trestbps], 'chol': [chol],
            'fbs': [1 if fbs=="Yes" else 0],
            'thalch': [thalch],
            'exang': [1 if exang=="Yes" else 0],
            'oldpeak': [oldpeak], 'ca': [ca],
            'sex_Male': [1 if sex=="Male" else 0],
            'cp_atypical angina': [1 if cp=="Atypical Angina" else 0],
            'cp_non-anginal':     [1 if cp=="Non-Anginal" else 0],
            'cp_typical angina':  [1 if cp=="Typical Angina" else 0],
            'restecg_normal':           [1 if restecg=="Normal" else 0],
            'restecg_st-t abnormality': [1 if restecg=="ST-T Abnormality" else 0],
            'slope_flat':      [1 if slope=="Flat" else 0],
            'slope_upsloping': [1 if slope=="Upsloping" else 0],
            'thal_normal':            [1 if thal=="Normal" else 0],
            'thal_reversable defect': [1 if thal=="Reversable Defect" else 0],
        }
        df = pd.DataFrame(input_data)
        means = {'age':53.5,'trestbps':131.6,'chol':199.4,'thalch':137.6,'oldpeak':0.88,'ca':0.67}
        stds  = {'age':9.4, 'trestbps':17.6, 'chol':110.0,'thalch':25.1, 'oldpeak':1.07,'ca':0.94}
        for col in means:
            df[col] = (df[col] - means[col]) / stds[col]

        prob    = model.predict_proba(df)[0][1]
        is_high = prob >= 0.5
        pct     = int(prob * 100)
        label   = "High Risk" if is_high else "Low Risk"
        card    = "res-high" if is_high else "res-low"
        bar_col = "#ef4444" if is_high else "#22c55e"
        desc    = ("Elevated cardiovascular risk detected. Further clinical evaluation and specialist referral are recommended." if is_high
                   else "No significant cardiovascular risk indicators at this time. Routine monitoring and healthy lifestyle practices are advised.")

        st.markdown(f"""
        <div class="{card}">
            <div class="res-tag">Risk Classification</div>
            <div class="res-score">{label}</div>
            <div class="res-desc">{desc}</div>
            <div class="bar-label">
                <span>Risk Probability</span><span>{pct}%</span>
            </div>
            <div class="bar-bg">
                <div style="background:{bar_col};width:{pct}%;height:6px;border-radius:99px;"></div>
            </div>
        </div>

        <div class="sum-card">
            <div class="sum-title">Clinical Summary</div>
            <div class="sum-row"><span class="sum-k">Age</span><span class="sum-v">{age} yrs</span></div>
            <div class="sum-row"><span class="sum-k">Sex</span><span class="sum-v">{sex}</span></div>
            <div class="sum-row"><span class="sum-k">Resting BP</span><span class="sum-v">{trestbps} mmHg</span></div>
            <div class="sum-row"><span class="sum-k">Cholesterol</span><span class="sum-v">{chol} mg/dl</span></div>
            <div class="sum-row"><span class="sum-k">Max Heart Rate</span><span class="sum-v">{thalch} bpm</span></div>
            <div class="sum-row"><span class="sum-k">ST Depression</span><span class="sum-v">{oldpeak}</span></div>
            <div class="sum-row"><span class="sum-k">Major Vessels</span><span class="sum-v">{ca}</span></div>
            <div class="sum-row"><span class="sum-k">Chest Pain</span><span class="sum-v">{cp}</span></div>
            <div class="sum-row"><span class="sum-k">Thalassemia</span><span class="sum-v">{thal}</span></div>
        </div>

        <div class="disclaimer">
            For educational and research purposes only. Not a substitute for
            clinical diagnosis. Always consult a qualified cardiologist.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">♥</div>
            <div class="empty-title">No Assessment Generated</div>
            <div class="empty-desc">
                Complete the clinical parameters on the left
                and click <strong>Generate Risk Assessment</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)