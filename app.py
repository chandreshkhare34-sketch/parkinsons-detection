"""
app.py - Streamlit Interactive Web Application
Parkinson's Disease Detection from Voice Measurements
Submission-ready ML Project (IET DAVV, Indore)
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Add src to system path for modular imports
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from preprocess import FEATURE_CATEGORIES, ALL_FEATURE_COLUMNS, load_dataset, clean_dataset
from predict import load_model, predict_sample, get_sample_patient

# Page configuration
st.set_page_config(
    page_title="Parkinson's Disease Detection | Voice Biomarkers",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern medical dashboard appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1rem;
    }
    .disclaimer-box {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        color: #991b1b;
        font-size: 0.92rem;
        margin-bottom: 1.2rem;
    }
    .healthy-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 2px solid #10b981;
        border-radius: 10px;
        padding: 1.5rem;
        color: #065f46;
        text-align: center;
    }
    .pd-card {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 1.5rem;
        color: #991b1b;
        text-align: center;
    }
    .stat-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cached_model():
    """Cache loaded model for quick inference."""
    try:
        model_path = Path(__file__).resolve().parent / "models" / "best_model.joblib"
        return load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def get_cached_dataset():
    """Cache raw dataset for exploration."""
    return load_dataset()


model_artifact = get_cached_model()
raw_df = get_cached_dataset()

# Title and Academic Affiliation Banner
st.markdown('<div class="main-header">🩺 Parkinson\'s Disease Detection from Voice Measurements</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Biomedical Machine Learning System | <b>IET DAVV, Indore</b></div>', unsafe_allow_html=True)

# Mandatory Medical Disclaimer
st.markdown("""
<div class="disclaimer-box">
    <b>⚠️ ACADEMIC & EDUCATIONAL PROJECT NOTICE:</b> This software is developed strictly as a machine learning engineering project for research and educational purposes. It is <b>NOT</b> certified as a medical diagnostic tool. Do not use its output for clinical diagnosis or treatment planning. Always consult a certified medical specialist or neurologist.
</div>
""", unsafe_allow_html=True)

# Sidebar: Quick Presets & Project Metadata
st.sidebar.header("🕹️ Quick Patient Presets")
st.sidebar.caption("Autofill typical biomedical acoustic measurements:")

col_side1, col_side2 = st.sidebar.columns(2)
btn_healthy = col_side1.button("🟢 Healthy Sample", width="stretch")
btn_pd = col_side2.button("🔴 PD Sample", width="stretch")
btn_median = st.sidebar.button("🔄 Reset to Dataset Median", width="stretch")

# Handle session state for feature inputs
default_healthy = get_sample_patient("healthy")
default_pd = get_sample_patient("parkinsons")
default_median = get_sample_patient("median")

if "current_features" not in st.session_state:
    st.session_state.current_features = default_median.copy()

if btn_healthy:
    st.session_state.current_features = default_healthy.copy()
    st.sidebar.success("Loaded Healthy Patient profile!")
elif btn_pd:
    st.session_state.current_features = default_pd.copy()
    st.sidebar.warning("Loaded Parkinson's Patient profile!")
elif btn_median:
    st.session_state.current_features = default_median.copy()
    st.sidebar.info("Reset to dataset medians.")

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Project Details")
st.sidebar.markdown("""
- **Institution:** IET DAVV, Indore
- **Dataset:** UCI Parkinson's Voice (195 recordings)
- **Active Model:** XGBoost Classifier Pipeline
- **Validation:** 5-Fold Stratified Cross-Validation
- **CV ROC-AUC:** `0.9595` | **Test Recall:** `96.55%`
""")

# Main Navigation Tabs
tab_predict, tab_performance, tab_dataset = st.tabs([
    "🔍 Patient Prediction & Analysis",
    "📊 Model Performance & Reports",
    "📚 Acoustic Features & Dataset"
])

# ----------------- TAB 1: PREDICTION -----------------
with tab_predict:
    st.markdown("### 🎙️ Input Patient Voice Measurements")
    st.caption("Adjust the sliders below or select one of the sidebar presets to populate typical parameters.")

    input_data = {}

    # Category 1: Fundamental Frequency
    with st.expander("1. Vocal Fundamental Frequency (Pitch in Hz)", expanded=True):
        c1, c2, c3 = st.columns(3)
        input_data["MDVP:Fo(Hz)"] = c1.slider(
            "MDVP:Fo(Hz) - Average Pitch",
            min_value=85.0, max_value=265.0,
            value=float(st.session_state.current_features["MDVP:Fo(Hz)"]),
            step=0.5, help="Average vocal fundamental frequency. Typically lower and less variable in PD."
        )
        input_data["MDVP:Fhi(Hz)"] = c2.slider(
            "MDVP:Fhi(Hz) - Max Pitch",
            min_value=100.0, max_value=595.0,
            value=float(st.session_state.current_features["MDVP:Fhi(Hz)"]),
            step=1.0, help="Maximum vocal fundamental frequency."
        )
        input_data["MDVP:Flo(Hz)"] = c3.slider(
            "MDVP:Flo(Hz) - Min Pitch",
            min_value=60.0, max_value=245.0,
            value=float(st.session_state.current_features["MDVP:Flo(Hz)"]),
            step=0.5, help="Minimum vocal fundamental frequency."
        )

    # Category 2: Jitter Variations
    with st.expander("2. Jitter Variants (Frequency Perturbation)", expanded=False):
        c1, c2, c3 = st.columns(3)
        input_data["MDVP:Jitter(%)"] = c1.number_input(
            "MDVP:Jitter(%)", min_value=0.001, max_value=0.035,
            value=float(st.session_state.current_features["MDVP:Jitter(%)"]),
            format="%.5f", help="Percentage of variation in fundamental frequency."
        )
        input_data["MDVP:Jitter(Abs)"] = c2.number_input(
            "MDVP:Jitter(Abs)", min_value=0.000005, max_value=0.000300,
            value=float(st.session_state.current_features["MDVP:Jitter(Abs)"]),
            format="%.6f", help="Absolute jitter in seconds."
        )
        input_data["MDVP:RAP"] = c3.number_input(
            "MDVP:RAP", min_value=0.0005, max_value=0.0250,
            value=float(st.session_state.current_features["MDVP:RAP"]),
            format="%.5f", help="Relative amplitude perturbation."
        )
        c4, c5 = st.columns(2)
        input_data["MDVP:PPQ"] = c4.number_input(
            "MDVP:PPQ", min_value=0.0005, max_value=0.0220,
            value=float(st.session_state.current_features["MDVP:PPQ"]),
            format="%.5f", help="Five-point period perturbation quotient."
        )
        input_data["Jitter:DDP"] = c5.number_input(
            "Jitter:DDP", min_value=0.0015, max_value=0.0700,
            value=float(st.session_state.current_features["Jitter:DDP"]),
            format="%.5f", help="Average absolute difference of differences between jitter cycles."
        )

    # Category 3: Shimmer Variations
    with st.expander("3. Shimmer Variants (Amplitude Perturbation)", expanded=False):
        c1, c2, c3 = st.columns(3)
        input_data["MDVP:Shimmer"] = c1.number_input(
            "MDVP:Shimmer", min_value=0.005, max_value=0.130,
            value=float(st.session_state.current_features["MDVP:Shimmer"]),
            format="%.5f", help="Variation in amplitude from cycle to cycle."
        )
        input_data["MDVP:Shimmer(dB)"] = c2.number_input(
            "MDVP:Shimmer(dB)", min_value=0.05, max_value=1.35,
            value=float(st.session_state.current_features["MDVP:Shimmer(dB)"]),
            format="%.4f", help="Shimmer in decibels."
        )
        input_data["Shimmer:APQ3"] = c3.number_input(
            "Shimmer:APQ3", min_value=0.003, max_value=0.060,
            value=float(st.session_state.current_features["Shimmer:APQ3"]),
            format="%.5f", help="Three-point amplitude perturbation quotient."
        )
        c4, c5, c6 = st.columns(3)
        input_data["Shimmer:APQ5"] = c4.number_input(
            "Shimmer:APQ5", min_value=0.004, max_value=0.085,
            value=float(st.session_state.current_features["Shimmer:APQ5"]),
            format="%.5f", help="Five-point amplitude perturbation quotient."
        )
        input_data["MDVP:APQ"] = c5.number_input(
            "MDVP:APQ", min_value=0.005, max_value=0.145,
            value=float(st.session_state.current_features["MDVP:APQ"]),
            format="%.5f", help="11-point amplitude perturbation quotient."
        )
        input_data["Shimmer:DDA"] = c6.number_input(
            "Shimmer:DDA", min_value=0.010, max_value=0.180,
            value=float(st.session_state.current_features["Shimmer:DDA"]),
            format="%.5f", help="Average absolute difference between consecutive amplitudes."
        )

    # Category 4: Noise & Harmonics
    with st.expander("4. Noise & Harmonics Ratios", expanded=True):
        c1, c2 = st.columns(2)
        input_data["NHR"] = c1.slider(
            "NHR - Noise-to-Harmonics Ratio",
            min_value=0.0005, max_value=0.3200,
            value=float(st.session_state.current_features["NHR"]),
            step=0.001, format="%.4f", help="Ratio of noise to tonal components in voice. High in PD."
        )
        input_data["HNR"] = c2.slider(
            "HNR - Harmonics-to-Noise Ratio (dB)",
            min_value=8.0, max_value=34.0,
            value=float(st.session_state.current_features["HNR"]),
            step=0.1, help="Harmonics-to-noise ratio. Healthy voices have higher values (> 20 dB)."
        )

    # Category 5: Nonlinear Dynamical Complexity
    with st.expander("5. Nonlinear Dynamic Biomarkers (High Diagnostic Relevance)", expanded=True):
        c1, c2, c3 = st.columns(3)
        input_data["spread1"] = c1.slider(
            "spread1 (Nonlinear Freq Variation)",
            min_value=-8.0, max_value=-2.0,
            value=float(st.session_state.current_features["spread1"]),
            step=0.05, help="Fundamental frequency spread. Less negative values indicate pathology."
        )
        input_data["spread2"] = c2.slider(
            "spread2 (Nonlinear Freq Variation)",
            min_value=0.005, max_value=0.460,
            value=float(st.session_state.current_features["spread2"]),
            step=0.005, help="Second nonlinear frequency variation parameter."
        )
        input_data["PPE"] = c3.slider(
            "PPE (Pitch Period Entropy)",
            min_value=0.04, max_value=0.55,
            value=float(st.session_state.current_features["PPE"]),
            step=0.01, help="Pitch period entropy. Strongly elevated in Parkinson's dysphonia."
        )
        c4, c5, c6 = st.columns(3)
        input_data["RPDE"] = c4.slider(
            "RPDE (Recurrence Period Density Entropy)",
            min_value=0.25, max_value=0.70,
            value=float(st.session_state.current_features["RPDE"]),
            step=0.01, help="Dynamical complexity measure."
        )
        input_data["DFA"] = c5.slider(
            "DFA (Fractal Scaling Exponent)",
            min_value=0.55, max_value=0.85,
            value=float(st.session_state.current_features["DFA"]),
            step=0.01, help="Signal fractal scaling exponent."
        )
        input_data["D2"] = c6.slider(
            "D2 (Correlation Dimension)",
            min_value=1.40, max_value=3.70,
            value=float(st.session_state.current_features["D2"]),
            step=0.02, help="Correlation dimension measuring vocal chaos."
        )

    # Prediction Action Button
    st.markdown("---")
    btn_predict = st.button("🚀 Analyze Voice Biomarkers & Predict", type="primary", width="stretch")

    if btn_predict:
        with st.spinner("Executing pipeline through StandardScaler and XGBoost Classifier..."):
            result = predict_sample(input_data, model_artifact=model_artifact)

        st.markdown("### 📋 Clinical Diagnostic Prediction Output")
        col_out1, col_out2 = st.columns([1, 1])

        with col_out1:
            if result["status"] == 1:
                st.markdown(f"""
                <div class="pd-card">
                    <div class="stat-label">Prediction Status</div>
                    <div class="stat-value">⚠️ Parkinson's Disease Detected</div>
                    <p style="margin-top: 0.5rem; font-size: 1.05rem;">Voice acoustic signature exhibits strong indicators of hypokinetic dysphonia.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="healthy-card">
                    <div class="stat-label">Prediction Status</div>
                    <div class="stat-value">✅ Normal / Healthy Voice Pattern</div>
                    <p style="margin-top: 0.5rem; font-size: 1.05rem;">Biomedical acoustics within expected healthy ranges.</p>
                </div>
                """, unsafe_allow_html=True)

        with col_out2:
            st.markdown("#### 🎯 Prediction Probabilities & Confidence")
            prob_pd = result["prob_parkinsons"]
            prob_h = result["prob_healthy"]

            st.write(f"**Probability of Parkinson's Disease:** `{prob_pd * 100:.2f}%`")
            st.progress(float(prob_pd))

            st.write(f"**Probability of Healthy Voice Pattern:** `{prob_h * 100:.2f}%`")
            st.progress(float(prob_h))

            st.markdown(f"**Risk Stratification:** {result['risk_level']}")
            st.markdown(f"**Model Confidence:** `{result['confidence_percent']}%` (via {result['model_used']})")

        # Feature interpretation highlight
        st.markdown("#### 🔬 Key Acoustic Biomarker Assessment")
        c_k1, c_k2, c_k3, c_k4 = st.columns(4)
        c_k1.metric("PPE (Pitch Entropy)", f"{input_data['PPE']:.3f}", "Healthy avg ~0.12" if input_data['PPE'] < 0.15 else "Elevated in PD", delta_color="inverse")
        c_k2.metric("spread1", f"{input_data['spread1']:.2f}", "Healthy avg ~-6.7" if input_data['spread1'] < -6.0 else "Pathologic shift", delta_color="inverse")
        c_k3.metric("HNR (Harmonics)", f"{input_data['HNR']:.1f} dB", "Healthy > 24 dB" if input_data['HNR'] > 22.0 else "Degraded tonal quality", delta_color="normal")
        c_k4.metric("Avg Pitch Fo", f"{input_data['MDVP:Fo(Hz)']:.1f} Hz", "Normal" if input_data['MDVP:Fo(Hz)'] > 160 else "Reduced pitch", delta_color="normal")


# ----------------- TAB 2: MODEL PERFORMANCE -----------------
with tab_performance:
    st.markdown("### 📊 Benchmark of 7 Machine Learning Algorithms")
    st.caption("Evaluated on an 80/20 stratified test split using 5-Fold Stratified Cross-Validation.")

    comparison_path = Path(__file__).resolve().parent / "reports" / "model_comparison.csv"
    if comparison_path.exists():
        comp_df = pd.read_csv(comparison_path)
        # Format percentages
        formatted_df = comp_df.copy()
        for col in ["Accuracy", "Precision", "Recall (Sensitivity)", "Specificity", "F1-Score", "ROC-AUC", "CV ROC-AUC"]:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2%}")
        st.dataframe(formatted_df, width="stretch", hide_index=True)
    else:
        st.info("Run `python src/evaluate.py` to generate the comparison table.")

    st.markdown("---")
    st.markdown("### 📈 Visual Evaluation Artifacts")

    fig_dir = Path(__file__).resolve().parent / "reports" / "figures"
    c_f1, c_f2 = st.columns(2)

    roc_fig = fig_dir / "04_roc_curves_comparison.png"
    cm_fig = fig_dir / "05_best_model_confusion_matrix.png"
    fi_fig = fig_dir / "06_feature_importance.png"
    dist_fig = fig_dir / "01_class_distribution.png"

    if roc_fig.exists():
        c_f1.image(str(roc_fig), caption="Figure 1: Multi-Model ROC Curves Comparison (Test Set)", width="stretch")
    if cm_fig.exists():
        c_f2.image(str(cm_fig), caption="Figure 2: Confusion Matrix of Best Model", width="stretch")

    c_f3, c_f4 = st.columns(2)
    if fi_fig.exists():
        c_f3.image(str(fi_fig), caption="Figure 3: Top Feature Importances (Permutation / Gini)", width="stretch")
    if dist_fig.exists():
        c_f4.image(str(dist_fig), caption="Figure 4: Target Class Distribution & Imbalance", width="stretch")


# ----------------- TAB 3: DATASET EXPLORER -----------------
with tab_dataset:
    st.markdown("### 📚 Biomedical Voice Features Dictionary")
    st.markdown("""
    The dataset was curated by Max Little of the University of Oxford in collaboration with the National Centre for Voice and Speech, Denver, Colorado.
    It comprises **195 sustained vowel [a] phonations** recorded from 31 individuals (23 diagnosed with Parkinson's disease and 8 healthy controls).
    """)

    feat_desc = [
        ("MDVP:Fo(Hz)", "Vocal Pitch", "Average vocal fundamental frequency. Pathological vocal fold stiffness typically depresses or destabilizes pitch."),
        ("MDVP:Fhi(Hz)", "Vocal Pitch", "Maximum vocal fundamental frequency during phonation."),
        ("MDVP:Flo(Hz)", "Vocal Pitch", "Minimum vocal fundamental frequency during phonation."),
        ("MDVP:Jitter(%)", "Jitter (Frequency Perturbation)", "Cycle-to-cycle variation in fundamental frequency (percentage)."),
        ("MDVP:Jitter(Abs)", "Jitter (Frequency Perturbation)", "Absolute cycle-to-cycle frequency variation in microseconds."),
        ("MDVP:RAP", "Jitter (Frequency Perturbation)", "Relative Amplitude Perturbation across 3 consecutive cycles."),
        ("MDVP:PPQ", "Jitter (Frequency Perturbation)", "Pitch Period Perturbation Quotient computed across 5 cycles."),
        ("Jitter:DDP", "Jitter (Frequency Perturbation)", "Average absolute difference of differences between cycles (3 × RAP)."),
        ("MDVP:Shimmer", "Shimmer (Amplitude Perturbation)", "Cycle-to-cycle variation in speech amplitude."),
        ("MDVP:Shimmer(dB)", "Shimmer (Amplitude Perturbation)", "Shimmer measured in decibels."),
        ("Shimmer:APQ3", "Shimmer (Amplitude Perturbation)", "3-point Amplitude Perturbation Quotient."),
        ("Shimmer:APQ5", "Shimmer (Amplitude Perturbation)", "5-point Amplitude Perturbation Quotient."),
        ("MDVP:APQ", "Shimmer (Amplitude Perturbation)", "11-point Amplitude Perturbation Quotient."),
        ("Shimmer:DDA", "Shimmer (Amplitude Perturbation)", "Average absolute difference between consecutive amplitudes (3 × APQ3)."),
        ("NHR", "Noise to Harmonics", "Noise-to-Harmonics Ratio; measures turbulent noise caused by incomplete vocal cord closure."),
        ("HNR", "Harmonics to Noise", "Harmonics-to-Noise Ratio in dB. Healthy phonation produces rich harmonic resonance (> 20 dB)."),
        ("RPDE", "Nonlinear Dynamics", "Recurrence Period Density Entropy; measures the dynamical complexity and chaos of vocal fold vibration."),
        ("DFA", "Nonlinear Dynamics", "Detrended Fluctuation Analysis; stochastic fractal scaling of turbulent airflow."),
        ("spread1", "Nonlinear Dynamics", "Nonlinear fundamental frequency variation parameter. Strongest discriminator in literature."),
        ("spread2", "Nonlinear Dynamics", "Secondary nonlinear fundamental frequency variation parameter."),
        ("D2", "Nonlinear Dynamics", "Correlation Dimension; measures fractal dimension and chaotic degrees of freedom in vocal attractor space."),
        ("PPE", "Nonlinear Dynamics", "Pitch Period Entropy; measures nonsmoothness and entropy of natural frequency transitions.")
    ]
    df_desc = pd.DataFrame(feat_desc, columns=["Feature Name", "Category", "Biomedical & Clinical Significance"])
    st.dataframe(df_desc, width="stretch", hide_index=True)

    st.markdown("### 🔍 Raw Dataset Sample (First 10 Records)")
    st.dataframe(raw_df.head(10), width="stretch")

st.markdown("---")
st.caption("College ML Capstone Project | Department of Computer Engineering / Information Technology, IET DAVV, Indore.")
