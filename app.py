"""
Heart Disease Risk Prediction — Streamlit App
Features: Patient Input Form, Risk Gauge, SHAP Explanation, Radar Chart, PDF Report, AI Chatbot,
          Model Comparison Dashboard, Exploratory Data Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import shap
import os
import io
from datetime import datetime
from fpdf import FPDF
from dotenv import load_dotenv

# Page Config
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load environment
load_dotenv()

# Custom CSS for premium dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Hide Deploy button, hamburger menu and header */
    .stDeployButton {
        display: none !important;
    }
    #MainMenu {
        visibility: hidden !important;
    }
    header {
        visibility: hidden !important;
    }

    * { font-family: 'Inter', sans-serif; }

    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #00D4AA, #00B4D8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #8899AA;
        font-size: 1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(26, 31, 46, 0.9), rgba(20, 25, 40, 0.95));
        border: 1px solid rgba(0, 212, 170, 0.15);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 212, 170, 0.4);
    }
    .metric-card .label {
        color: #8899AA;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        color: #00D4AA;
        font-size: 1.5rem;
        font-weight: 700;
    }

    .risk-container {
        background: linear-gradient(135deg, rgba(26, 31, 46, 0.9), rgba(20, 25, 40, 0.95));
        border: 1px solid rgba(0, 212, 170, 0.15);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }

    .chat-container {
        background: linear-gradient(135deg, rgba(26, 31, 46, 0.9), rgba(20, 25, 40, 0.95));
        border: 1px solid rgba(0, 212, 170, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        max-height: 500px;
        overflow-y: auto;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00D4AA, #00B4D8) !important;
        color: #0E1117 !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover {
        opacity: 0.85 !important;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1A1F2E 100%);
    }

    .quick-btn {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        background: rgba(0, 212, 170, 0.1);
        border: 1px solid rgba(0, 212, 170, 0.3);
        color: #00D4AA;
        font-size: 0.8rem;
        cursor: pointer;
    }

    .section-title {
        color: #00D4AA;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0, 212, 170, 0.2);
    }

    .eda-stat-card {
        background: linear-gradient(135deg, rgba(26, 31, 46, 0.9), rgba(20, 25, 40, 0.95));
        border: 1px solid rgba(0, 212, 170, 0.15);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .eda-stat-card .stat-value {
        color: #00D4AA;
        font-size: 2rem;
        font-weight: 700;
    }
    .eda-stat-card .stat-label {
        color: #8899AA;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


# DATA LOADING FUNCTIONS

@st.cache_resource
def load_artifacts():
    """Load all model artifacts"""
    artifacts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    try:
        model = joblib.load(os.path.join(artifacts_dir, "model.pkl"))
        preprocessor = joblib.load(os.path.join(artifacts_dir, "preprocessor.pkl"))
        feature_names = joblib.load(os.path.join(artifacts_dir, "feature_names.pkl"))
        metrics = joblib.load(os.path.join(artifacts_dir, "metrics.pkl"))
        explainer = joblib.load(os.path.join(artifacts_dir, "explainer.pkl"))
        return model, preprocessor, feature_names, metrics, explainer
    except FileNotFoundError:
        return None, None, None, None, None


@st.cache_resource
def load_all_results():
    """Load all model comparison results (metrics, ROC curves, confusion matrices)"""
    artifacts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    try:
        return joblib.load(os.path.join(artifacts_dir, "all_results.pkl"))
    except FileNotFoundError:
        return None


@st.cache_data
def load_raw_dataset():
    """Load the raw dataset for EDA"""
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heart_disease_uci.csv")
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        return None


@st.cache_data
def load_healthy_averages():
    """Load dataset and compute healthy population averages for radar chart"""
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heart_disease_uci.csv")
    try:
        df = pd.read_csv(data_path)
        healthy = df[df['num'] == 0]
        return {
            'Age': healthy['age'].mean(),
            'Blood Pressure': healthy['trestbps'].mean(),
            'Cholesterol': healthy['chol'].mean(),
            'Max Heart Rate': healthy['thalch'].mean(),
            'ST Depression': healthy['oldpeak'].mean(),
        }
    except Exception:
        return None


# VISUALIZATION FUNCTIONS

def create_risk_gauge(risk_percentage):
    """Create a plotly gauge chart for risk visualization"""
    if risk_percentage < 30:
        color = "#00D4AA"
        risk_level = "LOW RISK"
    elif risk_percentage < 60:
        color = "#FFB347"
        risk_level = "MODERATE RISK"
    else:
        color = "#FF6B6B"
        risk_level = "HIGH RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_percentage,
        number={'suffix': '%', 'font': {'size': 48, 'color': color}},
        title={'text': risk_level, 'font': {'size': 20, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#2A3040',
                     'tickfont': {'color': '#8899AA'}},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': '#1A1F2E',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0, 212, 170, 0.1)'},
                {'range': [30, 60], 'color': 'rgba(255, 179, 71, 0.1)'},
                {'range': [60, 100], 'color': 'rgba(255, 107, 107, 0.1)'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': risk_percentage,
            },
        },
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#E0E0E0'},
    )
    return fig


def create_shap_chart(shap_values, feature_names, input_values):
    """Create a horizontal bar chart for SHAP values"""
    # Get SHAP values for class 1 (at risk)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    elif len(shap_values.shape) == 3:
        sv = shap_values[0, :, 1]
    else:
        sv = shap_values[0]

    # Create DataFrame and sort
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': sv,
        'abs_shap': np.abs(sv),
    }).sort_values('abs_shap', ascending=True).tail(10)

    colors = ['#FF6B6B' if v > 0 else '#00D4AA' for v in shap_df['shap_value']]

    fig = go.Figure(go.Bar(
        x=shap_df['shap_value'],
        y=shap_df['feature'],
        orientation='h',
        marker_color=colors,
        text=[f'{v:+.3f}' for v in shap_df['shap_value']],
        textposition='outside',
        textfont={'color': '#E0E0E0', 'size': 11},
    ))

    fig.update_layout(
        title={
            'text': 'Feature Impact on Risk Prediction',
            'font': {'color': '#00D4AA', 'size': 16},
            'x': 0.5,
        },
        xaxis_title='SHAP Value (Impact on Risk)',
        yaxis_title='',
        height=400,
        margin=dict(l=10, r=80, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis={'gridcolor': 'rgba(136, 153, 170, 0.1)', 'color': '#8899AA',
               'zerolinecolor': 'rgba(136, 153, 170, 0.3)'},
        yaxis={'color': '#E0E0E0'},
        font={'color': '#E0E0E0'},
    )

    # Add annotations
    fig.add_annotation(
        text="🔴 Red = Increases Risk  |  🟢 Green = Decreases Risk",
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=11, color='#8899AA'),
    )

    return fig


def create_radar_chart(patient_values, healthy_avg, labels):
    """Create a radar chart comparing patient values to healthy population averages"""
    # Normalize both sets to 0-1 range for fair comparison
    max_vals = [max(p, h) * 1.2 for p, h in zip(patient_values, healthy_avg)]
    patient_norm = [p / m if m > 0 else 0 for p, m in zip(patient_values, max_vals)]
    healthy_norm = [h / m if m > 0 else 0 for h, m in zip(healthy_avg, max_vals)]

    fig = go.Figure()

    # Healthy population average
    fig.add_trace(go.Scatterpolar(
        r=healthy_norm + [healthy_norm[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(0, 212, 170, 0.15)',
        line=dict(color='#00D4AA', width=2),
        name='Healthy Avg',
    ))

    # Patient values
    fig.add_trace(go.Scatterpolar(
        r=patient_norm + [patient_norm[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(255, 107, 107, 0.15)',
        line=dict(color='#FF6B6B', width=2),
        name='Patient',
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(color='#8899AA'),
        ),
        showlegend=True,
        legend=dict(font=dict(color='#E0E0E0'), orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
        height=380,
        margin=dict(l=60, r=60, t=40, b=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
    )
    return fig


# PDF REPORT

def create_pdf_report(patient_data, risk_pct, feature_names_display):
    """Generate a professional PDF medical report"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(0, 120, 100)
    pdf.cell(0, 15, 'Heart Disease Risk Assessment Report', ln=True, align='C')
    pdf.ln(3)

    # Date
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', ln=True, align='C')
    pdf.ln(5)

    # Divider
    pdf.set_draw_color(0, 180, 150)
    pdf.set_line_width(0.8)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(8)

    # Risk Score Section
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 10, 'Risk Assessment', ln=True)
    pdf.ln(2)

    if risk_pct < 30:
        risk_level = 'LOW RISK'
        risk_color = (0, 160, 120)
        risk_desc = 'The model indicates a low probability of heart disease based on the provided clinical data.'
    elif risk_pct < 60:
        risk_level = 'MODERATE RISK'
        risk_color = (220, 160, 30)
        risk_desc = 'Some risk factors are present. Consider consulting a cardiologist for further evaluation.'
    else:
        risk_level = 'HIGH RISK'
        risk_color = (220, 60, 60)
        risk_desc = 'Multiple risk factors detected. Immediate medical consultation is strongly recommended.'

    pdf.set_font('Helvetica', 'B', 28)
    pdf.set_text_color(*risk_color)
    pdf.cell(0, 15, f'{risk_pct}% - {risk_level}', ln=True, align='C')
    pdf.ln(3)
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 6, risk_desc, align='C')
    pdf.ln(8)

    # Patient Data Table
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 10, 'Patient Clinical Data', ln=True)
    pdf.ln(3)

    # Table header
    pdf.set_fill_color(0, 150, 120)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(95, 9, 'Parameter', border=1, fill=True, align='C')
    pdf.cell(95, 9, 'Value', border=1, fill=True, align='C')
    pdf.ln()

    # Table rows
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(40, 40, 40)
    fill = False
    for display_name, value in feature_names_display.items():
        if fill:
            pdf.set_fill_color(240, 248, 245)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.cell(95, 8, display_name, border=1, fill=True)
        pdf.cell(95, 8, str(value), border=1, fill=True, align='C')
        pdf.ln()
        fill = not fill

    pdf.ln(10)

    # Disclaimer
    pdf.set_draw_color(200, 200, 200)
    pdf.set_line_width(0.4)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(5)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(140, 140, 140)
    pdf.multi_cell(0, 5,
        'Disclaimer: This report is generated by an AI-powered clinical decision support tool '
        'and is intended for informational purposes only. It does not constitute medical advice, '
        'diagnosis, or treatment. Please consult a qualified healthcare professional for clinical decisions.',
        align='C'
    )

    return pdf.output(dest='S').encode('latin-1')


# CHATBOT

def get_chat_response(messages, patient_context):
    """Get response from local Ollama via OpenAI compatible API"""
    try:
        from openai import OpenAI
        
        # Connect to local Ollama instance
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama" # API key is required by the library but ignored by Ollama
        )

        system_msg = f"""You are a helpful medical AI assistant for a heart disease risk prediction application.
You provide clear, educational information about cardiovascular health.

Current patient context:
{patient_context}

Important guidelines:
- Provide informative, educational responses
- Always remind that this is not a substitute for professional medical advice
- Be empathetic and clear
- Keep responses concise but thorough
- Respond in the same language as the user's question"""

        full_messages = [{"role": "system", "content": system_msg}] + messages

        response = client.chat.completions.create(
            model="llama3.2", # Using the fast Llama 3.2 model
            messages=full_messages,
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error connecting to local AI: {str(e)}\n\nPlease make sure Ollama is running in the background."


# MAIN APP

def main():
    # Load artifacts
    model, preprocessor, feature_names, metrics, explainer = load_artifacts()

    if model is None:
        st.error("⚠️ Model artifacts not found! Please run `python train_model.py` first.")
        st.stop()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>❤️ Heart Disease Risk Predictor</h1>
        <p>AI-Powered Clinical Decision Support with Explainable AI (SHAP)</p>
    </div>
    """, unsafe_allow_html=True)

    # Model Metrics Bar
    mcols = st.columns(5)
    metric_items = [
        ("Accuracy", f"{metrics['Accuracy']*100:.1f}%"),
        ("Precision", f"{metrics['Precision']*100:.1f}%"),
        ("Recall", f"{metrics['Recall']*100:.1f}%"),
        ("F1-Score", f"{metrics['F1-Score']*100:.1f}%"),
        ("ROC-AUC", f"{metrics['ROC-AUC']*100:.1f}%"),
    ]
    for col, (label, value) in zip(mcols, metric_items):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sidebar: Patient Input Form (always visible)
    with st.sidebar:
        st.markdown('<div class="section-title">📋 Patient Clinical Data</div>', unsafe_allow_html=True)

        age = st.number_input("Age", min_value=20, max_value=100, value=55, step=1)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
        cp = st.selectbox("Chest Pain Type", options=[
            ("Typical Angina", 0), ("Atypical Angina", 1),
            ("Non-anginal Pain", 2), ("Asymptomatic", 3)
        ], format_func=lambda x: x[0])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=130, step=1)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=240, step=1)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)],
                           format_func=lambda x: x[0])
        restecg = st.selectbox("Resting ECG", options=[
            ("Normal", 0), ("ST-T Abnormality", 1), ("LV Hypertrophy", 2)
        ], format_func=lambda x: x[0])
        thalch = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150, step=1)
        exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)],
                             format_func=lambda x: x[0])
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of ST Segment", options=[
            ("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)
        ], format_func=lambda x: x[0])
        ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0, step=1)
        thal = st.selectbox("Thalassemia", options=[
            ("Normal", 0), ("Fixed Defect", 1), ("Reversible Defect", 2)
        ], format_func=lambda x: x[0])

        predict_btn = st.button("🔍 Predict Risk", use_container_width=True)

    # Prediction Logic (runs on button click, stores in session_state)
    if predict_btn:
        raw_input = {
            'age': age, 'trestbps': trestbps, 'chol': chol, 'thalch': thalch, 'oldpeak': oldpeak, 'ca': ca,
            'sex': str(sex[1]), 'cp': str(cp[1]), 'fbs': str(fbs[1]), 'restecg': str(restecg[1]),
            'exang': str(exang[1]), 'slope': str(slope[1]), 'thal': str(thal[1])
        }
        st.session_state['patient_data'] = raw_input
        input_df_raw = pd.DataFrame([raw_input])
        input_df_prep = preprocessor.transform(input_df_raw)
        risk_proba = model.predict_proba(input_df_prep)[0][1]
        risk_pct = round(risk_proba * 100, 1)
        shap_values = explainer.shap_values(input_df_prep)

        st.session_state['risk_pct'] = risk_pct
        st.session_state['shap_values'] = shap_values
        st.session_state['input_df_prep'] = input_df_prep
        st.session_state['predicted'] = True

    # ============================================================
    # TABS
    # ============================================================
    tab1, tab2, tab3 = st.tabs(["🔍 Risk Prediction", "📊 Model Comparison", "📈 Data Analysis"])

    # ============================================================
    # TAB 1: RISK PREDICTION
    # ============================================================
    with tab1:
        if st.session_state.get('predicted', False):
            risk_pct = st.session_state['risk_pct']
            shap_values = st.session_state['shap_values']
            input_df_prep = st.session_state['input_df_prep']

            col1, col2 = st.columns([1, 1.5])

            with col1:
                st.markdown('<div class="section-title">📊 Risk Assessment</div>', unsafe_allow_html=True)
                fig_gauge = create_risk_gauge(risk_pct)
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Risk level description
                if risk_pct < 30:
                    st.success("✅ **Low Risk** — The model indicates a low probability of heart disease based on the provided clinical data.")
                elif risk_pct < 60:
                    st.warning("⚠️ **Moderate Risk** — Some risk factors are present. Consider consulting a cardiologist for further evaluation.")
                else:
                    st.error("🚨 **High Risk** — Multiple risk factors detected. Immediate medical consultation is strongly recommended.")

            with col2:
                st.markdown('<div class="section-title">🔬 Explainable AI (SHAP Analysis)</div>', unsafe_allow_html=True)
                fig_shap = create_shap_chart(shap_values, feature_names, input_df_prep[0])
                st.plotly_chart(fig_shap, use_container_width=True)

            st.markdown("---")

            # Radar Chart & PDF Report Section
            patient_data = st.session_state.get('patient_data', {})
            radar_col, pdf_col = st.columns([1.5, 1])

            with radar_col:
                st.markdown('<div class="section-title">📡 Patient vs. Healthy Population</div>', unsafe_allow_html=True)
                healthy_avg = load_healthy_averages()
                if healthy_avg and patient_data:
                    radar_labels = list(healthy_avg.keys())
                    patient_radar = [
                        float(patient_data.get('age', 0)),
                        float(patient_data.get('trestbps', 0)),
                        float(patient_data.get('chol', 0)),
                        float(patient_data.get('thalch', 0)),
                        float(patient_data.get('oldpeak', 0)),
                    ]
                    healthy_radar = list(healthy_avg.values())
                    fig_radar = create_radar_chart(patient_radar, healthy_radar, radar_labels)
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.info("Radar chart data is unavailable.")

            with pdf_col:
                st.markdown('<div class="section-title">📄 Download Report</div>', unsafe_allow_html=True)
                if patient_data:
                    # Build human-readable labels for the PDF
                    sex_map = {'1': 'Male', '0': 'Female'}
                    cp_map = {'0': 'Typical Angina', '1': 'Atypical Angina', '2': 'Non-anginal Pain', '3': 'Asymptomatic'}
                    fbs_map = {'0': 'No', '1': 'Yes'}
                    ecg_map = {'0': 'Normal', '1': 'ST-T Abnormality', '2': 'LV Hypertrophy'}
                    exang_map = {'0': 'No', '1': 'Yes'}
                    slope_map = {'0': 'Upsloping', '1': 'Flat', '2': 'Downsloping'}
                    thal_map = {'0': 'Normal', '1': 'Fixed Defect', '2': 'Reversible Defect'}

                    display_data = {
                        'Age': patient_data.get('age', ''),
                        'Sex': sex_map.get(str(patient_data.get('sex', '')), ''),
                        'Chest Pain Type': cp_map.get(str(patient_data.get('cp', '')), ''),
                        'Resting Blood Pressure (mm Hg)': patient_data.get('trestbps', ''),
                        'Cholesterol (mg/dl)': patient_data.get('chol', ''),
                        'Fasting Blood Sugar > 120': fbs_map.get(str(patient_data.get('fbs', '')), ''),
                        'Resting ECG': ecg_map.get(str(patient_data.get('restecg', '')), ''),
                        'Max Heart Rate': patient_data.get('thalch', ''),
                        'Exercise Induced Angina': exang_map.get(str(patient_data.get('exang', '')), ''),
                        'ST Depression (Oldpeak)': patient_data.get('oldpeak', ''),
                        'Slope of ST Segment': slope_map.get(str(patient_data.get('slope', '')), ''),
                        'Major Vessels (0-3)': patient_data.get('ca', ''),
                        'Thalassemia': thal_map.get(str(patient_data.get('thal', '')), ''),
                    }

                    pdf_bytes = create_pdf_report(patient_data, risk_pct, display_data)

                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"heart_risk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                    st.markdown("""
                    <p style="color: #8899AA; font-size: 0.85rem; margin-top: 0.8rem;">
                        This report contains the patient's clinical data, computed risk score,
                        and risk-level interpretation. It can be shared with a healthcare professional.
                    </p>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # Chatbot Section
            st.markdown('<div class="section-title">🤖 AI Medical Assistant</div>', unsafe_allow_html=True)

            # Initialize chat history
            if 'chat_messages' not in st.session_state:
                st.session_state['chat_messages'] = []

            # Quick questions
            st.markdown("**Quick Questions:**")
            qcols = st.columns(4)
            quick_questions = [
                "What does my risk score mean?",
                "How can I lower my risk?",
                "What is cholesterol?",
                "What is SHAP analysis?",
            ]

            for i, (qcol, qq) in enumerate(zip(qcols, quick_questions)):
                if qcol.button(qq, key=f"qq_{i}", use_container_width=True):
                    st.session_state['chat_input'] = qq

            # Chat input
            chat_input = st.chat_input("Ask about your heart health...")

            # Handle quick question click
            if 'chat_input' in st.session_state and st.session_state['chat_input']:
                chat_input = st.session_state.pop('chat_input')

            if chat_input:
                st.session_state['chat_messages'].append({"role": "user", "content": chat_input})

                # Build patient context
                patient_data = st.session_state.get('patient_data', {})
                patient_context = f"Risk Score: {risk_pct}%\n"
                patient_context += "\n".join([f"{k}: {v}" for k, v in patient_data.items()])

                response = get_chat_response(st.session_state['chat_messages'], patient_context)
                st.session_state['chat_messages'].append({"role": "assistant", "content": response})

            # Display chat history
            for msg in st.session_state.get('chat_messages', []):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        else:
            # Initial state - show instructions
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; color: #8899AA;">
                <h2 style="color: #00D4AA; margin-bottom: 1rem;">Welcome to Heart Disease Risk Predictor</h2>
                <p style="font-size: 1.1rem; max-width: 600px; margin: 0 auto; line-height: 1.6;">
                    Fill in the patient's clinical data in the sidebar and click
                    <strong style="color: #00D4AA;">Predict Risk</strong>
                    to get an AI-powered risk assessment with explainable insights.
                </p>
                <br>
                <p style="font-size: 0.9rem;">
                    Powered by Random Forest + SHAP + LLaMA 3.2
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ============================================================
    # TAB 2: MODEL COMPARISON
    # ============================================================
    with tab2:
        st.markdown('<div class="section-title">📊 Model Performance Comparison</div>', unsafe_allow_html=True)

        all_results = load_all_results()

        if all_results is None:
            st.warning("⚠️ Model comparison data not available. Please re-run training with the updated `train_model.py`.")
            st.info("Run: `python train_model.py`")
        else:
            metrics_list = all_results['metrics']
            roc_data = all_results['roc_data']
            cm_data = all_results['confusion_matrices']

            # Best Model Highlight
            best = metrics_list[0]
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(0, 212, 170, 0.1), rgba(0, 180, 216, 0.1));
                        border: 1px solid rgba(0, 212, 170, 0.3); border-radius: 12px; padding: 1rem 1.5rem;
                        margin-bottom: 1.5rem; text-align: center;">
                <span style="color: #8899AA; font-size: 0.9rem;">🏆 Best Performing Model</span><br>
                <span style="color: #00D4AA; font-size: 1.8rem; font-weight: 700;">{best['Model']}</span>
                <span style="color: #8899AA; font-size: 1rem;"> — ROC-AUC: {best['ROC-AUC']*100:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

            # Metrics Table
            st.markdown("### 📋 All Model Metrics")
            metrics_df = pd.DataFrame(metrics_list)
            display_df = metrics_df.copy()
            for c in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
                display_df[c] = display_df[c].apply(lambda x: f"{x*100:.2f}%")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Grouped Bar Chart
            st.markdown("### 📊 Metrics Comparison")
            metrics_df_raw = pd.DataFrame(metrics_list)
            metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            bar_colors = ['#00D4AA', '#00B4D8', '#FFB347', '#FF6B6B', '#B47AFF']

            fig_bar = go.Figure()
            for i, metric in enumerate(metric_cols):
                fig_bar.add_trace(go.Bar(
                    name=metric,
                    x=metrics_df_raw['Model'],
                    y=metrics_df_raw[metric],
                    marker_color=bar_colors[i],
                    text=[f'{v*100:.1f}%' for v in metrics_df_raw[metric]],
                    textposition='outside',
                    textfont={'color': '#E0E0E0', 'size': 10},
                ))
            fig_bar.update_layout(
                barmode='group',
                height=450,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#E0E0E0'},
                xaxis={'color': '#8899AA'},
                yaxis={'color': '#8899AA', 'gridcolor': 'rgba(136,153,170,0.1)',
                       'range': [0, 1.15], 'title': 'Score'},
                legend={'font': {'color': '#E0E0E0'}, 'orientation': 'h',
                        'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'center', 'x': 0.5},
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("---")

            # ROC Curves & Confusion Matrix
            col_roc, col_cm = st.columns([1.2, 1])

            with col_roc:
                st.markdown("### 📈 ROC Curves")
                roc_colors = ['#00D4AA', '#00B4D8', '#FFB347', '#FF6B6B']
                fig_roc = go.Figure()
                for i, (name, data) in enumerate(roc_data.items()):
                    fig_roc.add_trace(go.Scatter(
                        x=data['fpr'], y=data['tpr'],
                        name=f"{name} (AUC={data['auc']:.3f})",
                        line=dict(color=roc_colors[i % len(roc_colors)], width=2.5),
                        mode='lines',
                    ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    name='Random Baseline',
                    line=dict(color='#555', width=1, dash='dash'),
                    mode='lines',
                ))
                fig_roc.update_layout(
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=420,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#E0E0E0'},
                    xaxis={'gridcolor': 'rgba(136,153,170,0.1)', 'color': '#8899AA'},
                    yaxis={'gridcolor': 'rgba(136,153,170,0.1)', 'color': '#8899AA'},
                    legend={'font': {'color': '#E0E0E0', 'size': 11}},
                )
                st.plotly_chart(fig_roc, use_container_width=True)

            with col_cm:
                st.markdown("### 🔢 Confusion Matrix")
                selected_model = st.selectbox("Select Model", list(cm_data.keys()), key="cm_model_select")
                cm = cm_data[selected_model]

                fig_cm = go.Figure(go.Heatmap(
                    z=cm,
                    x=['Predicted<br>Healthy', 'Predicted<br>At Risk'],
                    y=['Actual<br>Healthy', 'Actual<br>At Risk'],
                    colorscale=[[0, '#0E1117'], [0.5, '#1A3A4A'], [1, '#00D4AA']],
                    text=cm,
                    texttemplate='<b>%{text}</b>',
                    textfont={'size': 22, 'color': '#E0E0E0'},
                    showscale=False,
                    hovertemplate='%{y} → %{x}<br>Count: %{z}<extra></extra>',
                ))
                fig_cm.update_layout(
                    height=420,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#E0E0E0'},
                    xaxis={'color': '#8899AA', 'side': 'bottom'},
                    yaxis={'color': '#8899AA', 'autorange': 'reversed'},
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_cm, use_container_width=True)

    # ============================================================
    # TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
    # ============================================================
    with tab3:
        st.markdown('<div class="section-title">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)

        df_raw = load_raw_dataset()

        if df_raw is None:
            st.error("⚠️ Dataset file not found!")
        else:
            # Dataset Overview Cards
            df_target = df_raw['num'].apply(lambda x: 1 if x > 0 else 0)
            total_missing = int(df_raw.isnull().sum().sum())
            positive_pct = df_target.mean() * 100

            ov_cols = st.columns(4)
            overview_items = [
                ("Rows", str(df_raw.shape[0])),
                ("Columns", str(df_raw.shape[1])),
                ("Missing Values", str(total_missing)),
                ("Positive Class", f"{positive_pct:.1f}%"),
            ]
            for ov_col, (lbl, val) in zip(ov_cols, overview_items):
                ov_col.markdown(f"""
                <div class="eda-stat-card">
                    <div class="stat-value">{val}</div>
                    <div class="stat-label">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Target Distribution & Missing Values
            col_target, col_missing = st.columns(2)

            with col_target:
                st.markdown("### 🎯 Target Variable Distribution")
                target_counts = df_target.value_counts()
                labels = ['Healthy (0)', 'At Risk (1+)']
                fig_target = go.Figure(go.Pie(
                    labels=labels,
                    values=[target_counts.get(0, 0), target_counts.get(1, 0)],
                    marker=dict(colors=['#00D4AA', '#FF6B6B']),
                    textinfo='percent+label+value',
                    textfont={'color': '#E0E0E0', 'size': 13},
                    hole=0.45,
                    pull=[0, 0.05],
                ))
                fig_target.update_layout(
                    height=380,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#E0E0E0'},
                    showlegend=False,
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig_target, use_container_width=True)

            with col_missing:
                st.markdown("### ❓ Missing Values by Feature")
                missing = df_raw.isnull().sum()
                missing = missing[missing > 0].sort_values(ascending=True)
                if len(missing) > 0:
                    fig_missing = go.Figure(go.Bar(
                        x=missing.values,
                        y=missing.index,
                        orientation='h',
                        marker_color='#FFB347',
                        text=[f'{v} ({v/len(df_raw)*100:.1f}%)' for v in missing.values],
                        textposition='outside',
                        textfont={'color': '#E0E0E0', 'size': 11},
                    ))
                    fig_missing.update_layout(
                        height=380,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#E0E0E0'},
                        xaxis={'gridcolor': 'rgba(136,153,170,0.1)', 'color': '#8899AA',
                               'title': 'Count'},
                        yaxis={'color': '#E0E0E0'},
                        margin=dict(l=10, r=80, t=20, b=40),
                    )
                    st.plotly_chart(fig_missing, use_container_width=True)
                else:
                    st.success("✅ No missing values in the dataset!")

            st.markdown("---")

            # Correlation Heatmap
            st.markdown("### 🔥 Feature Correlation Heatmap")
            numeric_df = df_raw.select_dtypes(include=[np.number])
            corr = numeric_df.corr()

            # Create custom text with rounded values
            text_matrix = np.round(corr.values, 2).astype(str)

            fig_corr = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale=[
                    [0, '#FF6B6B'],
                    [0.5, '#1A1F2E'],
                    [1, '#00D4AA']
                ],
                zmid=0,
                text=text_matrix,
                texttemplate='%{text}',
                textfont={'size': 9, 'color': '#E0E0E0'},
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
                colorbar=dict(
                    tickfont={'color': '#8899AA'},
                    title=dict(text='Correlation', font={'color': '#8899AA'}),
                ),
            ))
            fig_corr.update_layout(
                height=550,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#E0E0E0'},
                xaxis={'color': '#8899AA', 'tickangle': 45},
                yaxis={'color': '#8899AA', 'autorange': 'reversed'},
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            st.markdown("---")

            # Feature Distributions by Target
            st.markdown("### 📊 Feature Distributions by Heart Disease Status")
            df_plot = df_raw.copy()
            df_plot['target'] = df_plot['num'].apply(lambda x: 'At Risk' if x > 0 else 'Healthy')

            num_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
            feature_labels = {
                'age': 'Age', 'trestbps': 'Blood Pressure',
                'chol': 'Cholesterol', 'thalch': 'Max Heart Rate',
                'oldpeak': 'ST Depression'
            }

            box_cols = st.columns(len(num_features))
            for i, feat in enumerate(num_features):
                with box_cols[i]:
                    fig_box = go.Figure()
                    for status, color in [('Healthy', '#00D4AA'), ('At Risk', '#FF6B6B')]:
                        data = df_plot[df_plot['target'] == status][feat].dropna()
                        fig_box.add_trace(go.Box(
                            y=data, name=status,
                            marker_color=color,
                            boxmean=True,
                            line=dict(color=color),
                        ))
                    fig_box.update_layout(
                        title={'text': feature_labels.get(feat, feat),
                               'font': {'size': 13, 'color': '#00D4AA'}, 'x': 0.5},
                        height=320,
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#E0E0E0'},
                        yaxis={'gridcolor': 'rgba(136,153,170,0.1)', 'color': '#8899AA'},
                        xaxis={'color': '#8899AA'},
                        margin=dict(l=10, r=10, t=45, b=20),
                    )
                    st.plotly_chart(fig_box, use_container_width=True)

            st.markdown("---")

            # Age Distribution by Target
            st.markdown("### 📊 Age Distribution by Heart Disease Status")
            fig_age = go.Figure()
            for status, color in [('Healthy', '#00D4AA'), ('At Risk', '#FF6B6B')]:
                data = df_plot[df_plot['target'] == status]['age'].dropna()
                fig_age.add_trace(go.Histogram(
                    x=data, name=status,
                    marker_color=color, opacity=0.7,
                    nbinsx=25,
                ))
            fig_age.update_layout(
                barmode='overlay',
                height=380,
                xaxis_title='Age',
                yaxis_title='Count',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#E0E0E0'},
                xaxis={'gridcolor': 'rgba(136,153,170,0.1)', 'color': '#8899AA'},
                yaxis={'gridcolor': 'rgba(136,153,170,0.1)', 'color': '#8899AA'},
                legend={'font': {'color': '#E0E0E0'}, 'orientation': 'h',
                        'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'center', 'x': 0.5},
            )
            st.plotly_chart(fig_age, use_container_width=True)

            st.markdown("---")

            # Categorical Feature Analysis
            st.markdown("### 📊 Categorical Feature Analysis")
            cat_features = {
                'sex': {'0': 'Female', '1': 'Male'},
                'cp': {'1': 'Typical Angina', '2': 'Atypical Angina', '3': 'Non-anginal', '4': 'Asymptomatic'},
                'fbs': {'0': 'FBS ≤ 120', '1': 'FBS > 120'},
                'exang': {'0': 'No', '1': 'Yes'},
            }

            cat_cols = st.columns(len(cat_features))
            for i, (feat, label_map) in enumerate(cat_features.items()):
                with cat_cols[i]:
                    if feat in df_plot.columns:
                        temp = df_plot[[feat, 'target']].dropna()
                        temp[feat] = temp[feat].astype(str).map(label_map).fillna(temp[feat].astype(str))
                        counts = temp.groupby([feat, 'target']).size().reset_index(name='count')

                        fig_cat = go.Figure()
                        for status, color in [('Healthy', '#00D4AA'), ('At Risk', '#FF6B6B')]:
                            subset = counts[counts['target'] == status]
                            fig_cat.add_trace(go.Bar(
                                x=subset[feat], y=subset['count'],
                                name=status, marker_color=color,
                            ))
                        fig_cat.update_layout(
                            title={'text': feat.upper(), 'font': {'size': 13, 'color': '#00D4AA'}, 'x': 0.5},
                            barmode='group',
                            height=300,
                            showlegend=False,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': '#E0E0E0'},
                            yaxis={'gridcolor': 'rgba(136,153,170,0.1)', 'color': '#8899AA'},
                            xaxis={'color': '#8899AA'},
                            margin=dict(l=10, r=10, t=45, b=20),
                        )
                        st.plotly_chart(fig_cat, use_container_width=True)


if __name__ == "__main__":
    main()
