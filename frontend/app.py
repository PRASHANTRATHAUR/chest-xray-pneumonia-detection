# ============================================
# Streamlit Frontend for Chest X-Ray Diagnosis
# ============================================

import streamlit as st
import requests
from PIL import Image
import io
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detector",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-normal {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #155724;
    }
    .prediction-pneumonia {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #721c24;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# API CONFIGURATION
# ============================================
API_URL = "http://localhost:8000"

# ============================================
# HELPER FUNCTIONS
# ============================================
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_model_info():
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        return response.json()
    except:
        return None

def predict(image_bytes):
    try:
        files    = {"file": ("xray.jpg", image_bytes, "image/jpeg")}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def plot_confidence(probs):
    fig, ax = plt.subplots(figsize=(8, 3))
    classes = list(probs.keys())
    values  = list(probs.values())
    colors  = ['#28a745' if c == 'NORMAL' else '#dc3545' for c in classes]

    bars = ax.barh(classes, values, color=colors, edgecolor='black', height=0.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confidence Score')
    ax.set_title('Prediction Confidence')

    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', va='center', fontweight='bold')

    ax.axvline(0.5, color='black', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend()
    plt.tight_layout()
    return fig

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/lungs.png", width=80)
    st.title("⚙️ Settings & Info")
    st.divider()

    # API Status
    st.subheader("🔌 API Status")
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline")
        st.info("Start API with:\n```\nuvicorn api.main:app --reload\n```")

    st.divider()

    # Model Info
    st.subheader("🤖 Model Info")
    model_info = get_model_info()
    if model_info:
        st.write(f"**Architecture:** {model_info['architecture']}")
        st.write(f"**Test Accuracy:** {model_info['test_accuracy']}")
        st.write(f"**AUC-ROC:** {model_info['auc_roc']}")
        st.write(f"**Input Size:** {model_info['input_size']}")
        st.write(f"**Device:** {model_info['device']}")

    st.divider()

    # About
    st.subheader("ℹ️ About")
    st.write("""
    This app uses **EfficientNetB0** deep learning model
    to detect pneumonia from chest X-ray images.

    **Built with:**
    - PyTorch
    - FastAPI
    - Streamlit
    - OpenCV
    """)

    st.markdown("**GitHub:** [View Project](https://github.com/PRASHANTRATHAUR/chest-xray-pneumonia-detection)")

# ============================================
# MAIN PAGE
# ============================================

# Header
st.markdown('<div class="main-header">🫁 Chest X-Ray Pneumonia Detector</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered pneumonia detection using EfficientNetB0 | Accuracy: 93.37% | AUC: 0.9705</div>',
            unsafe_allow_html=True)

st.divider()

# Model metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🎯 Test Accuracy", "93.37%")
with col2:
    st.metric("📈 AUC-ROC", "0.9705")
with col3:
    st.metric("🔬 Avg Precision", "0.9817")
with col4:
    st.metric("💊 Pneumonia Recall", "94.1%")

st.divider()

# Upload section
st.subheader("📤 Upload Chest X-Ray Image")
st.write("Upload a chest X-ray image (JPG or PNG) to get an instant diagnosis.")

uploaded_file = st.file_uploader(
    "Choose an X-ray image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a chest X-ray image for pneumonia detection"
)

if uploaded_file is not None:
    # Read image
    image_bytes = uploaded_file.read()
    image       = Image.open(io.BytesIO(image_bytes))

    # Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🖼️ Uploaded X-Ray")
        st.image(image, caption="Uploaded Chest X-Ray",
                use_container_width=True)
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {image.size[0]}x{image.size[1]} pixels")

    with col2:
        st.subheader("🔍 Analysis Results")

        if not api_healthy:
            st.error("❌ API is not running! Please start the FastAPI server.")
        else:
            with st.spinner("🔄 Analyzing X-ray... Please wait"):
                result = predict(image_bytes)

            if "error" in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                prediction = result['prediction']
                confidence = result['confidence']
                probs      = result['probabilities']
                inf_time   = result['inference_time_ms']

                # Prediction result
                if prediction == "NORMAL":
                    st.markdown(
                        f'<div class="prediction-normal">✅ NORMAL<br>'
                        f'<small>Confidence: {confidence:.1%}</small></div>',
                        unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(
                        f'<div class="prediction-pneumonia">⚠️ PNEUMONIA DETECTED<br>'
                        f'<small>Confidence: {confidence:.1%}</small></div>',
                        unsafe_allow_html=True)
                    st.snow()

                st.divider()

                # Confidence chart
                fig = plot_confidence(probs)
                st.pyplot(fig)

                # Details
                st.write(f"⏱️ **Inference Time:** {inf_time:.1f}ms")
                st.write(f"🖥️ **Device:** GPU (CUDA)")

                # Warning
                st.warning("""
                ⚠️ **Medical Disclaimer:** This tool is for educational
                purposes only. Always consult a qualified medical
                professional for diagnosis and treatment.
                """)

else:
    # Show example
    st.info("👆 Please upload a chest X-ray image to get started.")
    st.subheader("📋 How to use:")
    st.write("""
    1. **Upload** a chest X-ray image (JPG or PNG)
    2. **Wait** for the AI to analyze the image
    3. **View** the prediction and confidence scores
    4. **Consult** a doctor for medical advice
    """)

# Footer
st.markdown("""
<div class="footer">
    Built by Arnav | EfficientNetB0 | PyTorch + FastAPI + Streamlit |
    Dataset: Chest X-Ray Images (Pneumonia) - Kaggle
</div>
""", unsafe_allow_html=True)