

import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model # Explicit Keras import for clarity
from PIL import Image
import os
import time
import pandas as pd

# --- 1. Custom CSS for Jaw-Dropping Professional Design (Apple-esque, Light Theme) ---
st.markdown("""
<style>
/* ----------------------------------------------------- */
/* 1. Global Setup and Background */
/* ----------------------------------------------------- */
.stApp {
    background-color: #f0f3f8; /* Very Light Blue-Gray for soft depth */
    font-family: 'Inter', sans-serif;
}

/* 2. AGGRESSIVE WHITESPACE COMPRESSION (Fixes the 'Empty Box' issue) */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 5rem;
    padding-right: 5rem;
}
div[data-testid="stVerticalBlock"] > div {
    gap: 0.5rem; 
}
h1, h2, h3, h4, p, div.stMarkdown, div.stText, div.stWrite {
    margin: 0 !important;
    padding: 0 !important;
}

/* ----------------------------------------------------- */
/* 3. CORE DESIGN - CARDS (Simulating 3D Depth) */
/* ----------------------------------------------------- */
.st-card-style {
    border-radius: 18px;
    /* Stronger, more layered shadow for "lift" and 3D feel */
    box-shadow: 0 10px 30px rgba(0,0,0,0.1), 0 4px 8px rgba(0,0,0,0.05); 
    padding: 30px;
    background-color: white;
    transition: all 0.3s ease;
    margin-bottom: 30px;
}
.st-card-style:hover {
    box-shadow: 0 15px 40px rgba(0,0,0,0.15), 0 6px 12px rgba(0,0,0,0.08); 
}

/* 4. Sidebar Aesthetic */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff, #e6edf7);
    box-shadow: 2px 0 10px rgba(0,0,0,0.05);
}

/* ----------------------------------------------------- */
/* 5. TYPOGRAPHY & ANIMATION EFFECTS */
/* ----------------------------------------------------- */

/* Hero Title Fade-In Animation */
@keyframes fadeInSlide {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

.hero-title {
    font-size: 4.5em !important;
    font-weight: 900;
    color: #147efb;
    text-align: center;
    /* Deeper 3D shadow effect */
    text-shadow: 2px 2px 0 #cfd5e3, 4px 4px 0 #b4bcc8;
    margin-bottom: 0.2em !important;
    animation: fadeInSlide 1s ease-out; /* Apply animation */
}
.hero-subtitle {
    font-size: 1.8em !important;
    color: #5c6c80;
    text-align: center;
    font-weight: 400;
    margin-bottom: 2em !important;
}
.section-header {
    font-size: 1.8em;
    font-weight: 700;
    color: #2c3e50;
    border-bottom: 3px solid #147efb; 
    padding-bottom: 5px;
    margin-bottom: 15px;
}
.sub-section-header {
    font-size: 1.2em;
    font-weight: 600;
    color: #147efb;
    margin-top: 15px;
    margin-bottom: 10px;
}


/* ----------------------------------------------------- */
/* 6. BUTTONS (Animated Pop Effect) */
/* ----------------------------------------------------- */
div.stButton > button:first-child {
    background-color: #147efb;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 28px;
    font-size: 1.2em;
    font-weight: 700; /* Bolder font */
    box-shadow: 0 6px 12px rgba(20, 126, 251, 0.5); /* Stronger initial shadow */
    transition: all 0.2s ease-out;
}
div.stButton > button:first-child:hover {
    background-color: #0066d1;
    box-shadow: 0 8px 20px rgba(20, 126, 251, 0.8); /* Stronger lift on hover */
    transform: translateY(-3px); /* More pronounced lift */
}
div.stButton > button:first-child:active {
    transform: translateY(0);
    box-shadow: 0 2px 5px rgba(20, 126, 251, 0.4);
}

/* ----------------------------------------------------- */
/* 7. METRIC (Pulsing Attention Effect) */
/* ----------------------------------------------------- */

/* Keyframe for a subtle pulse/glow */
@keyframes metricPulse {
    0% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.01); opacity: 0.95; }
    100% { transform: scale(1); opacity: 1; }
}

[data-testid="stMetricValue"] {
    font-size: 4.5em; /* Bigger metric value */
    font-weight: 900;
    letter-spacing: -2px;
}

/* Apply pulse animation to the metric container after prediction */
.metric-pulse {
    animation: metricPulse 1.5s infinite ease-in-out;
    border-radius: 12px;
    padding: 10px;
}


</style>
""", unsafe_allow_html=True)

# --- 2. Setup and Initial Configuration ---
st.set_page_config(
    page_title="Pneumonia AI Detector",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for the 6-step navigation and data storage
if 'page' not in st.session_state:
    st.session_state.page = 'splash'
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.85 # Adjusted default threshold for high confidence
if 'patient_info' not in st.session_state:
    st.session_state.patient_info = {}
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'prediction_ran' not in st.session_state: # Track if prediction was run for animation control
    st.session_state.prediction_ran = False

# --- Model Parameters ---
MODEL_PATH = os.environ.get('MODEL_PATH', 'pneumonia_model.keras')
TARGET_SIZE = (128, 128)
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

# --- Model Loading (Cached) ---
@st.cache_resource
def load_pneumonia_model():
    if not os.path.exists(MODEL_PATH):
        # Using a styled warning instead of a raw error for better UX
        st.warning(f"Model file not found! Using a placeholder model. Please ensure '{MODEL_PATH}' is available.")
        # Create a mock model for demonstration if file is missing
        class MockModel:
            def predict(self, data):
                # Returns a probability of 0.1 for NORMAL, 0.9 for PNEUMONIA (for testing UI flow)
                return np.array([[0.9]]) 
        return MockModel()
    try:
        # Load the model using the appropriate function
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_pneumonia_model()
if model is None:
    st.stop()

# --- Navigation Functions ---
def navigate_to(page_name):
    st.session_state.page = page_name

# --- 3. Sidebar Content ---
with st.sidebar:
    st.markdown("<h2 style='color: #147efb; text-align: center; margin-bottom: 20px;'>Analysis Pipeline 📋</h2>", unsafe_allow_html=True)
    
    st.markdown("<p style='font-weight: 600; color: #5c6c80;'>Current Session Parameters:</p>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background-color: #e6edf7; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
        <strong>Patient:</strong> {st.session_state.patient_info.get('name', 'N/A')}<br>
        <strong>Threshold:</strong> {st.session_state.threshold:.2f}
    </div>
    """, unsafe_allow_html=True)
    
    st.warning("⚠️ **Disclaimer:** This tool is for educational and research purposes only. Always consult a qualified physician for a final diagnosis.")
    
    st.button("🏠 Start New Session", on_click=navigate_to, args=['splash'], use_container_width=True)


# --- Page Definitions ---

# --- A. SPLASH SCREEN (Welcome Screen with Animated Feel) ---
if st.session_state.page == 'splash':
    st.session_state.feedback_submitted = False
    st.session_state.prediction_ran = False # Reset prediction state
    
    # Hero section with large, center text, no vertical spacing needed now
    st.markdown("<div style='height: 10vh;'></div>", unsafe_allow_html=True)
    st.markdown("<p class='hero-title'>Chest X-ray Assistant 🩺</p>", unsafe_allow_html=True)
    st.markdown("<p class='hero-subtitle'>Rapid, systematic diagnostic support powered by Deep Learning.</p>", unsafe_allow_html=True)
    
    # Centered Button using columns
    col1, col2, col3 = st.columns([1, 0.5, 1])
    with col2:
        st.markdown("<div style='height: 5vh;'></div>", unsafe_allow_html=True) # Add minor spacing above button
        if st.button("Start Application", use_container_width=True):
            navigate_to('home')
    st.markdown("<div style='height: 10vh;'></div>", unsafe_allow_html=True)


# --- B. HOME SCREEN (Project Overview - Card Layout) ---
elif st.session_state.page == 'home':
    st.markdown("<h1 style='text-align: center; color: #2c3e50; font-weight: 800; margin-bottom: 20px;'>Deep Learning Diagnostic Platform</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.4em; color: #5c6c80;'>Leveraging state-of-the-art MobileNetV2 for systematic, high-accuracy diagnostic support.</p>", unsafe_allow_html=True)
    st.write("---")

    # Wrap the entire content area in a single card
    st.markdown('<div class="st-card-style">', unsafe_allow_html=True)

    col_mission, col_model = st.columns(2, gap="large")

    with col_mission:
        st.markdown("<p class='section-header'>Platform Mission & Goals</p>", unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size: 1.1em; color: #34495e;">This tool seamlessly integrates advanced AI into clinical research and workflow:</p>
        <ul style="list-style-type: disc; padding-left: 20px;">
            <li><strong style="color: #147efb;">High Fidelity:</strong> Deploying a fine-tuned MobileNetV2 for robust feature analysis.</li>
            <li><strong style="color: #147efb;">Rapid Screening:</strong> Instant classification to accelerate preliminary assessments.</li>
            <li><strong style="color: #147efb;">Auditability:</strong> Generating systematic reports that track all model and patient parameters.</li>
        </ul>
        """, unsafe_allow_html=True)
        
    with col_model:
        st.markdown("<p class='section-header'>Technical Snapshot</p>", unsafe_allow_html=True)
        # Custom internal box for technical details
        st.markdown("""
        <div style="background-color: #f7f9fc; padding: 25px; border-radius: 12px; border: 1px solid #e6edf7;">
            <p class='sub-section-header' style="margin-top: 0px;">Model Architecture</p>
            <p style="font-size: 1.1em; font-weight: bold; color: #2c3e50;">MobileNetV2 (Transfer Learning)</p>
            <p class='sub-section-header'>Input Specs</p>
            <p><strong>Resolution:</strong> 128x128 pixels</p>
            <p><strong>Classification:</strong> Binary (Normal vs. Pneumonia)</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True) # Close Card

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("▶️ Begin Patient Data Entry", use_container_width=True):
        navigate_to('user_data')

# --- C. PATIENT DATA SCREEN (Form Design) ---
elif st.session_state.page == 'user_data':
    st.markdown("<h1 style='text-align: center; color: #2c3e50; font-weight: 800;'>Step 1: Patient Details 👤</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2em; color: #5c6c80;'>Enter identifying information for systematic report generation.</p>", unsafe_allow_html=True)
    st.write("---")

    # Wrap the form in a single, explicit card
    st.markdown('<div class="st-card-style">', unsafe_allow_html=True)

    st.markdown("<p class='section-header'>Patient Information Required</p>", unsafe_allow_html=True)
    
    col_id, col_name = st.columns(2)
    with col_name:
        patient_name = st.text_input("Full Name / Initials", value=st.session_state.patient_info.get('name', ''), help="Full name or initials.")
    with col_id:
        patient_id = st.text_input("Patient ID / Record No.", value=st.session_state.patient_info.get('id', ''), help="Unique patient identifier.")
    
    col_age, col_gender = st.columns(2)
    with col_age:
        patient_age = st.number_input("Age (Years)", min_value=0, max_value=120, value=st.session_state.patient_info.get('age', 30))
    with col_gender:
        default_index = ["Male", "Female", "Other", "N/A"].index(st.session_state.patient_info.get('gender', 'N/A'))
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other", "N/A"], index=default_index)
    
    # Store data in session state
    st.session_state.patient_info = {
        'name': patient_name,
        'id': patient_id,
        'age': patient_age,
        'gender': patient_gender
    }
    
    st.markdown('</div>', unsafe_allow_html=True) # Close Card
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Continue to Configuration ⚙️", use_container_width=True):
        if not patient_name or not patient_id:
             st.error("🚨 Please enter both Name and ID to proceed.")
        else:
             navigate_to('config')

# --- D. CONFIGURATION SCREEN (Slider/Info Layout) ---
elif st.session_state.page == 'config':
    st.markdown("<h1 style='text-align: center; color: #2c3e50; font-weight: 800;'>Step 2: Analysis Configuration ⚙️</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2em; color: #5c6c80;'>Set the decision line (passing mark) for the AI model.</p>", unsafe_allow_html=True)
    st.write("---")

    # Wrap content in a single card
    st.markdown('<div class="st-card-style">', unsafe_allow_html=True)

    col_model, col_thresh = st.columns(2, gap="large")

    with col_model:
        st.markdown("<p class='section-header'>Model Parameters</p>", unsafe_allow_html=True)
        st.info("Deployed Model: **MobileNetV2 Fine-tuned**")
        st.markdown("""
        <p style="color: #5c6c80;">This fine-tuned network is optimized to prioritize sensitivity, ensuring low false-negative rates for critical diagnostics.</p>
        <p style="font-style: italic; color: #8899a6;">Current Patient: <strong style="color: #147efb;">{name}</strong> (ID: {id})</p>
        """.format(
            name=st.session_state.patient_info.get('name', 'N/A'),
            id=st.session_state.patient_info.get('id', 'N/A')
        ), unsafe_allow_html=True)

    with col_thresh:
        st.markdown("<p class='section-header'>Classification Threshold</p>", unsafe_allow_html=True)
        
        # Custom-styled slider for threshold setting
        new_threshold = st.slider(
            'Set Minimum PNEUMONIA Probability',
            min_value=0.05, max_value=0.99, value=st.session_state.threshold, step=0.01,
            help="Higher value = more cautious AI. Lower value = more sensitive AI."
        )
        st.session_state.threshold = new_threshold
        
        if new_threshold > 0.8:
            st.success(f"**High Confidence Setting:** Diagnosis is PNEUMONIA if Probability $\geq$ **{new_threshold:.2f}**")
        elif new_threshold < 0.3:
            st.warning(f"**High Sensitivity Setting:** Diagnosis is PNEUMONIA if Probability $\geq$ **{new_threshold:.2f}**")
        else:
            st.info(f"**Balanced Setting:** Diagnosis is PNEUMONIA if Probability $\geq$ **{new_threshold:.2f}**")
    
    st.markdown('</div>', unsafe_allow_html=True) # Close Card
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("▶️ Proceed to Image Upload", use_container_width=True):
        navigate_to('analysis')

# --- E. ANALYSIS SCREEN (Image Upload & Prediction) ---
elif st.session_state.page == 'analysis':
    st.markdown("<h1 style='text-align: center; color: #2c3e50; font-weight: 800;'>Step 3: Upload and Prediction 🔍</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size: 1.2em; color: #5c6c80;'>**Patient: {st.session_state.patient_info.get('name', 'N/A')} | Threshold: {st.session_state.threshold:.2f}**</p>", unsafe_allow_html=True)
    st.write("---")

    # Wrap content in a single card
    st.markdown('<div class="st-card-style">', unsafe_allow_html=True)

    input_col, output_col = st.columns([1, 1], gap="large")
    
    with input_col:
        st.markdown("<p class='section-header'>1. X-ray Image Uploader</p>", unsafe_allow_html=True)
        image = st.file_uploader("Upload Chest X-ray (PNG, JPG, JPEG)", accept_multiple_files=False, type=['png', 'jpg', 'jpeg'])

        if image is not None:
            st.markdown("<p class='sub-section-header' style='text-align: center;'>Uploaded Image Preview</p>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)

    with output_col:
        st.markdown("<p class='section-header'>2. AI Diagnostic Results</p>", unsafe_allow_html=True)
        
        # Determine if prediction has run (used to trigger pulse animation)
        prediction_ran_now = False

        if image is None:
            st.info("Please upload an image to begin the automated diagnostic analysis.")
        else:
            # --- Prediction Logic ---
            img_pil = Image.open(image).convert("RGB")
            img_resized = img_pil.resize(TARGET_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = tf.expand_dims(img_array, 0)
            
            with st.spinner("Analyzing high-resolution features (MobileNetV2)..."):
                time.sleep(1.0) 
                prediction = model.predict(img_array)
            
            prediction_ran_now = True
            prediction_score = prediction[0][0]
            is_pneumonia = prediction_score >= st.session_state.threshold

            # Store final result
            st.session_state.final_result = {
                'score': prediction_score, 
                'file_name': image.name, 
                'diagnosis': CLASS_NAMES[1] if is_pneumonia else CLASS_NAMES[0],
                'threshold': st.session_state.threshold,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            st.markdown("<p class='sub-section-header' style='text-align: center;'>Final Conclusion</p>", unsafe_allow_html=True)
            
            # Use the custom pulsing class based on whether prediction just ran
            metric_container_class = 'metric-pulse' if prediction_ran_now else ''
            
            # --- Display Metric with Animation ---
            st.markdown(f'<div class="{metric_container_class}">', unsafe_allow_html=True)
            if not is_pneumonia:
                confidence_percent = (1 - prediction_score) * 100
                st.metric(label="AI Diagnosis", value="NORMAL ✅", delta=f"Confidence: {confidence_percent:.1f}%", delta_color="normal")
                st.success("Result is below the set threshold. No pathology detected.")
                st.balloons()
            else:
                confidence_percent = prediction_score * 100
                st.metric(label="AI Diagnosis", value="PNEUMONIA 🚨", delta=f"Confidence: {confidence_percent:.1f}%", delta_color="inverse")
                st.error("Result exceeds the set threshold. **CRITICAL ALERT** for potential pathology.")
                st.snow()
            st.markdown('</div>', unsafe_allow_html=True) # Close custom container

            st.write("---")
            st.markdown("<p class='sub-section-header'>Probability Scores (Normalized):</p>", unsafe_allow_html=True)
            
            prob_data = {
                'Class': ['NORMAL', 'PNEUMONIA'],
                'Score': [f"{1 - prediction_score:.4f}", f"{prediction_score:.4f}"]
            }
            st.dataframe(pd.DataFrame(prob_data), hide_index=True, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Generate Final Report 📄", use_container_width=True):
                navigate_to('report')

    st.markdown('</div>', unsafe_allow_html=True) # Close Card


# --- F. REPORT SCREEN (Structured Documentation) ---
elif st.session_state.page == 'report':
    st.markdown("<h1 style='text-align: center; color: #2c3e50; font-weight: 800;'>Step 4: Final Diagnostic Report 📄</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2em; color: #5c6c80;'>Systematic documentation of patient data, analysis settings, and final prediction.</p>", unsafe_allow_html=True)
    st.write("---")

    if 'final_result' in st.session_state and 'patient_info' in st.session_state:
        result = st.session_state.final_result
        patient = st.session_state.patient_info
        
        # --- Section I & II: Metadata & Conclusion (Combined Card) ---
        st.markdown('<div class="st-card-style">', unsafe_allow_html=True)
        
        st.markdown('<p class="section-header">I. Report Metadata & Patient Info</p>', unsafe_allow_html=True)
        
        col_a, col_b, col_c = st.columns(3)
        col_a.markdown(f'<p style="font-weight: bold; color: #5c6c80;">Patient Name:</p><p style="font-size: 1.2em; color: #147efb; font-weight: 600;">{patient.get("name", "N/A")}</p>', unsafe_allow_html=True)
        col_b.markdown(f'<p style="font-weight: bold; color: #5c6c80;">Patient ID:</p><p style="font-size: 1.2em; color: #147efb; font-weight: 600;">{patient.get("id", "N/A")}</p>', unsafe_allow_html=True)
        col_c.markdown(f'<p style="font-weight: bold; color: #5c6c80;">Analysis Date:</p><p style="font-size: 1.2em; color: #2c3e50;">{result.get("timestamp", "N/A")}</p>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<p class="section-header">II. AI Diagnostic Conclusion</p>', unsafe_allow_html=True)
        
        final_diagnosis = result.get('diagnosis', 'N/A')
        col_diag, col_recommendation = st.columns([1, 2], gap="large")
        
        with col_diag:
            if final_diagnosis == 'NORMAL':
                st.success(f"## {final_diagnosis.upper()}", icon="✅")
                final_conf = (1 - result['score']) * 100
            else:
                st.error(f"## {final_diagnosis.upper()}", icon="🚨")
                final_conf = result['score'] * 100
            
            st.markdown(f"""
            <div style="background-color: #f7f9fc; padding: 15px; border-radius: 8px;">
                <p style="font-weight: 600; color: #5c6c80; margin-bottom: 5px;">Confidence Score:</p>
                <p style="font-size: 1.5em; font-weight: 800; color: #2c3e50;">{final_conf:.1f}%</p>
                <p style="font-size: 0.9em; color: #8899a6;">(Threshold: {result.get('threshold', 0.5):.2f})</p>
            </div>
            """, unsafe_allow_html=True)

        with col_recommendation:
            st.markdown("<p class='sub-section-header' style='margin-top: 0px;'>Clinical Recommendation</p>", unsafe_allow_html=True)
            if final_diagnosis == 'NORMAL':
                st.info("System Recommendation: No immediate radiological signs of acute pathology detected by AI. Clinical correlation with patient symptoms and history is strongly advised before discharge.")
            else:
                st.error("System Recommendation: High probability of pathology. **IMMEDIATE RADIOLOGIST REVIEW** and clinical investigation are mandatory. AI findings should be treated as a critical initial alert.")

        st.markdown('</div>', unsafe_allow_html=True) # Close Card (Metadata & Conclusion)

        # --- Section III: Technical Audit Data (Separate Card) ---
        st.markdown('<div class="st-card-style">', unsafe_allow_html=True)
        st.markdown('<p class="section-header" style="margin-top: 0px;">III. Technical Audit Data</p>', unsafe_allow_html=True)
        
        audit_data = {
            'Parameter': ['Model Architecture', 'Applied Threshold', 'Raw PNEUMONIA Score', 'Input Filename'],
            'Value': ['MobileNetV2 Fine-tuned', f"{result.get('threshold', 0.5):.2f}", f"{result.get('score', 0.0):.4f}", result.get('file_name', 'N/A')]
        }
        st.dataframe(pd.DataFrame(audit_data), hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True) # Close Card (Audit Data)


        # --- Section IV: Feedback System (Separate Card) ---
        st.markdown("<h2 style='text-align: center; color: #2c3e50; font-weight: 800; margin-top: 20px;'>Platform Feedback</h2>", unsafe_allow_html=True)
        
        st.markdown('<div class="st-card-style">', unsafe_allow_html=True)
        if not st.session_state.feedback_submitted:
            with st.form("feedback_form", clear_on_submit=True):
                col_rating, col_feedback = st.columns([1, 2], gap="large")

                with col_rating:
                    st.markdown("<p class='sub-section-header' style='margin-top: 0px;'>Rating</p>", unsafe_allow_html=True)
                    rating = st.slider("Rate UI/UX & Speed (1=Poor, 5=Excellent)", min_value=1, max_value=5, step=1, value=5, key='rating_slider_input')

                with col_feedback:
                    st.markdown("<p class='sub-section-header' style='margin-top: 0px;'>Comments</p>", unsafe_allow_html=True)
                    feedback_text = st.text_area("Provide any specific feedback.", height=100, key='feedback_text_input')
                
                # Centered submit button
                col_submit, col_empty = st.columns([1, 4])
                with col_submit:
                    submitted = st.form_submit_button("Submit Feedback", type="secondary")

                if submitted:
                    # Simulation of saving feedback
                    st.session_state.feedback_submitted = True
                    st.success("✅ Thank you for your valuable feedback! Your input helps us maintain an industry-level platform.")
                    st.rerun() # Rerun to display success message outside the form structure
        else:
             st.success("Feedback successfully submitted. We appreciate your input.")
        st.markdown('</div>', unsafe_allow_html=True) # Close Card (Feedback)


        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("↩️ Start New Patient Session", use_container_width=True):
            navigate_to('splash')
            
    else:
        st.warning("No analysis data found. Please begin a new session.")
        st.button("Go to Start", on_click=navigate_to, args=['splash'])