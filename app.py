# ==============================================================================
# AI Inspection Supporter - Main Application (v9 - Syntax Fix)
#
# Description: This version fixes the NameError caused by a typo
#              in the st.file_uploader 'type' list.
#
# Author: Your Name (inspired by Gemini)
# Date: 2025-10-28
# ==============================================================================

import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
from PIL import Image
import numpy as np
import cv2  # OpenCV for drawing bounding boxes
from ultralytics import YOLO
# We are now explicitly using hf_hub_download
from huggingface_hub import hf_hub_download

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Inspection Supporter",
    page_icon="🤖",
    layout="wide"
)

# --- MODEL LOADING (MODIFIED with ROBUST Flow) ---

@st.cache_resource
def load_yolo_model():
    """
    Load the pre-trained YOLOv8 model for PCB defect detection.
    This function explicitly downloads the model file from Hugging Face
    (like the music project did) and then loads it from the local path.
    """
    print("[INFO] Loading YOLOv8 PCB Defect Model...")
    
    # Define the model repo and file on Hugging Face
    model_id = "keremberke/yolov8s-pcb-defect-segmentation"
    filename = "best.pt" # The actual weights file in that repo

    try:
        #
        # THIS IS THE "MUSIC PROJECT" FLOW
        #
        # Step 1: Explicitly download the model file.
        # This function is smart and will use a cache.
        print(f"[INFO] Downloading model file '{filename}' from repo '{model_id}'...")
        local_model_path = hf_hub_download(repo_id=model_id, filename=filename)
        print(f"[INFO] Model file is ready at local path: {local_model_path}")

        # Step 2: Load the model from the *local downloaded path*
        # Now YOLO has no ambiguity, it's loading a real file path.
        model = YOLO(local_model_path)
        
        print("[INFO] YOLOv8 Model loaded successfully.")
        return model
        
    except Exception as e:
        st.error(f"Failed to download or load the YOLO model '{model_id}'. Error: {e}")
        print(f"[ERROR] Failed to load YOLO model: {e}")
        return None

@st.cache_resource
def configure_gemini_api():
    """
    Configure and initialize the Google Gemini generative model.
    """
    print("[INFO] Configuring Gemini API...")
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API Key not found. Please set it in your .env file.")
        return None
    
    genai.configure(api_key=api_key)
    # Using the latest reliable model
    model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
    print("[INFO] Gemini API configured.")
    return model

# --- CORE FUNCTIONS ---

def run_defect_detection(image_data, yolo_model):
    """
    Run the YOLOv8 model on the uploaded image to detect defects.
    It returns the annotated image and a list of found defects.
    """
    print("[INFO] Running defect detection...")
    image_np = np.array(image_data.convert('RGB'))
    
    # Run prediction, but set an *extremely* low confidence threshold (conf=0.01)
    results = yolo_model.predict(image_np, conf=0.01)
    
    defect_list = []
    annotated_image = image_np.copy()
    
    if results and results[0].boxes:
        boxes = results[0].boxes
        # Get the class name map (e.g., 0: 'Dry_joint')
        class_names = results[0].names 

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_index = int(box.cls[0])
            cls_name = class_names.get(cls_index, 'Unknown')
            confidence = float(box.conf[0])
            
            defect_list.append({
                "type": cls_name,
                "confidence": f"{confidence*100:.2f}%"
            })
            
            # Draw bounding box (Red)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            label = f"{cls_name}: {confidence*100:.1f}%"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_image, (x1, y1 - h - 5), (x1 + w, y1), (0, 0, 255), -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    print(f"[INFO] Detection complete. Found {len(defect_list)} defects.")
    
    # Convert annotated image from BGR (OpenCV default) back to RGB (PIL/Streamlit default)
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB) # <-- Fixed: _ to 2
    
    return annotated_image_rgb, defect_list


def generate_inspection_report(defect_list, gemini_model):
    """
    Generate a professional QC report using Gemini based on the defect list.
    (This is updated for the 'keremberke' model's classes)
    """
    print("[INFO] Generating AI Manager's report...")
    if not defect_list:
        return "PASS: No defects detected during visual inspection."

    defect_string = "\n".join([f"- {d['type']} (Confidence: {d['confidence']})" for d in defect_list])

    # This prompt is updated for the new model's defect classes
    prompt = f"""
    You are an expert AI Quality Control Manager for an electronics production line.
    An AOI tool has scanned a PCB and found the following potential defects:

    {defect_string}

    Defect Guide:
    - 'Dry_joint': A poor solder connection (虚焊).
    - 'Incorrect_installation': A component is installed incorrectly (安装错误).
    - 'PCB_damage': Physical damage to the board itself (PCB板损伤).
    - 'Short_circuit': An improper connection between two points (短路).

    Your task is to write a concise, professional inspection report.
    The report MUST include three sections:
    1.  **Inspection Summary:** A one-sentence overview.
    2.  **Detected Defects:** A bullet-point list of the items found.
    3.  **Recommended Action:** A clear, actionable next step.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        print("[INFO] Report generated successfully.")
        return response.text
    except Exception as e:
        print(f"[ERROR] Gemini API call failed: {e}")
        return f"Error generating report: {e}"

# --- STREAMLIT UI ---

st.title("🤖 AI Inspection Supporter")
st.markdown("##### Upload a Printed Circuit Board (PCB) image to run an AI-powered defect inspection.")

# Load models
yolo_model = load_yolo_model()
gemini_model = configure_gemini_api()

st.sidebar.header("About This App")
st.sidebar.info(
    "This application demonstrates an 'Image-to-Report' pipeline. \n\n"
    "1. A **YOLOv8** model (from Hugging Face) detects visual defects. \n\n"
    "2. A **Generative AI** (Google Gemini) interprets the findings to write a formal QC report."
)

#
# === THIS IS THE FIX (Line 195) ===
#
uploaded_file = st.file_uploader("Choose a PCB image...", type=["jpg", "jpeg", "png"]) # <-- Fixed: Removed _CHAR_

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.header("Inspection Results")
    
    if yolo_model is None or gemini_model is None:
        st.error("One of the AI models failed to load. Please check the terminal for errors.")
    else:
        with st.spinner("Inspecting image with YOLOv8... (This may take a moment on first run)..."):
            annotated_image, defect_list = run_defect_detection(image, yolo_model)
        
        if not defect_list:
            st.success("✅ **Inspection PASSED**")
            st.markdown("The AI vision model did not find any defects on this board.")
            st.image(image, caption="Original Uploaded Image", use_column_width=True)
        
        else:
            st.error(f"❌ **Inspection FAILED: {len(defect_list)} potential defects detected.**")
            
            with st.spinner("AI Manager is analyzing defects and writing the report..."):
                report_text = generate_inspection_report(defect_list, gemini_model)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Annotated Defects")
                st.image(annotated_image, caption="AI visual inspection results. Defects are marked in red.")
            
            with col2:
                st.subheader("AI Manager's Report")
                st.markdown(report_text)

