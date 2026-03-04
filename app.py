# ==============================================================================
# AI Inspection Supporter - Main Application
#
# What this app does:
# 1) Detects PCB defects using a YOLOv8 model (weights hosted on Hugging Face).
# 2) Generates a concise QC report using Google Gemini (Generative AI).
#
# Key features:
# - Uses stable Gemini model names + robust fallback via list_models().
# - Uses Streamlit Secrets or local .env for API key.
# - Correct color-space handling (OpenCV draws in BGR, Streamlit displays RGB).
# - Adjustable YOLO thresholds (confidence + IoU) to reduce false positives.
#
# Author: HU KAIXIAO
# ==============================================================================

import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from dotenv import load_dotenv

import google.generativeai as genai
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="AI Inspection Supporter",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------
# Utilities: key loading
# -----------------------------
def get_google_api_key() -> str | None:
    """
    Retrieve the Google API key from (1) Streamlit secrets, or (2) local .env file.
    """
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]

    load_dotenv()
    return os.getenv("GOOGLE_API_KEY")


# -----------------------------
# Model loading: YOLO
# -----------------------------
@st.cache_resource
def load_yolo_model():
    """
    Load the YOLOv8 model for PCB defect detection.
    Downloads the weights explicitly from Hugging Face, then loads from local path.
    """
    print("[INFO] Loading YOLOv8 PCB Defect Model...")

    model_id = "keremberke/yolov8s-pcb-defect-segmentation"
    filename = "best.pt"

    try:
        print(f"[INFO] Downloading weights '{filename}' from '{model_id}'...")
        local_model_path = hf_hub_download(repo_id=model_id, filename=filename)
        print(f"[INFO] Weights downloaded/cached at: {local_model_path}")

        model = YOLO(local_model_path)
        print("[INFO] YOLO model loaded successfully.")
        return model

    except Exception as e:
        print(f"[ERROR] YOLO loading failed: {e}")
        st.error(f"Failed to download or load YOLO model: {e}")
        return None


# -----------------------------
# Model loading: Gemini
# -----------------------------
@st.cache_resource
def configure_gemini_model():
    """
    Configure Gemini and return a usable GenerativeModel instance.
    Includes fallback logic to avoid 404 model issues.
    """
    print("[INFO] Configuring Gemini model...")

    api_key = get_google_api_key()
    if not api_key:
        st.error("Google API Key not found. Set it in Streamlit Secrets or local .env.")
        return None

    genai.configure(api_key=api_key)

    preferred_models = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
    ]

    for name in preferred_models:
        try:
            model = genai.GenerativeModel(name)
            _ = model.generate_content("ping")
            print(f"[INFO] Gemini model OK: {name}")
            return model
        except Exception as e:
            print(f"[WARN] Gemini model not usable: {name} -> {e}")

    try:
        for m in genai.list_models():
            supported = getattr(m, "supported_generation_methods", [])
            if "generateContent" in supported:
                discovered_name = m.name.replace("models/", "")
                model = genai.GenerativeModel(discovered_name)
                _ = model.generate_content("ping")
                print(f"[INFO] Gemini fallback model OK: {discovered_name}")
                return model
    except Exception as e:
        print(f"[ERROR] Gemini list_models fallback failed: {e}")
        st.error(f"Failed to find a usable Gemini model: {e}")
        return None

    st.error("No usable Gemini model found (generateContent not supported).")
    return None


# -----------------------------
# Core: defect detection
# -----------------------------
def run_defect_detection(
    image_pil: Image.Image,
    yolo_model: YOLO,
    conf: float = 0.25,
    iou: float = 0.50
):
    """
    Run YOLO detection on the input image and return:
    - annotated RGB image (numpy array) for display
    - defect_list: list of dicts with 'type' and 'confidence'

    Notes:
    - conf: confidence threshold (higher => fewer boxes)
    - iou: IoU threshold for NMS (lower => fewer overlapping duplicate boxes)
    """
    print(f"[INFO] Running defect detection (conf={conf}, iou={iou})...")

    rgb = np.array(image_pil.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    annotated_bgr = bgr.copy()

    # Pass both confidence and IoU thresholds to reduce noisy detections
    results = yolo_model.predict(rgb, conf=conf, iou=iou)

    defect_list = []

    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        class_names = results[0].names

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_index = int(box.cls[0])
            cls_name = class_names.get(cls_index, "Unknown")
            confidence = float(box.conf[0])

            defect_list.append({
                "type": cls_name,
                "confidence": f"{confidence * 100:.2f}%"
            })

            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

            label = f"{cls_name}: {confidence * 100:.1f}%"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_bgr, (x1, y1 - h - 6), (x1 + w, y1), (0, 0, 255), -1)
            cv2.putText(
                annotated_bgr, label, (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

    print(f"[INFO] Detection complete. Found {len(defect_list)} defects.")

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, defect_list


# -----------------------------
# Core: report generation
# -----------------------------
def generate_inspection_report(defect_list: list[dict], gemini_model) -> str:
    """
    Generate a professional QC report using Gemini based on detected defects.
    """
    print("[INFO] Generating QC report...")

    if not defect_list:
        return "PASS: No defects detected during visual inspection."

    defect_string = "\n".join(
        [f"- {d['type']} (Confidence: {d['confidence']})" for d in defect_list]
    )

    prompt = f"""
You are an expert AI Quality Control Manager for an electronics production line.
An AOI tool scanned a PCB and found the following potential defects:

{defect_string}

Defect Guide:
- 'Dry_joint': A poor solder connection (cold solder / insufficient solder).
- 'Incorrect_installation': A component is installed incorrectly.
- 'PCB_damage': Physical damage to the PCB.
- 'Short_circuit': An improper electrical connection between two points.

Write a concise, professional inspection report with exactly three sections:

1. Inspection Summary: One sentence overview.
2. Detected Defects: Bullet list of the items found (include confidence).
3. Recommended Action: Clear and actionable next steps for QC/repair/rework.
"""

    try:
        response = gemini_model.generate_content(prompt)
        print("[INFO] Report generated successfully.")
        return response.text
    except Exception as e:
        print(f"[ERROR] Gemini generate_content failed: {e}")
        return f"Error generating report: {e}"


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🤖 AI Inspection Supporter")
st.markdown("##### Upload a PCB image to run an AI-powered defect inspection.")

with st.sidebar:
    st.header("About This App")
    st.info(
        "This application demonstrates an 'Image-to-Report' pipeline.\n\n"
        "1) A **YOLOv8** model detects visual defects.\n\n"
        "2) **Gemini** generates a formal QC report from the detections."
    )

    st.divider()
    st.subheader("Detection Settings")

    # Adjustable thresholds to reduce noisy boxes
    conf_threshold = st.slider(
        "Confidence threshold (higher = fewer boxes)",
        min_value=0.01,
        max_value=0.80,
        value=0.25,
        step=0.01
    )

    iou_threshold = st.slider(
        "IoU threshold for NMS (lower = fewer duplicate boxes)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    st.caption("Tip: Try conf=0.25~0.40 and IoU=0.45~0.60 for cleaner results.")


# Load models (cached)
yolo_model = load_yolo_model()
gemini_model = configure_gemini_model()

uploaded_file = st.file_uploader("Choose a PCB image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.header("Inspection Results")

    if yolo_model is None or gemini_model is None:
        st.error("One of the AI models failed to load. Check logs and API key configuration.")
    else:
        with st.spinner("Inspecting image with YOLOv8..."):
            annotated_image, defect_list = run_defect_detection(
                image,
                yolo_model,
                conf=conf_threshold,
                iou=iou_threshold
            )

        if not defect_list:
            st.success("✅ Inspection PASSED")
            st.image(image, caption="Original Uploaded Image", use_container_width=True)
        else:
            st.error(f"❌ Inspection FAILED: {len(defect_list)} potential defects detected.")

            with st.spinner("Generating QC report with Gemini..."):
                report_text = generate_inspection_report(defect_list, gemini_model)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Annotated Defects")
                st.image(annotated_image, caption="Defects are marked in red.", use_container_width=True)

            with col2:
                st.subheader("AI Manager's Report")
                st.markdown(report_text)