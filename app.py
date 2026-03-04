# ==============================================================================
# AI Inspection Supporter - Main Application
#
# What this app does:
# 1) Detects PCB defects using a YOLOv8 model (weights hosted on Hugging Face).
# 2) Generates a concise QC report using Google Gemini (Generative AI).
#
# Stability choices:
# - Avoid OpenCV (cv2) to prevent missing-system-library errors on Streamlit Cloud.
# - Use PIL to draw bounding boxes and labels.
#
# Author: HU KAIXIAO
# ==============================================================================

from __future__ import annotations

import os
from typing import Optional, Any, List, Dict, Tuple

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
    layout="wide",
)


# -----------------------------
# API key loading
# -----------------------------
def get_google_api_key() -> Optional[str]:
    """
    Retrieve Google API key from:
    1) Streamlit Secrets (recommended for deployment)
    2) Local .env file (for local development)
    """
    if "GOOGLE_API_KEY" in st.secrets:
        return str(st.secrets["GOOGLE_API_KEY"])

    load_dotenv()
    return os.getenv("GOOGLE_API_KEY")


# -----------------------------
# YOLO model loading
# -----------------------------
@st.cache_resource
def load_yolo_model() -> Optional[YOLO]:
    """
    Download the YOLOv8 weights from Hugging Face and load them locally.
    Uses Hugging Face caching automatically.
    """
    model_id = "keremberke/yolov8s-pcb-defect-segmentation"
    filename = "best.pt"

    try:
        print("[INFO] Loading YOLOv8 PCB Defect Model...")
        local_path = hf_hub_download(repo_id=model_id, filename=filename)
        print(f"[INFO] Weights downloaded/cached at: {local_path}")

        model = YOLO(local_path)
        print("[INFO] YOLO model loaded successfully.")
        return model
    except Exception as e:
        print(f"[ERROR] YOLO loading failed: {e}")
        st.error(f"Failed to download or load YOLO model: {e}")
        return None


# -----------------------------
# Gemini model loading (with fallback)
# -----------------------------
@st.cache_resource
def load_gemini_model() -> Optional[Any]:
    """
    Configure Gemini and return a usable GenerativeModel instance.

    Includes fallback logic to avoid 404 issues when certain model names
    are unavailable under a given API/SDK version.
    """
    print("[INFO] Configuring Gemini model...")

    api_key = get_google_api_key()
    if not api_key:
        st.error("Google API Key not found. Set GOOGLE_API_KEY in Streamlit Secrets or local .env.")
        return None

    genai.configure(api_key=api_key)

    preferred_models = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
    ]

    # Try preferred models first
    for name in preferred_models:
        try:
            model = genai.GenerativeModel(name)
            _ = model.generate_content("ping")  # quick health check
            print(f"[INFO] Gemini model OK: {name}")
            return model
        except Exception as e:
            print(f"[WARN] Gemini model not usable: {name} -> {e}")

    # Fallback: find any model that supports generateContent
    try:
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                discovered_name = str(m.name).replace("models/", "")
                model = genai.GenerativeModel(discovered_name)
                _ = model.generate_content("ping")
                print(f"[INFO] Gemini fallback model OK: {discovered_name}")
                return model
    except Exception as e:
        print(f"[ERROR] Gemini fallback failed: {e}")
        st.error(f"Failed to find a usable Gemini model: {e}")
        return None

    st.error("No usable Gemini model found (generateContent not supported).")
    return None


# -----------------------------
# Drawing utilities (PIL)
# -----------------------------
def load_font() -> ImageFont.ImageFont:
    """
    Load a font for label rendering.
    Falls back to a default bitmap font if TrueType fonts are unavailable.
    """
    try:
        # Many Linux environments have DejaVuSans available.
        return ImageFont.truetype("DejaVuSans.ttf", size=14)
    except Exception:
        return ImageFont.load_default()


def draw_boxes_pil(
    image_rgb: Image.Image,
    boxes_xyxy: List[Tuple[int, int, int, int]],
    labels: List[str],
) -> Image.Image:
    """
    Draw bounding boxes and labels on an RGB PIL image.
    """
    img = image_rgb.copy()
    draw = ImageDraw.Draw(img)
    font = load_font()

    for (x1, y1, x2, y2), label in zip(boxes_xyxy, labels):
        # Red rectangle
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

        # Label background
        text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
        pad = 4
        bg_x1 = x1
        bg_y1 = max(0, y1 - text_h - pad * 2)
        bg_x2 = x1 + text_w + pad * 2
        bg_y2 = bg_y1 + text_h + pad * 2

        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(255, 0, 0))
        draw.text((bg_x1 + pad, bg_y1 + pad), label, fill=(255, 255, 255), font=font)

    return img


# -----------------------------
# Core: defect detection
# -----------------------------
def run_defect_detection(
    image_pil: Image.Image,
    yolo_model: YOLO,
    conf: float,
) -> Tuple[Image.Image, List[Dict[str, str]]]:
    """
    Run YOLO detection on the input image.

    Parameters:
    - conf: confidence threshold. Lower => more sensitive (more boxes).
            Higher => more conservative (fewer boxes).

    Returns:
    - annotated PIL image (RGB) for display
    - defect_list: list of dicts with 'type' and 'confidence'
    """
    print(f"[INFO] Running defect detection (conf={conf})...")

    rgb_pil = image_pil.convert("RGB")
    rgb_np = np.array(rgb_pil)

    results = yolo_model.predict(rgb_np, conf=conf)

    defect_list: List[Dict[str, str]] = []
    drawn_boxes: List[Tuple[int, int, int, int]] = []
    labels: List[str] = []

    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        class_names = results[0].names  # dict: class_index -> class_name

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_index = int(box.cls[0])
            cls_name = class_names.get(cls_index, "Unknown")
            confidence = float(box.conf[0])

            defect_list.append(
                {"type": cls_name, "confidence": f"{confidence * 100:.2f}%"}
            )
            drawn_boxes.append((x1, y1, x2, y2))
            labels.append(f"{cls_name}: {confidence * 100:.1f}%")

    annotated = draw_boxes_pil(rgb_pil, drawn_boxes, labels)
    print(f"[INFO] Detection complete. Found {len(defect_list)} defects.")
    return annotated, defect_list


# -----------------------------
# Core: report generation
# -----------------------------
def generate_inspection_report(defect_list: List[Dict[str, str]], gemini_model: Any) -> str:
    """
    Generate a concise QC report using Gemini based on detected defects.
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
    st.sidebar.info(
        "This application demonstrates an 'Image-to-Report' pipeline.\n\n"
        "1) A **YOLOv8** model detects visual defects.\n\n"
        "2) **Gemini** generates a formal QC report from the detections."
    )

    st.divider()
    st.subheader("Detection Controls")
    st.caption(
        "Lower the confidence value to make the AI more sensitive (it may highlight more areas). "
        "Increase it to be looser and reduce false alarms."
    )

    conf_threshold = st.slider(
        "AI sensitivity (lower = more careful, more highlights)",
        min_value=0.01,
        max_value=0.50,
        value=0.25,
        step=0.01,
    )
    st.caption("Suggested range: 0.02–0.20 (start from 0.25).")


# Load models (cached)
yolo_model = load_yolo_model()
gemini_model = load_gemini_model()

uploaded_file = st.file_uploader("Choose a PCB image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.header("Inspection Results")

    if yolo_model is None or gemini_model is None:
        st.error("One of the AI models failed to load. Check logs and API key configuration.")
    else:
        with st.spinner("Inspecting image with YOLOv8..."):
            annotated_image, defect_list = run_defect_detection(
                image_pil=image,
                yolo_model=yolo_model,
                conf=conf_threshold,
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
