# AI Inspection Supporter (PCB Defect Detection App)

**Live Demo:** https://ai-inspection-supporter-bgbwp9ytj3c57rmfclzakt.streamlit.app/

AI Inspection Supporter is a Streamlit web app that performs **PCB visual defect detection** using a YOLOv8 model and generates a **concise QC inspection report** using Google Gemini.

---

## Features

- **Image upload → detection → report** in one flow  
- **YOLOv8 PCB defect detection** (weights downloaded from Hugging Face)
- **Annotated results**: bounding boxes + class labels on the uploaded image
- **AI Manager Report**: Gemini generates a structured QC report from detections
- **User-friendly detection control**: adjust sensitivity with a single confidence slider
- **Cloud-stable implementation**: avoids OpenCV (`cv2`) and uses PIL for drawing

---

## How It Works

1. Upload a PCB image (`.jpg`, `.jpeg`, `.png`)
2. The app runs YOLOv8 inference and extracts detected defects (class + confidence)
3. The app draws bounding boxes and labels on the image
4. Gemini generates a QC report with:
   - **Inspection Summary**
   - **Detected Defects**
   - **Recommended Action**

---

## Model & Defect Classes

**YOLO weights (Hugging Face):**
- `keremberke/yolov8s-pcb-defect-segmentation` (file: `best.pt`)

Common defect classes include (depending on model training):
- `Dry_joint`
- `Incorrect_installation`
- `PCB_damage`
- `Short_circuit`

---

## Requirements

- Python **3.11+** recommended
- Google Gemini API key (use `.env` locally or Streamlit Secrets in cloud)

---

## Local Setup

### 1) Clone the repo
```bash
git clone https://github.com/4322565133-boop/ai-inspection-supporter.git
cd ai-inspection-supporter
