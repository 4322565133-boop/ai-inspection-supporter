# AI Inspection Supporter (PCB Defect Detection App)

**Live Demo:** https://ai-inspection-supporter-bgbwp9ytj3c57rmfclzakt.streamlit.app/

AI Inspection Supporter is a Streamlit web application that performs **PCB visual defect detection** using a YOLOv8 model and generates a **concise QC inspection report** using Google Gemini.

---

## Features

- **Image upload → detection → report** in one flow
- **YOLOv8 PCB defect detection** (weights downloaded from Hugging Face)
- **Annotated results:** bounding boxes + class labels on the uploaded image
- **AI Manager Report:** Gemini generates a structured QC report from detections
- **User-friendly detection control:** adjust sensitivity with a single confidence slider
- **Cloud-stable implementation:** avoids OpenCV (`cv2`) and uses PIL for drawing

---

## How It Works

1. Upload a PCB image (`.jpg`, `.jpeg`, `.png`)
2. The app runs YOLOv8 inference and extracts detected defects (class + confidence)
3. The app draws bounding boxes and labels on the image (PIL)
4. Gemini generates a QC report with exactly three sections:
   - **Inspection Summary**
   - **Detected Defects**
   - **Recommended Action**

---

## Model & Defect Classes

**YOLO weights (Hugging Face):**
- `keremberke/yolov8s-pcb-defect-segmentation` (file: `best.pt`)

Typical defect classes include (depending on the model’s training):
- `Dry_joint`
- `Incorrect_installation`
- `PCB_damage`
- `Short_circuit`

---

## Requirements

- Python **3.11+** recommended
- A Google Gemini API key (use `.env` locally or Streamlit Secrets in the cloud)

---

## How to Use This Repository

### Run Locally

1. Create a Python environment (recommended: virtual environment)
2. Install dependencies from `requirements.txt`
3. Set `GOOGLE_API_KEY` in a local `.env` file (do not commit this file)
4. Start the app with Streamlit and open the local URL shown in your terminal

### Deploy on Streamlit Community Cloud

1. Connect this GitHub repository in Streamlit Cloud
2. Set the main file to `app.py`
3. Add `GOOGLE_API_KEY` to Streamlit **Secrets**
4. Deploy (or reboot/rebuild if already deployed)

---

## Security Notes

- Do **not** commit API keys to GitHub
- Use Streamlit **Secrets** for deployment
- Keep `.env` for local development only

---

## Project Structure

- `app.py` — Streamlit application entry point
- `requirements.txt` — Python dependencies
- `README.md` — Project documentation
- `.gitignore` — ignores `.env`, caches, etc.

---

## Troubleshooting

### Too many boxes / too many false positives
Increase the confidence slider value to make the detector more conservative (fewer highlights).

### Missed defects
Lower the confidence slider value to make the detector more sensitive (more highlights).

### Gemini report not generated
- Confirm `GOOGLE_API_KEY` is correctly set (Secrets or `.env`)
- The app includes fallback logic to select an available Gemini model automatically

---

## License

Add your preferred license here (MIT / Apache-2.0 / etc.).  
If you don’t have one yet, create it in GitHub via **Add file → Create new file → LICENSE**.
