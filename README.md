AI Inspection Supporter (PCB Defect Detection App)

Live Demo: https://ai-inspection-supporter-bgbwp9ytj3c57rmfclzakt.streamlit.app/

AI Inspection Supporter is a Streamlit web application that performs PCB visual defect detection using a YOLOv8 model and generates a concise QC inspection report using Google Gemini.

Features

Image upload → detection → report in one flow

YOLOv8 PCB defect detection (weights downloaded from Hugging Face)

Annotated results: bounding boxes + class labels on the uploaded image

AI Manager Report: Gemini generates a structured QC report from detections

User-friendly detection control: adjust sensitivity with a single confidence slider

Cloud-stable implementation: avoids OpenCV (cv2) and uses PIL for drawing

How It Works

Upload a PCB image (.jpg, .jpeg, .png)

The app runs YOLOv8 inference and extracts detected defects (class + confidence)

The app draws bounding boxes and labels on the image (PIL)

Gemini generates a QC report with exactly three sections:

Inspection Summary

Detected Defects

Recommended Action

Model & Defect Classes

YOLO weights (Hugging Face):

keremberke/yolov8s-pcb-defect-segmentation (file: best.pt)

Typical defect classes include (depending on the model’s training):

Dry_joint

Incorrect_installation

PCB_damage

Short_circuit

Requirements

Python 3.11+ recommended

A Google Gemini API key (use .env locally or Streamlit Secrets in the cloud)

Quick Start (Run Locally)
1) Clone this repository
git clone https://github.com/4322565133-boop/ai-inspection-supporter.git
cd ai-inspection-supporter
2) Create and activate a virtual environment

macOS / Linux:

python -m venv .venv
source .venv/bin/activate

Windows (PowerShell):

python -m venv .venv
.venv\Scripts\Activate.ps1
3) Install dependencies
pip install -r requirements.txt
4) Configure your API key (local .env)

Create a .env file in the project root:

echo 'GOOGLE_API_KEY="YOUR_KEY_HERE"' > .env

Important: Never commit .env to GitHub. Your API key must stay private.

5) Run the Streamlit app
streamlit run app.py

Then open the local URL shown in the terminal (usually http://localhost:8501).

Deploy on Streamlit Community Cloud

Go to Streamlit Cloud and click Deploy an app

Select:

Repository: 4322565133-boop/ai-inspection-supporter

Branch: main

Main file path: app.py

Open Advanced settings → Secrets and add:

GOOGLE_API_KEY="YOUR_KEY_HERE"

Click Deploy (or reboot/rebuild if already deployed)

Security Notes

Do not commit API keys to GitHub

Use Streamlit Secrets for deployment

Keep .env for local development only

Project Structure
.
├── app.py               # Streamlit application entry point
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation (this file)
└── .gitignore           # Ignore .env, caches, etc.
Troubleshooting
Too many boxes / too many false positives

Increase the confidence slider value to make the detector more conservative (fewer highlights).

Missed defects

Lower the confidence slider value to make the detector more sensitive (more highlights).

Gemini report not generated

Confirm GOOGLE_API_KEY is correctly set (Secrets or .env)

The app includes fallback logic to select an available Gemini model automatically

License

Add your preferred license here (MIT / Apache-2.0 / etc.).
If you don’t have one yet, create it on GitHub via: Add file → Create new file → LICENSE.
