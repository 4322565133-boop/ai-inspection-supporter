PCB Defect Detection App

Try the live app: https://ai-inspection-supporter-epoi6qvwcjc26csg2bhsjr.streamlit.app

This is an advanced web application built with Streamlit, YOLO models, and the Gemini API to automatically detect defects in PCB (Printed Circuit Board) images and generate comprehensive analysis reports.

🌟 Project Overview

This application aims to automate and enhance the quality control process in PCB manufacturing. Users can upload an image of a PCB, and the application will:

Process it through a trained YOLO model to detect and mark potential defects (e.g., short circuits, open circuits, poor soldering) in real-time.

Leverage the Gemini API to analyze the detected defects and automatically generate a detailed, human-readable inspection report, assessing the overall quality and providing actionable insights.

✨ Key Features

Image Upload: Allows users to upload PCB images in formats like .jpg and .png through the browser.

AI-Powered Defect Detection: Uses an advanced YOLO (You Only Look Once) object detection model for high-speed, accurate defect identification.

Result Visualization: Utilizes OpenCV to draw bounding boxes on the original image, clearly highlighting the detected defects and their categories.

Generative AI Reporting: Integrates with the Gemini API to summarize inspection findings, assess the severity of defects, and generate a comprehensive quality assessment.

Automated Report Generation: Provides a clear, downloadable report detailing all detected defects, their types, locations, and an overall quality summary provided by the AI.

Web Interface: Built on Streamlit for a simple and user-friendly interactive experience.

💻 Tech Stack

Web Framework: Streamlit

AI / Object Detection: YOLO (e.g., YOLOv5 or YOLOv8)

Generative AI: Google Gemini API (via generativelanguage client)

Computer Vision: OpenCV (cv2)

Core Language: Python 3

