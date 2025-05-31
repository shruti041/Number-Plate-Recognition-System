# 🚗 Number Plate Recognition System

## 🎯 Project Overview
This project implements a robust **Indian Number Plate Recognition System** leveraging **YOLOv5** for number plate detection and **EasyOCR** for text recognition.

---

## 🔍 Pipeline Breakdown

### 1️⃣ Dataset Preparation
- **Dataset:** [Indian Number Plates Dataset](https://www.kaggle.com/datasets/dataclusterlabs/indian-number-plates-dataset/code)
- **Annotations:** Converted from PASCAL VOC XML format to YOLO TXT format (only `number_plate` class)
- **Normalization:** Bounding box coordinates normalized relative to image dimensions

---

### 2️⃣ Data Organization & Splitting
- Dataset split into:
  - 🟩 **80% Training set**
  - 🟦 **20% Validation set**

---

### 3️⃣ Model Training
- **Model:** YOLOv5 (custom)
- **Classes:** 1 (`number_plate`)
- **Config:** `data.yaml` specifying dataset paths and classes
- **Output:** Best model weights saved as `best.pt`

---

### 4️⃣ Number Plate Detection & Recognition
- YOLOv5 detects bounding boxes around number plates
- EasyOCR reads cropped number plate images to extract text
- Results are displayed with bounding boxes and recognized text overlays

---

### 5️⃣ Streamlit Web App
- Interactive UI for uploading images and videos
- Real-time detection and OCR visualization
- Runs locally

---
## 🔄 Typical Workflow

```mermaid
graph LR
    A[Input Acquisition] --> B[Detection with YOLOv5]
    B --> C[Cropping Plate Regions]
    C --> D[OCR Recognition]
    D --> E[Output: Display or Store Results]
