# 🚗 Number Plate Recognition System
## 🎯 Project Overview
This project implements a robust Indian Number Plate Recognition System leveraging YOLOv5 for detection and EasyOCR for text recognition.
## 🔍 Pipeline Breakdown
1️⃣ **Dataset Preparation** <br>
- **Dataset:** [Indian Number Plates Dataset] (https://www.kaggle.com/datasets/dataclusterlabs/indian-number-plates-dataset/code)<br>
- **Annotations:** Converted from PASCAL VOC XML ➡️ YOLO TXT format (only number_plate class)
- **Normalization:** Bounding boxes normalized relative to image size
