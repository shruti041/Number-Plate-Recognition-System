import streamlit as st
import cv2
import torch
import easyocr
import numpy as np
from PIL import Image
import tempfile

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\INSPIRON\\Desktop\\AI&ML project\\Number Plate Recognition System\\yolov5\\runs\\train\\exp3\\weights\\best.pt')

model = load_model()
model.conf = 0.4

# Load OCR reader
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

reader = load_reader()

# Detection + OCR
def detect_and_recognize(img):
    results = model(img)
    plates = []

    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        plate_crop = img[y1:y2, x1:x2]

        ocr_result = reader.readtext(plate_crop)
        text = ''
        if ocr_result:
            text = ocr_result[0][-2]

        plates.append((text, (x1, y1, x2, y2)))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img, plates

# Streamlit UI
st.title("Indian Number Plate Recognition")
st.write("Upload an image or video to detect and recognize number plates.")

option = st.radio("Select input type", ["Image", "Video"])

if option == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        result_img, plates = detect_and_recognize(img)
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Detected Image", use_column_width=True)
        st.write("Detected Plates:", plates)

elif option == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result_frame, plates = detect_and_recognize(frame)
            stframe.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB), channels="RGB")
        cap.release()
