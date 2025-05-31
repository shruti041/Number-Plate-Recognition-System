import cv2
import torch
import easyocr
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\INSPIRON\\Desktop\\AI&ML project\\Number Plate Recognition System\\yolov5\\runs\\train\\exp3\\weights\\best.pt')  
model.conf = 0.4  

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])  # 'en' for English

def detect_and_recognize(img):
    results = model(img)
    plates = []

    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        plate_crop = img[y1:y2, x1:x2]

        # Use EasyOCR to recognize text
        ocr_result = reader.readtext(plate_crop)
        text = ''
        if ocr_result:
            text = ocr_result[0][-2]  # Extract recognized text

        plates.append((text, (x1, y1, x2, y2)))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img, plates

def process_image(image_path):
    img = cv2.imread(image_path)
    result_img, plates = detect_and_recognize(img)
    print("Detected Plates:", plates)
    cv2.imshow("Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result_frame, plates = detect_and_recognize(frame)
        cv2.imshow('Video', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

