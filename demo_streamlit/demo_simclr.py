import cv2
import os
import streamlit as st
from ultralytics import YOLO
from keras.models import load_model
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array

# Directory Paths
TMP_DIR_PATH = "PATH"
MODEL_DIR_PATH = "PATH"
GENDER_MODEL_PATH = "PATH"

# Create directories if not exist
if not os.path.exists(TMP_DIR_PATH):
    os.makedirs(TMP_DIR_PATH)

# Streamlit Title
st.title("Boar Count Master")

# File upload
img_file = st.file_uploader("Upload an image or video", type=["png", "jpg", "jpeg", "mp4"])

# Load YOLO and Gender Classification Models
yolo_model = YOLO(os.path.join(MODEL_DIR_PATH, "best.pt"))
gender_model = load_model(GENDER_MODEL_PATH)
yolo_names = yolo_model.names

def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    return image

def predict_gender(image):
    processed_image = preprocess_image(image)
    prediction = gender_model.predict(processed_image)
    classes = ['female', 'male']
    return classes[np.argmax(prediction)], max(prediction[0])

# frame_count の初期化
frame_count = 0
# frame_interval の設定（例: 5フレームごとに処理）
frame_interval = 5

if img_file is not None:
    file_path = os.path.join(TMP_DIR_PATH, img_file.name)
    with open(file_path, "wb") as f:
        f.write(img_file.getvalue())

    # Process for image and video
# ビデオファイルの処理
# ビデオファイルの処理
    if img_file.type == "video/mp4":
        cap = cv2.VideoCapture(file_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(TMP_DIR_PATH, f"analysis_{img_file.name}"), fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % frame_interval == 0:
                # YOLOモデルで物体検出
                results = yolo_model.predict(frame, save=False)
                for box in results[0].boxes:
                    xmin, ymin, xmax, ymax = map(int, box.xyxy.cpu().numpy()[0])
                    cropped_img = Image.fromarray(frame[ymin:ymax, xmin:xmax])
                    gender, confidence = predict_gender(cropped_img)

                    # 性別に基づいて矩形とテキストを描画
                    text = f"{gender} ({confidence:.2f})"
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            out.write(frame)

        cap.release()
        out.release()
        st.video(os.path.join(TMP_DIR_PATH, f"analysis_{img_file.name}"))

    else:
        # Process image file
        results = yolo_model.predict(file_path, save=False)
        img = cv2.imread(file_path)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        bbox_count = 0
        female_count, male_count = 0, 0

        for box in results[0].boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy.cpu().numpy()[0])
            cropped_img = img_pil.crop((xmin, ymin, xmax, ymax))

            gender, confidence = predict_gender(cropped_img)
            if gender == "female":
                female_count += 1
            else:
                male_count += 1

            text = f"{gender} ({confidence:.2f})"
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 5)
            cv2.putText(img, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            bbox_count += 1

        cv2.putText(img, f"Boar Count: {bbox_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f"Female Boar Count: {female_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f"Male Boar Count: {male_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        analysis_img_path = os.path.join(TMP_DIR_PATH, f"analysis_{img_file.name}")
        cv2.imwrite(analysis_img_path, img)
        st.image(analysis_img_path, use_column_width=True)
