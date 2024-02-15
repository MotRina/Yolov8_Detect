import cv2
import os
import streamlit as st
from ultralytics import YOLO

TMP_DIR_PATH = "PATH"
MODEL_DIR_PATH = "PATH"
if not os.path.exists(TMP_DIR_PATH):
    os.makedirs(TMP_DIR_PATH)

st.title("Boar Count Master")

# File upload
img_file = st.file_uploader("Upload an image or video", type=["png", "jpg", "jpeg", "mp4"])

if img_file is not None:
    file_path = os.path.join(TMP_DIR_PATH, img_file.name)

    with open(file_path, "wb") as f:
        f.write(img_file.getvalue())

    # Load YOLO model
    model = YOLO(os.path.join(MODEL_DIR_PATH, "best.pt"))
    names = model.names

    if img_file.type == "video/mp4":
        # ビデオファイルの処理
        cap = cv2.VideoCapture(file_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(TMP_DIR_PATH, f"analysis_{img_file.name}"), fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % frame_interval == 0:
                # YOLOの実行
                results = model.predict(frame, save=False)
                for point in results[0].boxes.xyxy:
                    cv2.rectangle(frame,
                                (int(point[0]), int(point[1])),
                                (int(point[2]), int(point[3])),
                                (0, 0, 255),
                                thickness=5)

            out.write(frame)

        cap.release()
        out.release()
        st.video(os.path.join(TMP_DIR_PATH, f"analysis_{img_file.name}"))

    else:
        # Process image file
        results = model.predict(file_path, save=False)
        img = cv2.imread(file_path)

        # Draw bounding boxes and labels
        bbox_count = 0  # Initialize bounding box counter
        for box in results[0].boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy.cpu().numpy()[0])
            label = names[box.cls.cpu().numpy()[0]]
            conf = box.conf.cpu().numpy()[0]

            # Draw rectangle
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 5)

            # Draw label and confidence
            text = f"{label}: {conf:.2f}"
            cv2.putText(img, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            bbox_count += 1  # Increment bounding box counter

        # Display bounding box count
        cv2.putText(img, f"Boar Count: {bbox_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Save and display the image
        analysis_img_path = os.path.join(TMP_DIR_PATH, f"analysis_{img_file.name}")
        cv2.imwrite(analysis_img_path, img)
        st.image(analysis_img_path, use_column_width=True)
