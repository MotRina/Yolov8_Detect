from ultralytics import YOLO

# Load a model
model = YOLO('./best.pt')  # load an official model

# Predict with the model
results = model('./01.jpg')  # predict on an image