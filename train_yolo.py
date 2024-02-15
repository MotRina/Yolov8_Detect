from ultralytics import YOLO
model = YOLO("yolov8l.pt")  # load a pretrained model (recommended for training)
model.train(data="./data.yaml", epochs=300)  # train the model
metrics = model.val()  # evaluate model performance on the validation set

# !yolo detect train data=data.yaml model=yolov8l.pt epochs=300 device=0 batch=96 name=boar optimizer=SGD
