from ultralytics import YOLO

# Load base model
model = YOLO("yolov8n.pt")

# Train model
model.train(
    data="License Plate Detection.v1i.yolov8/data.yaml",
    epochs=20,
    imgsz=640,
    plots=False   # disables plotting (avoids crash)
)
