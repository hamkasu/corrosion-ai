# train_roboflow_segmentation.py
import torch
from ultralytics import YOLO
import os

# Settings
MODEL_NAME = "yolov8s-seg.pt"  # or yolov8n-seg.pt (smaller)
DATA_YAML = "roboflow_dataset/data.yaml"  # Roboflow gives this
EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 640
PROJECT = "corrosion_detection"
NAME = "yolov8_segmentation_run"

# Create models directory
os.makedirs("models", exist_ok=True)

# Load pre-trained segmentation model
model = YOLO(MODEL_NAME)

# Train
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    project=PROJECT,
    name=NAME,
    save=True,
    exist_ok=True,
    device=0 if torch.cuda.is_available() else "cpu"
)

# Export ONNX (for deployment)
model.export(format="onnx")

print("✅ Training complete!")
print(f"Model saved in {PROJECT}/{NAME}/")
print("Best weights:", f"{PROJECT}/{NAME}/weights/best.pt")