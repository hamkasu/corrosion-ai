from ultralytics import YOLO

# Load your model
model = YOLO("models/corrosion_model.pt")

# Print model task
print("Model task:", model.model.task)  # Should say 'segment'