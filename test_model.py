# test_model.py
from ultralytics import YOLO
from PIL import Image

# Load your model
model = YOLO("models/corrosion_model.pt")

# Load a corrosion image
image = Image.open("uploads/test_corrosion.jpg")

# Run inference
results = model(image, imgsz=640, conf=0.05)
r = results[0]

# Print results
print("Boxes:", len(r.boxes))
if len(r.boxes) > 0:
    for i, box in enumerate(r.boxes):
        cls_id = int(box.cls.item())
        label = model.model.names[cls_id]
        conf = box.conf.item()
        print(f"Detection {i+1}: {label} (Conf: {conf:.3f})")
else:
    print("❌ No corrosion detected")

# Show image with boxes
r.show()