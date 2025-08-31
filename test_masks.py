# test_masks.py

from ultralytics import YOLO
from PIL import Image
import os

# ========================================
# Configuration
# ========================================
# Path to your trained segmentation model
model_path = "models/corrosion_model.pt"

# Path to a test image with corrosion
image_path = "uploads/test_corrosion.jpg"

# ========================================
# Step 1: Check if files exist
# ========================================
if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    print("Make sure you have placed 'corrosion_model.pt' in the 'models/' folder.")
    exit()

if not os.path.exists(image_path):
    print(f"❌ Test image not found: {image_path}")
    print("Please place a corrosion image at 'uploads/test_corrosion.jpg'")
    print("You can use any image from your Roboflow dataset.")
    exit()

# ========================================
# Step 2: Load the model
# ========================================
try:
    model = YOLO(model_path)
    print(f"✅ Loaded model: {model_path}")
    print(f"Model task: '{model.model.task}'")  # Should be 'segment'
    if model.model.task != 'segment':
        print("⚠️  WARNING: This is not a segmentation model!")
        print("   You must train with 'yolov8s-seg.pt', not 'yolov8s.pt'")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

# ========================================
# Step 3: Load the test image
# ========================================
try:
    image = Image.open(image_path)
    print(f"✅ Loaded image: {image_path} (Size: {image.size}, Mode: {image.mode})")
except Exception as e:
    print(f"❌ Failed to open image: {e}")
    exit()

# ========================================
# Step 4: Run inference
# ========================================
print("🔍 Running inference...")
results = model(image, imgsz=640, conf=0.25)
r = results[0]  # Get first result

# ========================================
# Step 5: Check detection results
# ========================================
print(f"📦 Number of detections: {len(r.boxes)}")
if len(r.boxes) > 0:
    for i, box in enumerate(r.boxes):
        cls_id = int(box.cls.item())
        confidence = box.conf.item()
        label = model.model.names[cls_id]
        print(f"  Detection {i+1}: {label} (Confidence: {confidence:.3f})")
else:
    print("  No corrosion detected in this image.")

# ========================================
# Step 6: Check for segmentation masks
# ========================================
if r.masks is not None:
    print(f"🩸 Segmentation masks detected: {len(r.masks)}")
    print(f"  Mask data shape: {r.masks.data.shape} [num_masks, height, width]")
    print("✅ SUCCESS: Your model is a segmentation model and detected corrosion areas!")
else:
    print("❌ No masks detected.")
    print("⚠️  Possible reasons:")
    print("   1. You trained with 'yolov8s.pt' instead of 'yolov8s-seg.pt'")
    print("   2. No corrosion was detected in this image")
    print("   3. The model was not saved correctly")

# ========================================
# Step 7: Show the result with bounding boxes
# ========================================
try:
    # Get annotated image (boxes and labels)
    result_array = r.plot()
    result_image = Image.fromarray(result_array)
    
    # Display the image
    result_image.show()
    print("🖼️  Displaying detection results. Check the image window.")
    print("📌 Look for bounding boxes around corrosion areas.")
    
except Exception as e:
    print(f"❌ Failed to display image: {e}")

# ========================================
# Final Notes
# ========================================
print("\n" + "="*50)
print("✅ If masks are detected, your model is working correctly.")
print("🔴 If no masks, retrain using 'yolov8s-seg.pt' in Google Colab.")
print("="*50)