# app/model.py

from PIL import Image, ImageDraw
from ultralytics import YOLO
import os
import numpy as np
import cv2

# Path to your trained model
MODEL_PATH = "models/corrosion_model.pt"

# Global model instance
model = None


def load_model():
    """
    Lazy-load the YOLOv8 model
    """
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Make sure 'corrosion_model.pt' is in the 'models/' folder."
            )
        model = YOLO(MODEL_PATH)
        print(f"✅ Loaded model from {MODEL_PATH}")
        print(f"🔧 Model task: {model.model.task}")
        print(f"🏷️  Model names: {model.model.names}")
    return model


def predict(image: Image.Image):
    """
    Returns: (prediction_label, confidence)
    """
    results = predict_with_boxes(image)
    label = results["label"]
    confidence = results["confidence"]
    return label, confidence


def predict_with_boxes(image: Image.Image):
    """
    Runs inference and returns:
    - label: "corrosion" or "no_corrosion"
    - confidence: 0.99 if any corrosion detected
    - annotated_image: PIL Image with thin, non-overlapping red dashed boxes
    """
    model = load_model()

    # Run inference
    results = model(image, imgsz=640, conf=0.05, iou=0.5)
    r = results[0]

    # Use original image
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    has_corrosion = False

    if len(r.boxes) > 0:
        # Extract boxes and class names
        boxes = []
        for box in r.boxes:
            cls_id = int(box.cls.item())
            class_name = model.model.names[cls_id]

            # Accept multiple possible class names
            if class_name.lower() in ["corrosion", "rust", "segmentation", "defect", "oxidation"]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append([x1, y1, x2, y2])

        # Convert to NumPy array for NMS
        if boxes:
            boxes = np.array(boxes)
            # Apply Non-Maximum Suppression to reduce overlap
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores=[1.0]*len(boxes), score_threshold=0.1, nms_threshold=0.3)

            # Draw only non-overlapping boxes
            for i in indices.flatten():
                x1, y1, x2, y2 = boxes[i]
                draw_dashed_rectangle(draw, (x1, y1, x2, y2), dash_length=6, gap_length=4, outline=(255, 0, 0), width=1)
                has_corrosion = True

    # Force high confidence if any corrosion is detected
    label = "corrosion" if has_corrosion else "no_corrosion"
    confidence = 0.99 if has_corrosion else 0.01

    return {
        "label": label,
        "confidence": confidence,
        "annotated_image": annotated_image
    }


def draw_dashed_rectangle(draw, xy, dash_length=5, gap_length=5, outline="black", width=1):
    """
    Draw a dashed rectangle with no fill.
    """
    x1, y1, x2, y2 = xy
    # Top line
    draw_dashed_line(draw, x1, y1, x2, y1, dash_length, gap_length, outline, width)
    # Right line
    draw_dashed_line(draw, x2, y1, x2, y2, dash_length, gap_length, outline, width)
    # Bottom line
    draw_dashed_line(draw, x2, y2, x1, y2, dash_length, gap_length, outline, width)
    # Left line
    draw_dashed_line(draw, x1, y2, x1, y1, dash_length, gap_length, outline, width)


def draw_dashed_line(draw, x1, y1, x2, y2, dash_length, gap_length, outline, width):
    """
    Draw a dashed line between two points.
    """
    dx = x2 - x1
    dy = y2 - y1
    length = (dx**2 + dy**2)**0.5
    if length == 0:
        return

    # Unit vector
    ux = dx / length
    uy = dy / length

    segment_length = dash_length + gap_length
    num_segments = int(length // segment_length)

    for j in range(num_segments):
        start = j * segment_length
        end = start + dash_length
        if end > length:
            break
        x_start = x1 + ux * start
        y_start = y1 + uy * start
        x_end = x1 + ux * end
        y_end = y1 + uy * end
        draw.line([(x_start, y_start), (x_end, y_end)], fill=outline, width=width)

    # Final dash
    last_start = num_segments * segment_length
    if last_start < length:
        x_start = x1 + ux * last_start
        y_start = y1 + uy * last_start
        draw.line([(x_start, y_start), (x2, y2)], fill=outline, width=width)