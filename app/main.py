# app/main.py

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from PIL import Image
import os
import asyncio
from uuid import uuid4
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import io
import pandas as pd
from fastapi.responses import StreamingResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create required directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("annotated", exist_ok=True)

# Create thread pool
thread_pool = ThreadPoolExecutor(max_workers=1)

# Import local modules
import app.database as database
import app.schemas as schemas
import app.model as model

# Create FastAPI app
app = FastAPI(title="Corrosion Detection API")

# Serve uploaded and annotated images
app.mount("/images", StaticFiles(directory="uploads"), name="images")
app.mount("/annotated", StaticFiles(directory="annotated"), name="annotated")

# Create database tables
database.Base.metadata.create_all(bind=database.engine)

# Dependency to get DB session
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def read_root():
    return {"message": "Corrosion Detection API is running!"}

@app.post("/upload", response_model=schemas.InspectionResponse)
async def upload_image(
    file: UploadFile = File(...),
    project_id: str = Form("default"),
    project_description: str = Form(None),
    db: Session = Depends(get_db)
):
    logger.info("Upload request received")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    def run_prediction():
        return model.predict_with_boxes(image)

    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(thread_pool, run_prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    filename = f"{uuid4()}_{file.filename}"
    annotated_filename = f"annotated_{filename}"

    original_path = os.path.join("uploads", filename)
    try:
        image.save(original_path, "JPEG", quality=95)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save original image")

    annotated_path = os.path.join("annotated", annotated_filename)
    try:
        save_img = results["annotated_image"].convert("RGB")
        save_img.save(annotated_path, "JPEG", quality=95)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save annotated image")

    inspection = database.Inspection(
        image_path=f"/images/{filename}",
        annotated_path=f"/annotated/{annotated_filename}",
        prediction=results["label"],
        confidence=results["confidence"],
        image_metadata={"filename": file.filename, "content_type": file.content_type},
        project_id=project_id,
        project_description=project_description
    )
    db.add(inspection)
    try:
        db.commit()
        db.refresh(inspection)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Database save failed")

    return inspection

# ✅ New: Get latest inspection
@app.get("/inspections/latest", response_model=schemas.InspectionResponse)
def get_latest_inspection(db: Session = Depends(get_db)):
    latest = db.query(database.Inspection).order_by(database.Inspection.id.desc()).first()
    if not latest:
        raise HTTPException(status_code=404, detail="No inspections found")
    return latest

# ✅ Get all inspections
@app.get("/inspections", response_model=list[schemas.InspectionResponse])
def get_inspections(db: Session = Depends(get_db)):
    return db.query(database.Inspection).all()

# ✅ Export to CSV
@app.get("/inspections/export")
def export_to_csv(db: Session = Depends(get_db)):
    inspections = db.query(database.Inspection).all()
    data = [{
        "id": i.id,
        "image_path": i.image_path,
        "annotated_path": i.annotated_path,
        "prediction": i.prediction,
        "confidence": i.confidence,
        "uploaded_at": i.uploaded_at,
        "project_id": i.project_id,
        "project_description": i.project_description
    } for i in inspections]
    
    df = pd.DataFrame(data)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=corrosion_inspections.csv"
    return response