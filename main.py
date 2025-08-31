# app/main.py

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from PIL import Image
import os
from uuid import uuid4
from datetime import datetime

# Create required directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("annotated", exist_ok=True)

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

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "Corrosion Detection API is running!",
        "docs": "Go to /docs for API interface"
    }

# Upload endpoint
@app.post("/upload", response_model=schemas.InspectionResponse)
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Open image
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Get prediction with mask
    try:
        results = model.predict_with_mask(image)
        prediction = results["label"]
        confidence = results["confidence"]
        annotated_image = results["annotated_image"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Generate unique filenames
    filename = f"{uuid4()}_{file.filename}"
    annotated_filename = f"annotated_{filename}"

    # Save original image
    original_path = os.path.join("uploads", filename)
    try:
        image.save(original_path)
        print(f"✅ Original image saved: {original_path}")
    except Exception as e:
        print(f"❌ Failed to save original: {e}")
        raise HTTPException(status_code=500, detail="Could not save original image")

    # Save annotated image (with red mask)
    annotated_path = os.path.join("annotated", annotated_filename)
    try:
        # Ensure RGB mode for JPEG
        save_img = annotated_image.convert("RGB") if annotated_image.mode != "RGB" else annotated_image
        save_img.save(annotated_path, "JPEG", quality=95)
        print(f"✅ Annotated image saved: {annotated_path}")
    except Exception as e:
        print(f"❌ Failed to save annotated image: {e}")
        raise HTTPException(status_code=500, detail=f"Could not save annotated image: {str(e)}")

    # Create database record
    inspection = database.Inspection(
        image_path=f"/images/{filename}",
        annotated_path=f"/annotated/{annotated_filename}",
        prediction=prediction,
        confidence=confidence,
        image_metadata={"filename": file.filename, "content_type": file.content_type}
    )
    db.add(inspection)
    try:
        db.commit()
        db.refresh(inspection)
        print(f"✅ Saved to database: ID {inspection.id}")
    except Exception as e:
        db.rollback()
        print(f"❌ Database error: {e}")
        raise HTTPException(status_code=500, detail="Database save failed")

    return inspection