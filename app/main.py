# app/main.py

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from PIL import Image
import os
import asyncio
from uuid import uuid4
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create required directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("annotated", exist_ok=True)

# Debug: Verify folders are created and writable
try:
    test_file = "annotated/.test_write"
    with open(test_file, "w") as f:
        f.write("test")
    os.remove(test_file)
    logger.info("✅ 'annotated/' folder is writable")
except Exception as e:
    logger.error(f"❌ Cannot write to 'annotated/': {e}")

# Create thread pool for background tasks
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
    return {
        "message": "Corrosion Detection API is running!",
        "docs": "/docs"
    }

@app.post("/upload", response_model=schemas.InspectionResponse)
async def upload_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    logger.info("📤 Upload request received")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Open image
    try:
        image = Image.open(file.file).convert("RGB")
        logger.info(f"🖼️  Image opened: {image.size} ({image.mode})")
    except Exception as e:
        logger.error(f"❌ Invalid image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Run prediction in background thread
    def run_prediction():
        return model.predict_with_boxes(image)

    try:
        loop = asyncio.get_event_loop()
        logger.info("🔍 Starting prediction in background thread...")
        results = await loop.run_in_executor(thread_pool, run_prediction)
        logger.info("✅ Prediction completed")
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Generate filenames
    filename = f"{uuid4()}_{file.filename}"
    annotated_filename = f"annotated_{filename}"

    # Save original image
    original_path = os.path.join("uploads", filename)
    try:
        image.save(original_path, "JPEG", quality=95)
        logger.info(f"✅ Original saved: {original_path}")
    except Exception as e:
        logger.error(f"❌ Save failed (original): {e}")
        raise HTTPException(status_code=500, detail="Failed to save original image")

    # Save annotated image
    annotated_path = os.path.join("annotated", annotated_filename)
    
    # Debug: Print full path and image info
    logger.info(f"📁 Saving annotated image to: {annotated_path}")
    logger.info(f"📄 Full path: {os.path.abspath(annotated_path)}")
    
    try:
        save_img = results["annotated_image"]
        if save_img.mode not in ("RGB", "RGBA"):
            logger.info(f"🔄 Converting image mode from '{save_img.mode}' to 'RGB'")
            save_img = save_img.convert("RGB")
        logger.info(f"🖼️  Annotated image mode: {save_img.mode}, Size: {save_img.size}")
        
        save_img.save(annotated_path, "JPEG", quality=95)
        logger.info(f"✅ SUCCESS: Annotated image saved at {annotated_path}")
        
        # Verify file exists after save
        if os.path.exists(annotated_path):
            logger.info(f"🔍 Verified: File exists at {annotated_path}")
        else:
            logger.error(f"❌ Failed: File not found after save: {annotated_path}")
            
    except Exception as e:
        logger.error(f"❌ FAILED to save annotated image: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to save annotated image: {str(e)}")

    # Create database record
    inspection = database.Inspection(
        image_path=f"/images/{filename}",
        annotated_path=f"/annotated/{annotated_filename}",
        prediction=results["label"],
        confidence=results["confidence"],
        image_metadata={"filename": file.filename, "content_type": file.content_type}
    )
    db.add(inspection)
    try:
        db.commit()
        db.refresh(inspection)
        logger.info(f"✅ Saved to database: ID {inspection.id}")
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Database save failed: {e}")
        raise HTTPException(status_code=500, detail="Database save failed")

    logger.info("📤 Returning response to client")
    return inspection