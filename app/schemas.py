# app/schemas.py

from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class InspectionCreate(BaseModel):
    image_path: str
    annotated_path: Optional[str] = None
    prediction: str
    confidence: float
    masks: Optional[List[List[List[float]]]] = None  # List of 2D masks
    image_meta: Optional[dict] = None

class InspectionResponse(InspectionCreate):
    id: int
    uploaded_at: datetime
    corrected_label: Optional[str] = None
    is_corrected: bool = False

    class Config:
        from_attributes = True