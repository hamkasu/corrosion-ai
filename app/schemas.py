# app/schemas.py

from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict

class InspectionCreate(BaseModel):
    image_path: str
    annotated_path: Optional[str] = None
    prediction: str
    confidence: float
    image_meta: Optional[Dict] = None
    project_id: str = "default"
    project_description: Optional[str] = None

class InspectionResponse(InspectionCreate):
    id: int
    uploaded_at: datetime
    corrected_label: Optional[str] = None
    is_corrected: bool = False

    class Config:
        from_attributes = True