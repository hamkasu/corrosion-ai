# app/database.py

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Replace with your real Supabase URL
DATABASE_URL = "postgresql://postgres:ljdNQ1fnNnNRr1Tu@db.hrkclrwvolzypwphpxpz.supabase.co:5432/postgres"


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Inspection(Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, nullable=False)
    annotated_path = Column(String, nullable=True)
    prediction = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    image_metadata = Column(JSON)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    is_corrected = Column(Boolean, default=False)
    corrected_label = Column(String, nullable=True)
    project_id = Column(String, default="default")
    project_description = Column(String, nullable=True)