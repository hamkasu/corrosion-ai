# app/database.py

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Supabase connection
DATABASE_URL = "postgresql://postgres:061167@aB1@db.hrkclrwvolzypwphpxpz.supabase.co:5432/postgres"

# Replace with your actual Supabase DB password (from Supabase dashboard)
# Or use environment variable:
# DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
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

print("Creating tables in Supabase...")
Base.metadata.create_all(bind=engine)
print("✅ Tables created!")