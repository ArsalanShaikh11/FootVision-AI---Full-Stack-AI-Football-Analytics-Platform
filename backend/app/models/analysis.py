import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)
    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False)
    output_video_filename = Column(String(255), nullable=True)
    export_csv_filename = Column(String(255), nullable=True)
    
    # Status: queued, processing, completed, failed
    status = Column(String(32), default="queued", index=True, nullable=False)
    error_message = Column(Text, nullable=True)
    
    # Video metadata
    video_width = Column(Integer, nullable=True)
    video_height = Column(Integer, nullable=True)
    fps = Column(Float, nullable=True)
    total_frames = Column(Integer, nullable=True)
    video_duration = Column(Float, nullable=True)  # in seconds
    
    # Processing statistics
    processed_frames = Column(Integer, default=0)
    detection_count = Column(Integer, default=0)
    distinct_tracks_count = Column(Integer, default=0)
    processing_duration = Column(Float, nullable=True)  # in seconds
    
    # Configuration
    sample_interval = Column(Integer, default=1)
    conf_threshold = Column(Float, default=0.35)
    model_name = Column(String(64), default="yolov8n.pt")
    
    # Cached metrics & heatmap JSON
    analytics_summary = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    detections = relationship("Detection", back_populates="analysis", cascade="all, delete-orphan")
