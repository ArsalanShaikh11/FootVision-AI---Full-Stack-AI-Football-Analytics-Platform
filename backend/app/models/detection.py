from sqlalchemy import Column, Integer, Float, String, ForeignKey, Index
from sqlalchemy.orm import relationship
from app.database import Base

class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String(36), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True)
    frame_number = Column(Integer, nullable=False, index=True)
    timestamp = Column(Float, nullable=False)
    track_id = Column(Integer, nullable=True, index=True)
    class_id = Column(Integer, default=0, nullable=False)
    class_label = Column(String(32), default="player", nullable=False)
    confidence = Column(Float, nullable=False)
    
    # Bounding box
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)

    # Relationship
    analysis = relationship("Analysis", back_populates="detections")

# Composite index for querying detections by analysis and frame
Index("idx_analysis_frame", Detection.analysis_id, Detection.frame_number)
