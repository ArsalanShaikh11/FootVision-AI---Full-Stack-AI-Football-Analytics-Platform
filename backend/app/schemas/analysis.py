from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, ConfigDict, Field

class AnalysisConfig(BaseModel):
    sample_interval: int = Field(default=1, ge=1, le=10, description="Process every Nth frame")
    conf_threshold: float = Field(default=0.35, ge=0.1, le=1.0, description="Detection confidence threshold")

class AnalysisBase(BaseModel):
    original_filename: str

class AnalysisCreateResponse(BaseModel):
    id: str
    original_filename: str
    status: str
    sample_interval: int
    conf_threshold: float
    created_at: datetime
    message: str

class TrackDurationItem(BaseModel):
    track_id: int
    appearances: int
    first_seen: float
    last_seen: float
    duration_seconds: float

class HeatmapPoint(BaseModel):
    x: float
    y: float
    intensity: float

class DetectionOverTimeItem(BaseModel):
    frame_number: int
    timestamp: float
    detection_count: int

class AnalyticsSummary(BaseModel):
    total_processed_frames: int
    total_detections: int
    avg_detections_per_frame: float
    distinct_tracks_count: int
    tracks_summary: list[TrackDurationItem]
    detections_over_time: list[DetectionOverTimeItem]
    heatmap_points: list[HeatmapPoint]

class AnalysisResponse(BaseModel):
    id: str
    original_filename: str
    status: str
    error_message: Optional[str] = None
    
    # Video metadata
    video_width: Optional[int] = None
    video_height: Optional[int] = None
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    video_duration: Optional[float] = None
    
    # Processing statistics
    processed_frames: int
    detection_count: int
    distinct_tracks_count: int
    processing_duration: Optional[float] = None
    
    # Configuration
    sample_interval: int
    conf_threshold: float
    model_name: str
    
    # Detailed analytics
    analytics_summary: Optional[AnalyticsSummary] = None
    
    # Timestamps
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

class AnalysisListResponse(BaseModel):
    items: list[AnalysisResponse]
    total: int
    page: int
    page_size: int
    total_pages: int

class OverviewStats(BaseModel):
    total_analyses: int
    completed_analyses: int
    processing_analyses: int
    failed_analyses: int
    total_detections: int
    recent_analyses: list[AnalysisResponse]
