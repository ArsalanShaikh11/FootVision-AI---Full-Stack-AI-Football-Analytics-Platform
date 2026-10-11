from typing import Optional
from pydantic import BaseModel, ConfigDict

class DetectionBase(BaseModel):
    frame_number: int
    timestamp: float
    track_id: Optional[int] = None
    class_id: int = 0
    class_label: str = "player"
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

class DetectionCreate(DetectionBase):
    analysis_id: str

class DetectionResponse(DetectionBase):
    id: int
    analysis_id: str

    model_config = ConfigDict(from_attributes=True)

class DetectionsListResponse(BaseModel):
    items: list[DetectionResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
