import logging
import threading
from typing import Optional, List, Dict, Any
import numpy as np
from ultralytics import YOLO
from app.config import settings

logger = logging.getLogger(__name__)

class TrackedDetection:
    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        confidence: float,
        track_id: Optional[int],
        class_id: int = 0,
        class_label: str = "player"
    ):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.confidence = float(confidence)
        self.track_id = int(track_id) if track_id is not None else None
        self.class_id = int(class_id)
        self.class_label = class_label

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_label": self.class_label
        }

class TrackerService:
    _instance: Optional["TrackerService"] = None
    _lock = threading.Lock()

    def __init__(self, model_name: str = settings.YOLO_MODEL_NAME):
        self.model_name = model_name
        logger.info(f"Initializing TrackerService with model {model_name}")
        self.model = YOLO(model_name)
        self._inference_lock = threading.Lock()

    @classmethod
    def get_instance(cls, model_name: str = settings.YOLO_MODEL_NAME) -> "TrackerService":
        with cls._lock:
            if cls._instance is None or cls._instance.model_name != model_name:
                cls._instance = cls(model_name=model_name)
            return cls._instance

    def reset_tracker(self):
        """Reset internal tracker state before processing a new video sequence."""
        with self._inference_lock:
            try:
                if hasattr(self.model, "predictor") and self.model.predictor is not None:
                    if hasattr(self.model.predictor, "trackers") and self.model.predictor.trackers:
                        for trk in self.model.predictor.trackers:
                            if hasattr(trk, "reset"):
                                trk.reset()
            except Exception as e:
                logger.warning(f"Error resetting tracker state: {e}")

    def track_frame(
        self,
        frame: np.ndarray,
        conf_threshold: float = settings.DEFAULT_CONF_THRESHOLD,
        persist: bool = True
    ) -> List[TrackedDetection]:
        """
        Track players in a video frame using YOLO + ByteTrack.
        Returns list of TrackedDetection items with bounding boxes and track IDs.
        """
        with self._inference_lock:
            results = self.model.track(
                source=frame,
                persist=persist,
                tracker=settings.TRACKER_CONFIG,
                classes=[settings.PERSON_CLASS_ID],
                conf=conf_threshold,
                verbose=False
            )

        detections: List[TrackedDetection] = []
        if not results or len(results) == 0:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            
            track_id = None
            if box.id is not None:
                try:
                    track_id = int(box.id[0].item())
                except Exception:
                    track_id = None

            detections.append(
                TrackedDetection(
                    x1=xyxy[0],
                    y1=xyxy[1],
                    x2=xyxy[2],
                    y2=xyxy[3],
                    confidence=conf,
                    track_id=track_id,
                    class_id=cls_id,
                    class_label="player"
                )
            )

        return detections
