import logging
import threading
from typing import Optional, List, Dict, Any
import numpy as np
from ultralytics import YOLO
from app.config import settings

logger = logging.getLogger(__name__)

class DetectionResult:
    def __init__(self, x1: float, y1: float, x2: float, y2: float, confidence: float, class_id: int, class_label: str):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.class_label = class_label

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_label": self.class_label
        }

class DetectorService:
    _instance: Optional["DetectorService"] = None
    _lock = threading.Lock()

    def __init__(self, model_name: str = settings.YOLO_MODEL_NAME):
        self.model_name = model_name
        logger.info(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        self._inference_lock = threading.Lock()
        logger.info(f"YOLO model {model_name} loaded successfully.")

    @classmethod
    def get_instance(cls, model_name: str = settings.YOLO_MODEL_NAME) -> "DetectorService":
        with cls._lock:
            if cls._instance is None or cls._instance.model_name != model_name:
                cls._instance = cls(model_name=model_name)
            return cls._instance

    def detect_players(self, frame: np.ndarray, conf_threshold: float = settings.DEFAULT_CONF_THRESHOLD) -> List[DetectionResult]:
        """
        Runs YOLO object detection filtered to COCO person class (0).
        Thread-safe inference using a model lock.
        """
        with self._inference_lock:
            results = self.model.predict(
                source=frame,
                classes=[settings.PERSON_CLASS_ID],
                conf=conf_threshold,
                verbose=False
            )
        
        detections: List[DetectionResult] = []
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
            
            detections.append(
                DetectionResult(
                    x1=xyxy[0],
                    y1=xyxy[1],
                    x2=xyxy[2],
                    y2=xyxy[3],
                    confidence=conf,
                    class_id=cls_id,
                    class_label="player"
                )
            )
        return detections
