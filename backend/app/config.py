import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "FootVision AI"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # Database
    # Defaults to local SQLite for frictionless local testing and supports PostgreSQL in production/Docker
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/footvision.db")
    
    # Storage
    STORAGE_DIR: Path = Path(os.getenv("STORAGE_DIR", str(BASE_DIR / "storage")))
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    OUTPUT_DIR: Path = STORAGE_DIR / "outputs"
    EXPORTS_DIR: Path = STORAGE_DIR / "exports"
    
    # Video & Inference Configuration
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))
    ALLOWED_EXTENSIONS: list[str] = [".mp4", ".avi", ".mov"]
    DEFAULT_CONF_THRESHOLD: float = 0.35
    DEFAULT_SAMPLE_INTERVAL: int = 1  # 1 = process every frame
    MAX_SAMPLE_INTERVAL: int = 10
    
    # YOLO & Tracker configuration
    YOLO_MODEL_NAME: str = os.getenv("YOLO_MODEL_NAME", "yolov8n.pt")  # Official lightweight pretrained checkpoint
    TRACKER_CONFIG: str = "bytetrack.yaml"
    PERSON_CLASS_ID: int = 0  # COCO class index for 'person'
    
    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

settings = Settings()

# Ensure storage directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
settings.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
