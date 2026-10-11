import os
import sys
import tempfile
from pathlib import Path
import pytest
import numpy as np
import cv2

# Add backend directory to sys.path so 'app' is discoverable
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy.pool import StaticPool
from app.database import Base, get_db
from app.main import app
from app.config import settings

# Test database with StaticPool so all connections share the in-memory database
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

@pytest.fixture(scope="session", autouse=True)
def setup_test_directories():
    test_storage = Path(tempfile.mkdtemp(prefix="footvision_test_"))
    settings.STORAGE_DIR = test_storage
    settings.UPLOAD_DIR = test_storage / "uploads"
    settings.OUTPUT_DIR = test_storage / "outputs"
    settings.EXPORTS_DIR = test_storage / "exports"
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    yield test_storage

@pytest.fixture
def db_session():
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)

@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
def sample_video_path(tmp_path):
    """Generates a small 10-frame synthetic mp4 video with a moving rectangular player."""
    video_path = tmp_path / "sample_match.mp4"
    width, height = 320, 240
    fps = 10.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(10):
        # Green pitch background
        frame = np.full((height, width, 3), (35, 120, 35), dtype=np.uint8)
        # Pitch lines
        cv2.line(frame, (0, 120), (320, 120), (255, 255, 255), 2)
        cv2.circle(frame, (160, 120), 30, (255, 255, 255), 2)
        # Moving player figure
        x = int(50 + i * 15)
        y = int(80 + i * 5)
        # Head
        cv2.circle(frame, (x + 10, y + 10), 8, (200, 200, 240), -1)
        # Torso
        cv2.rectangle(frame, (x, y + 18), (x + 20, y + 60), (0, 0, 220), -1)
        out.write(frame)

    out.release()
    return video_path
