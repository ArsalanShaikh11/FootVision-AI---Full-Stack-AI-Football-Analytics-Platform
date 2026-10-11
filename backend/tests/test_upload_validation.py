import io
import pytest

def test_upload_invalid_extension(client):
    file_content = b"fake video bytes"
    files = {"video": ("match.txt", io.BytesIO(file_content), "text/plain")}
    response = client.post("/api/analyses", files=files)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

def test_upload_empty_file(client):
    files = {"video": ("match.mp4", io.BytesIO(b""), "video/mp4")}
    response = client.post("/api/analyses", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_upload_invalid_sample_interval(client):
    file_content = b"mock bytes"
    files = {"video": ("match.mp4", io.BytesIO(file_content), "video/mp4")}
    data = {"sample_interval": 0}
    response = client.post("/api/analyses", files=files, data=data)
    assert response.status_code == 400
    assert "Sample interval" in response.json()["detail"]

def test_upload_invalid_confidence_threshold(client):
    file_content = b"mock bytes"
    files = {"video": ("match.mp4", io.BytesIO(file_content), "video/mp4")}
    data = {"conf_threshold": 1.5}
    response = client.post("/api/analyses", files=files, data=data)
    assert response.status_code == 400
    assert "Confidence threshold" in response.json()["detail"]
