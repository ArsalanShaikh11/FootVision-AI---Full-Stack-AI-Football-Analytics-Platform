import io
from app.models.analysis import Analysis
from app.models.detection import Detection
from app.services.storage import StorageService

def test_create_and_get_analysis(client, db_session):
    video_bytes = b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2mp41" + b"A" * 1024
    files = {"video": ("match.mp4", io.BytesIO(video_bytes), "video/mp4")}
    data = {"sample_interval": 2, "conf_threshold": 0.4}

    # Create
    res = client.post("/api/analyses", files=files, data=data)
    assert res.status_code == 201
    analysis_data = res.json()
    assert "id" in analysis_data
    assert analysis_data["original_filename"] == "match.mp4"
    assert analysis_data["status"] == "queued"
    analysis_id = analysis_data["id"]

    # Get by ID
    get_res = client.get(f"/api/analyses/{analysis_id}")
    assert get_res.status_code == 200
    detail = get_res.json()
    assert detail["id"] == analysis_id
    assert detail["sample_interval"] == 2
    assert detail["conf_threshold"] == 0.4

    # List
    list_res = client.get("/api/analyses")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] >= 1
    assert any(item["id"] == analysis_id for item in list_data["items"])

def test_get_nonexistent_analysis(client):
    res = client.get("/api/analyses/nonexistent-id-123")
    assert res.status_code == 404

def test_delete_analysis(client, db_session):
    # Setup test record
    analysis = Analysis(
        id="del-test-1",
        original_filename="game.mp4",
        stored_filename="game_del.mp4",
        status="completed"
    )
    db_session.add(analysis)
    db_session.commit()

    del_res = client.delete("/api/analyses/del-test-1")
    assert del_res.status_code == 204

    # Verify deleted
    check_res = client.get("/api/analyses/del-test-1")
    assert check_res.status_code == 404
