from pathlib import Path
import pandas as pd
from app.services.analytics import AnalyticsService

def test_calculate_analytics_with_tracks():
    detections = [
        # Track 1
        {"analysis_id": "test-1", "frame_number": 0, "timestamp": 0.0, "track_id": 1, "confidence": 0.85, "x1": 50, "y1": 50, "x2": 70, "y2": 110, "class_id": 0, "class_label": "player"},
        {"analysis_id": "test-1", "frame_number": 1, "timestamp": 0.1, "track_id": 1, "confidence": 0.87, "x1": 55, "y1": 52, "x2": 75, "y2": 112, "class_id": 0, "class_label": "player"},
        {"analysis_id": "test-1", "frame_number": 2, "timestamp": 0.2, "track_id": 1, "confidence": 0.89, "x1": 60, "y1": 54, "x2": 80, "y2": 114, "class_id": 0, "class_label": "player"},
        # Track 2
        {"analysis_id": "test-1", "frame_number": 1, "timestamp": 0.1, "track_id": 2, "confidence": 0.72, "x1": 150, "y1": 80, "x2": 170, "y2": 140, "class_id": 0, "class_label": "player"},
        {"analysis_id": "test-1", "frame_number": 2, "timestamp": 0.2, "track_id": 2, "confidence": 0.75, "x1": 155, "y1": 82, "x2": 175, "y2": 142, "class_id": 0, "class_label": "player"},
    ]

    result = AnalyticsService.calculate_analytics(
        detections=detections,
        total_processed_frames=3,
        video_width=300,
        video_height=200
    )

    assert result["total_processed_frames"] == 3
    assert result["total_detections"] == 5
    assert result["distinct_tracks_count"] == 2
    assert result["avg_detections_per_frame"] == round(5 / 3, 2)
    
    tracks = result["tracks_summary"]
    assert len(tracks) == 2
    track1 = next(t for t in tracks if t["track_id"] == 1)
    assert track1["appearances"] == 3
    assert track1["first_seen"] == 0.0
    assert track1["last_seen"] == 0.2
    assert track1["duration_seconds"] == 0.2

    # Detections over time
    time_series = result["detections_over_time"]
    assert len(time_series) == 3
    assert time_series[0]["detection_count"] == 1  # frame 0
    assert time_series[1]["detection_count"] == 2  # frame 1
    assert time_series[2]["detection_count"] == 2  # frame 2

    # Heatmap points
    heatmap = result["heatmap_points"]
    assert len(heatmap) > 0
    for pt in heatmap:
        assert 0.0 <= pt["x"] <= 1.0
        assert 0.0 <= pt["y"] <= 1.0
        assert 0.0 <= pt["intensity"] <= 1.0

def test_export_to_csv(tmp_path):
    csv_file = tmp_path / "test_export.csv"
    detections = [
        {"analysis_id": "a1", "frame_number": 1, "timestamp": 0.04, "track_id": 5, "class_id": 0, "class_label": "player", "confidence": 0.91, "x1": 10, "y1": 20, "x2": 30, "y2": 70}
    ]
    out = AnalyticsService.export_to_csv(detections, csv_file)
    assert out.exists()
    df = pd.read_csv(csv_file)
    assert len(df) == 1
    assert df.iloc[0]["track_id"] == 5
    assert df.iloc[0]["class_label"] == "player"
