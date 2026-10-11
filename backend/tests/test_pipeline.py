import os
from pathlib import Path
from app.services.video_processor import VideoProcessor, VideoMetadata

def test_probe_video(sample_video_path):
    meta = VideoProcessor.probe_video(sample_video_path)
    assert meta.width == 320
    assert meta.height == 240
    assert meta.fps == 10.0
    assert meta.total_frames == 10
    assert meta.duration == 1.0

def test_process_video_pipeline(sample_video_path, tmp_path):
    output_video_path = tmp_path / "annotated_sample.mp4"
    
    result = VideoProcessor.process_video(
        input_path=sample_video_path,
        output_path=output_video_path,
        analysis_id="test-pipeline-1",
        sample_interval=1,
        conf_threshold=0.25
    )

    assert output_video_path.exists()
    assert output_video_path.stat().st_size > 0
    assert result["processed_frames"] == 10
    assert result["metadata"]["width"] == 320
    assert isinstance(result["detections"], list)
    assert result["processing_duration"] >= 0.0
