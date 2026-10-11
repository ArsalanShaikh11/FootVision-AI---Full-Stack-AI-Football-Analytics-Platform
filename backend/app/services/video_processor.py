import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import cv2
import numpy as np

from app.services.tracker import TrackerService, TrackedDetection
from app.config import settings

logger = logging.getLogger(__name__)

class VideoMetadata:
    def __init__(self, width: int, height: int, fps: float, total_frames: int, duration: float):
        self.width = width
        self.height = height
        self.fps = fps
        self.total_frames = total_frames
        self.duration = duration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "duration": round(self.duration, 2)
        }

class VideoProcessor:
    @staticmethod
    def probe_video(video_path: Path) -> VideoMetadata:
        """
        Extract dimensions, fps, frame count and duration from video.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found at {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path.name}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0 or np.isnan(fps):
                fps = 25.0  # safe default fallback
            if total_frames < 0:
                total_frames = 0
            
            duration = (total_frames / fps) if (total_frames > 0 and fps > 0) else 0.0

            return VideoMetadata(
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration=duration
            )
        finally:
            cap.release()

    @staticmethod
    def draw_annotations(
        frame: np.ndarray,
        detections: List[TrackedDetection],
        frame_idx: int,
        timestamp: float,
        fps: float
    ) -> np.ndarray:
        """
        Draws sleek, sports-tech bounding boxes, track IDs, and telemetry overlays.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Colors (BGR)
        LIME_GREEN = (50, 230, 80)
        DARK_BG = (20, 24, 28)
        CYAN_ACCENT = (230, 180, 40)
        WHITE = (245, 245, 245)

        for det in detections:
            x1, y1 = max(0, int(det.x1)), max(0, int(det.y1))
            x2, y2 = min(w, int(det.x2)), min(h, int(det.y2))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), LIME_GREEN, 2, lineType=cv2.LINE_AA)

            # Center foot anchor point
            cx = int((x1 + x2) / 2)
            cv2.circle(annotated, (cx, y2), 4, LIME_GREEN, -1, lineType=cv2.LINE_AA)

            # Badge Label
            track_label = f"ID #{det.track_id}" if det.track_id is not None else "DET"
            conf_label = f"{int(det.confidence * 100)}%"
            label_text = f"{track_label} | {conf_label}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            badge_y1 = max(0, y1 - text_h - 8)
            badge_y2 = y1
            badge_x2 = min(w, x1 + text_w + 10)

            # Draw badge background
            cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), DARK_BG, -1)
            cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), LIME_GREEN, 1)

            # Draw text
            text_pos = (x1 + 5, y1 - 4)
            cv2.putText(annotated, label_text, text_pos, font, font_scale, WHITE, thickness, lineType=cv2.LINE_AA)

        # Top-Left Telemetry HUD
        hud_w, hud_h = 240, 56
        overlay = annotated.copy()
        cv2.rectangle(overlay, (12, 12), (12 + hud_w, 12 + hud_h), DARK_BG, -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(annotated, (12, 12), (12 + hud_w, 12 + hud_h), CYAN_ACCENT, 1)

        # Time format: mm:ss.ms
        mins = int(timestamp // 60)
        secs = timestamp % 60
        time_str = f"{mins:02d}:{secs:04.1f}"

        cv2.putText(
            annotated,
            f"FOOTVISION AI | T: {time_str}",
            (22, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            WHITE,
            1,
            lineType=cv2.LINE_AA
        )
        cv2.putText(
            annotated,
            f"FRAME: {frame_idx} | ACTIVE: {len(detections)}",
            (22, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            LIME_GREEN,
            1,
            lineType=cv2.LINE_AA
        )

        return annotated

    @staticmethod
    def process_video(
        input_path: Path,
        output_path: Path,
        analysis_id: str,
        sample_interval: int = 1,
        conf_threshold: float = 0.35,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes streaming video decoding, YOLO+ByteTrack inference,
        annotation generation, and video writing.
        """
        metadata = VideoProcessor.probe_video(input_path)
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file for processing: {input_path}")

        # Reset tracker state for a clean run
        tracker_service = TrackerService.get_instance()
        tracker_service.reset_tracker()

        # Target output FPS matches effective output frame rate
        effective_fps = max(1.0, metadata.fps / max(1, sample_interval))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            effective_fps,
            (metadata.width, metadata.height)
        )

        raw_detections: List[Dict[str, Any]] = []
        frame_idx = 0
        processed_count = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Frame sampling check
                if frame_idx % sample_interval == 0:
                    timestamp = frame_idx / metadata.fps if metadata.fps > 0 else 0.0

                    # Run tracker on frame
                    tracked = tracker_service.track_frame(
                        frame=frame,
                        conf_threshold=conf_threshold,
                        persist=True
                    )

                    # Store detections
                    for det in tracked:
                        raw_detections.append({
                            "analysis_id": analysis_id,
                            "frame_number": frame_idx,
                            "timestamp": round(timestamp, 3),
                            "track_id": det.track_id,
                            "class_id": det.class_id,
                            "class_label": det.class_label,
                            "confidence": round(det.confidence, 4),
                            "x1": round(det.x1, 2),
                            "y1": round(det.y1, 2),
                            "x2": round(det.x2, 2),
                            "y2": round(det.y2, 2)
                        })

                    # Draw sleek annotations
                    annotated_frame = VideoProcessor.draw_annotations(
                        frame=frame,
                        detections=tracked,
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        fps=metadata.fps
                    )

                    writer.write(annotated_frame)
                    processed_count += 1

                    if progress_callback and metadata.total_frames > 0:
                        progress_callback(frame_idx + 1, metadata.total_frames)

                frame_idx += 1

        finally:
            cap.release()
            writer.release()

        # Transcode output to web-compatible H.264 for instant HTML5 playback
        VideoProcessor.transcode_to_web_h264(output_path)

        processing_duration = round(time.time() - start_time, 2)

        return {
            "metadata": metadata.to_dict(),
            "processed_frames": processed_count,
            "total_frames": frame_idx,
            "processing_duration": processing_duration,
            "detections": raw_detections
        }

    @staticmethod
    def transcode_to_web_h264(video_path: Path) -> None:
        """
        Transcodes OpenCV output video to web-standard H.264 (avc1 + yuv420p)
        with faststart metadata for instantaneous playback across modern browsers.
        """
        try:
            ffmpeg_exe = None
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                import shutil
                ffmpeg_exe = shutil.which("ffmpeg")

            if not ffmpeg_exe:
                logger.info("ffmpeg executable not found; skipping web H.264 transcode.")
                return

            temp_path = video_path.parent / f"web_{video_path.name}"
            cmd = [
                ffmpeg_exe,
                "-y",
                "-i", str(video_path),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(temp_path)
            ]
            import subprocess
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if temp_path.exists() and temp_path.stat().st_size > 0:
                os.replace(temp_path, video_path)
                logger.info(f"Video {video_path.name} successfully optimized for browser HTML5 streaming.")
        except Exception as e:
            logger.warning(f"Web H.264 transcode fallback: {e}")
