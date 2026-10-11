import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.analysis import Analysis
from app.models.detection import Detection
from app.services.video_processor import VideoProcessor
from app.services.analytics import AnalyticsService
from app.services.storage import StorageService
from app.config import settings

logger = logging.getLogger(__name__)

class JobOrchestrator:
    @staticmethod
    def run_analysis_job(analysis_id: str):
        """
        Background worker task that executes the computer vision pipeline,
        persists detections and analytics, and updates job status.
        """
        db: Session = SessionLocal()
        try:
            analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
            if not analysis:
                logger.error(f"Analysis job {analysis_id} not found in database.")
                return

            # Mark as processing
            analysis.status = "processing"
            analysis.started_at = datetime.now(timezone.utc)
            analysis.error_message = None
            db.commit()

            input_path = StorageService.get_upload_path(analysis.stored_filename)
            output_video_name = f"annotated_{analysis.stored_filename}"
            output_path = StorageService.get_output_video_path(output_video_name)

            logger.info(f"Starting processing for analysis {analysis_id} (file: {input_path.name})")

            # 1. Process Video
            result = VideoProcessor.process_video(
                input_path=input_path,
                output_path=output_path,
                analysis_id=analysis_id,
                sample_interval=analysis.sample_interval,
                conf_threshold=analysis.conf_threshold
            )

            meta = result["metadata"]
            raw_detections = result["detections"]
            processed_frames = result["processed_frames"]
            processing_duration = result["processing_duration"]

            # 2. Compute Analytics & Heatmap
            analytics = AnalyticsService.calculate_analytics(
                detections=raw_detections,
                total_processed_frames=processed_frames,
                video_width=meta.get("width"),
                video_height=meta.get("height")
            )

            # 3. Export CSV
            export_csv_name = f"detections_{analysis_id}.csv"
            export_csv_path = StorageService.get_export_csv_path(export_csv_name)
            AnalyticsService.export_to_csv(raw_detections, export_csv_path)

            # 4. Batch insert detections into Database
            if raw_detections:
                detection_objects = [
                    Detection(
                        analysis_id=d["analysis_id"],
                        frame_number=d["frame_number"],
                        timestamp=d["timestamp"],
                        track_id=d["track_id"],
                        class_id=d["class_id"],
                        class_label=d["class_label"],
                        confidence=d["confidence"],
                        x1=d["x1"],
                        y1=d["y1"],
                        x2=d["x2"],
                        y2=d["y2"]
                    )
                    for d in raw_detections
                ]
                # Efficient batch insert
                db.bulk_save_objects(detection_objects)

            # 5. Update Analysis Record
            analysis.video_width = meta.get("width")
            analysis.video_height = meta.get("height")
            analysis.fps = meta.get("fps")
            analysis.total_frames = meta.get("total_frames")
            analysis.video_duration = meta.get("duration")

            analysis.processed_frames = processed_frames
            analysis.detection_count = len(raw_detections)
            analysis.distinct_tracks_count = analytics.get("distinct_tracks_count", 0)
            analysis.processing_duration = processing_duration

            analysis.output_video_filename = output_video_name
            analysis.export_csv_filename = export_csv_name
            analysis.analytics_summary = analytics

            analysis.status = "completed"
            analysis.completed_at = datetime.now(timezone.utc)

            db.commit()
            logger.info(f"Analysis {analysis_id} completed successfully in {processing_duration}s.")

        except Exception as e:
            logger.error(f"Error processing analysis {analysis_id}: {str(e)}\n{traceback.format_exc()}")
            db.rollback()
            try:
                analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
                if analysis:
                    analysis.status = "failed"
                    analysis.error_message = (
                        f"Video processing failed: {str(e)}"
                        if len(str(e)) < 200 else "Processing failed due to a video decoding or inference error."
                    )
                    analysis.completed_at = datetime.now(timezone.utc)
                    db.commit()
            except Exception as inner_e:
                logger.error(f"Failed to record failure status for {analysis_id}: {inner_e}")
        finally:
            db.close()
