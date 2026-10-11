import os
import shutil
import math
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, status, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from app.database import get_db
from app.config import settings
from app.models.analysis import Analysis
from app.models.detection import Detection
from app.schemas.analysis import (
    AnalysisResponse,
    AnalysisCreateResponse,
    AnalysisListResponse,
    OverviewStats
)
from app.schemas.detection import DetectionsListResponse, DetectionResponse
from app.services.storage import StorageService
from app.services.jobs import JobOrchestrator

router = APIRouter()

@router.get("/health", tags=["System"])
def get_health():
    """Health and readiness endpoint."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model": settings.YOLO_MODEL_NAME,
        "tracker": settings.TRACKER_CONFIG
    }

@router.get("/overview", response_model=OverviewStats, tags=["Analytics"])
def get_overview(db: Session = Depends(get_db)):
    """Summary statistics for the overview dashboard."""
    total_analyses = db.query(Analysis).count()
    completed = db.query(Analysis).filter(Analysis.status == "completed").count()
    processing = db.query(Analysis).filter(Analysis.status == "processing").count()
    failed = db.query(Analysis).filter(Analysis.status == "failed").count()
    
    total_detections = db.query(func.coalesce(func.sum(Analysis.detection_count), 0)).scalar()

    recent = (
        db.query(Analysis)
        .order_by(desc(Analysis.created_at))
        .limit(5)
        .all()
    )

    return OverviewStats(
        total_analyses=total_analyses,
        completed_analyses=completed,
        processing_analyses=processing,
        failed_analyses=failed,
        total_detections=int(total_detections or 0),
        recent_analyses=[AnalysisResponse.model_validate(a) for a in recent]
    )

@router.post("/analyses", response_model=AnalysisCreateResponse, status_code=status.HTTP_201_CREATED, tags=["Analyses"])
async def create_analysis(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    sample_interval: int = Form(default=1),
    conf_threshold: float = Form(default=0.35),
    db: Session = Depends(get_db)
):
    """
    Accepts video upload, validates content and size, creates analysis record,
    and initiates asynchronous computer vision processing.
    """
    # 1. Validate File Extension
    original_filename = video.filename or "uploaded_video.mp4"
    file_ext = Path(original_filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file_ext}'. Allowed formats: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )

    # 2. Validate Parameters
    if sample_interval < 1 or sample_interval > settings.MAX_SAMPLE_INTERVAL:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Sample interval must be between 1 and {settings.MAX_SAMPLE_INTERVAL}"
        )
    if conf_threshold < 0.1 or conf_threshold > 1.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confidence threshold must be between 0.1 and 1.0"
        )

    # 3. Stream upload & validate file size
    stored_filename = StorageService.generate_unique_filename(original_filename)
    target_path = StorageService.get_upload_path(stored_filename)

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    total_bytes = 0

    try:
        with open(target_path, "wb") as buffer:
            while True:
                chunk = await video.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    buffer.close()
                    if target_path.exists():
                        target_path.unlink()
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB"
                    )
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if target_path.exists():
            target_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded video: {str(e)}"
        )

    if total_bytes == 0:
        if target_path.exists():
            target_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty"
        )

    # 4. Create Database Record
    analysis = Analysis(
        original_filename=original_filename,
        stored_filename=stored_filename,
        sample_interval=sample_interval,
        conf_threshold=conf_threshold,
        model_name=settings.YOLO_MODEL_NAME,
        status="queued"
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    # 5. Dispatch background processing job
    background_tasks.add_task(JobOrchestrator.run_analysis_job, analysis.id)

    return AnalysisCreateResponse(
        id=analysis.id,
        original_filename=analysis.original_filename,
        status=analysis.status,
        sample_interval=analysis.sample_interval,
        conf_threshold=analysis.conf_threshold,
        created_at=analysis.created_at,
        message="Video uploaded successfully. Analysis job queued."
    )

@router.get("/analyses", response_model=AnalysisListResponse, tags=["Analyses"])
def list_analyses(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Retrieve paginated list of video analyses."""
    query = db.query(Analysis).order_by(desc(Analysis.created_at))
    total = query.count()
    total_pages = math.ceil(total / page_size) if total > 0 else 1

    items = query.offset((page - 1) * page_size).limit(page_size).all()

    return AnalysisListResponse(
        items=[AnalysisResponse.model_validate(a) for a in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )

@router.get("/analyses/{analysis_id}", response_model=AnalysisResponse, tags=["Analyses"])
def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    """Retrieve full analysis details, metrics, and heatmap data."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis '{analysis_id}' not found"
        )
    return AnalysisResponse.model_validate(analysis)

@router.get("/analyses/{analysis_id}/detections", response_model=DetectionsListResponse, tags=["Analyses"])
def get_detections(
    analysis_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    frame_number: Optional[int] = Query(default=None),
    db: Session = Depends(get_db)
):
    """Retrieve paginated individual detection records for an analysis."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis '{analysis_id}' not found"
        )

    query = db.query(Detection).filter(Detection.analysis_id == analysis_id)
    if frame_number is not None:
        query = query.filter(Detection.frame_number == frame_number)

    query = query.order_by(Detection.frame_number, Detection.id)
    total = query.count()
    total_pages = math.ceil(total / page_size) if total > 0 else 1

    items = query.offset((page - 1) * page_size).limit(page_size).all()

    return DetectionsListResponse(
        items=[DetectionResponse.model_validate(d) for d in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )

@router.get("/analyses/{analysis_id}/video", tags=["Analyses"])
def get_analysis_video(analysis_id: str, db: Session = Depends(get_db)):
    """Stream or serve the annotated output video."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis '{analysis_id}' not found"
        )

    if analysis.status != "completed" or not analysis.output_video_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Annotated video is not available. Status: '{analysis.status}'"
        )

    video_path = StorageService.get_output_video_path(analysis.output_video_filename)
    if not video_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotated video file not found on disk"
        )

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=f"annotated_{analysis.original_filename}"
    )

@router.get("/analyses/{analysis_id}/export", tags=["Analyses"])
def export_analysis_csv(analysis_id: str, db: Session = Depends(get_db)):
    """Export detection records as CSV."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis '{analysis_id}' not found"
        )

    if not analysis.export_csv_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Export CSV has not been generated for this analysis"
        )

    csv_path = StorageService.get_export_csv_path(analysis.export_csv_filename)
    if not csv_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Export file not found on disk"
        )

    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=f"detections_{analysis_id}.csv"
    )

@router.post("/analyses/{analysis_id}/retry", response_model=AnalysisResponse, tags=["Analyses"])
def retry_analysis(
    analysis_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Retries a failed analysis job."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis '{analysis_id}' not found"
        )

    if analysis.status == "processing":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Analysis is currently in progress"
        )

    # Check input file still exists
    input_path = StorageService.get_upload_path(analysis.stored_filename)
    if not input_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Original uploaded video file no longer exists. Please upload again."
        )

    # Reset analysis record
    analysis.status = "queued"
    analysis.error_message = None
    analysis.started_at = None
    analysis.completed_at = None
    analysis.processed_frames = 0
    analysis.detection_count = 0
    analysis.distinct_tracks_count = 0

    # Clear previous detections
    db.query(Detection).filter(Detection.analysis_id == analysis_id).delete()
    db.commit()
    db.refresh(analysis)

    # Queue background task
    background_tasks.add_task(JobOrchestrator.run_analysis_job, analysis.id)

    return AnalysisResponse.model_validate(analysis)

@router.delete("/analyses/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Analyses"])
def delete_analysis(analysis_id: str, db: Session = Depends(get_db)):
    """Deletes an analysis record, related database detections, and artifact files safely."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis '{analysis_id}' not found"
        )

    # Delete disk artifacts
    StorageService.delete_analysis_files(
        stored_filename=analysis.stored_filename,
        output_video_filename=analysis.output_video_filename,
        export_csv_filename=analysis.export_csv_filename
    )

    # Delete database record (cascades detections)
    db.delete(analysis)
    db.commit()

    return None
