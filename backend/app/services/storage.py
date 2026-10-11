import os
import shutil
import uuid
from pathlib import Path
from fastapi import HTTPException, status
from app.config import settings

class StorageService:
    @staticmethod
    def generate_unique_filename(original_filename: str, prefix: str = "") -> str:
        ext = Path(original_filename).suffix.lower()
        if not ext:
            ext = ".mp4"
        clean_ext = ext if ext in settings.ALLOWED_EXTENSIONS else ".mp4"
        unique_id = uuid.uuid4().hex
        if prefix:
            return f"{prefix}_{unique_id}{clean_ext}"
        return f"{unique_id}{clean_ext}"

    @staticmethod
    def get_upload_path(filename: str) -> Path:
        # Validate path against traversal
        safe_filename = Path(filename).name
        target_path = (settings.UPLOAD_DIR / safe_filename).resolve()
        if not str(target_path).startswith(str(settings.UPLOAD_DIR.resolve())):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file path detected"
            )
        return target_path

    @staticmethod
    def get_output_video_path(filename: str) -> Path:
        safe_filename = Path(filename).name
        target_path = (settings.OUTPUT_DIR / safe_filename).resolve()
        if not str(target_path).startswith(str(settings.OUTPUT_DIR.resolve())):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file path detected"
            )
        return target_path

    @staticmethod
    def get_export_csv_path(filename: str) -> Path:
        safe_filename = Path(filename).name
        target_path = (settings.EXPORTS_DIR / safe_filename).resolve()
        if not str(target_path).startswith(str(settings.EXPORTS_DIR.resolve())):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file path detected"
            )
        return target_path

    @staticmethod
    def delete_analysis_files(
        stored_filename: str | None = None,
        output_video_filename: str | None = None,
        export_csv_filename: str | None = None
    ) -> None:
        """Safely delete files associated with an analysis without touching any other files."""
        if stored_filename:
            try:
                p = StorageService.get_upload_path(stored_filename)
                if p.exists() and p.is_file():
                    p.unlink()
            except Exception:
                pass
        
        if output_video_filename:
            try:
                p = StorageService.get_output_video_path(output_video_filename)
                if p.exists() and p.is_file():
                    p.unlink()
            except Exception:
                pass

        if export_csv_filename:
            try:
                p = StorageService.get_export_csv_path(export_csv_filename)
                if p.exists() and p.is_file():
                    p.unlink()
            except Exception:
                pass
