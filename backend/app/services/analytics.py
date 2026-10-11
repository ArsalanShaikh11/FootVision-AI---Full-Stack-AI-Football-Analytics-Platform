from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np

class AnalyticsService:
    @staticmethod
    def calculate_analytics(
        detections: List[Dict[str, Any]],
        total_processed_frames: int,
        video_width: Optional[int] = None,
        video_height: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculates aggregate metrics, track summaries, time-series counts,
        and heatmap coordinates from raw detection records.
        """
        total_detections = len(detections)
        avg_detections_per_frame = (
            round(total_detections / total_processed_frames, 2)
            if total_processed_frames > 0 else 0.0
        )

        if not detections:
            return {
                "total_processed_frames": total_processed_frames,
                "total_detections": 0,
                "avg_detections_per_frame": 0.0,
                "distinct_tracks_count": 0,
                "tracks_summary": [],
                "detections_over_time": [],
                "heatmap_points": []
            }

        df = pd.DataFrame(detections)

        # 1. Track summaries
        tracks_summary = []
        distinct_tracks_count = 0
        if "track_id" in df.columns and df["track_id"].notna().any():
            tracked_df = df[df["track_id"].notna()].copy()
            tracked_df["track_id"] = tracked_df["track_id"].astype(int)
            grouped = tracked_df.groupby("track_id")
            distinct_tracks_count = int(grouped.ngroups)

            for track_id, group in grouped:
                first_seen = float(group["timestamp"].min())
                last_seen = float(group["timestamp"].max())
                duration = round(last_seen - first_seen, 2)
                tracks_summary.append({
                    "track_id": int(track_id),
                    "appearances": int(len(group)),
                    "first_seen": round(first_seen, 2),
                    "last_seen": round(last_seen, 2),
                    "duration_seconds": max(duration, 0.0)
                })
            # Sort tracks by appearances descending
            tracks_summary.sort(key=lambda x: x["appearances"], reverse=True)

        # 2. Detections over time (grouped by frame)
        frame_counts = df.groupby(["frame_number", "timestamp"]).size().reset_index(name="detection_count")
        detections_over_time = [
            {
                "frame_number": int(row["frame_number"]),
                "timestamp": round(float(row["timestamp"]), 2),
                "detection_count": int(row["detection_count"])
            }
            for _, row in frame_counts.sort_values("frame_number").iterrows()
        ]

        # 3. Heatmap calculation (normalized 0.0 to 1.0)
        # Use bounding-box bottom-center (foot position) for more accurate player placement
        width = float(video_width) if video_width and video_width > 0 else 1.0
        height = float(video_height) if video_height and video_height > 0 else 1.0

        # Use foot position (bottom-center of bounding box) — more grounded than box center
        df["norm_x"] = (((df["x1"] + df["x2"]) / 2.0) / width).clip(0.0, 1.0)
        df["norm_y"] = (df["y2"] / height).clip(0.0, 1.0)

        # High-resolution grid — 50 cols × 34 rows matches 105m × 68m pitch proportions
        grid_x_bins = 50
        grid_y_bins = 34

        # Map normalized coords to integer grid indices
        df["gx"] = (df["norm_x"] * (grid_x_bins - 1)).round().astype(int).clip(0, grid_x_bins - 1)
        df["gy"] = (df["norm_y"] * (grid_y_bins - 1)).round().astype(int).clip(0, grid_y_bins - 1)

        # Accumulate raw counts into a 2-D density matrix
        density = np.zeros((grid_y_bins, grid_x_bins), dtype=float)
        for gx_val, gy_val in zip(df["gx"], df["gy"]):
            density[gy_val, gx_val] += 1.0

        # Apply Gaussian smoothing so nearby cells get partial credit
        from scipy.ndimage import gaussian_filter
        density = gaussian_filter(density, sigma=1.0)

        # Normalize to [0, 1]
        max_density = density.max()
        if max_density > 0:
            density /= max_density

        # Emit only cells with non-trivial intensity (> 2% of max) to keep payload small
        heatmap_points = []
        threshold = 0.02
        for gy_idx in range(grid_y_bins):
            for gx_idx in range(grid_x_bins):
                val = float(density[gy_idx, gx_idx])
                if val > threshold:
                    heatmap_points.append({
                        "x": round(gx_idx / (grid_x_bins - 1), 4),
                        "y": round(gy_idx / (grid_y_bins - 1), 4),
                        "intensity": round(val, 4)
                    })

        return {
            "total_processed_frames": total_processed_frames,
            "total_detections": total_detections,
            "avg_detections_per_frame": avg_detections_per_frame,
            "distinct_tracks_count": distinct_tracks_count,
            "tracks_summary": tracks_summary,
            "detections_over_time": detections_over_time,
            "heatmap_points": heatmap_points
        }

    @staticmethod
    def export_to_csv(detections: List[Dict[str, Any]], output_path: Path) -> Path:
        """
        Exports detection records to a CSV file.
        """
        if not detections:
            df = pd.DataFrame(columns=[
                "analysis_id", "frame_number", "timestamp", "track_id",
                "class_id", "class_label", "confidence", "x1", "y1", "x2", "y2"
            ])
        else:
            df = pd.DataFrame(detections)
            # Ensure nice column ordering
            desired_order = [
                "analysis_id", "frame_number", "timestamp", "track_id",
                "class_id", "class_label", "confidence", "x1", "y1", "x2", "y2"
            ]
            cols = [c for c in desired_order if c in df.columns] + [c for c in df.columns if c not in desired_order]
            df = df[cols]

        df.to_csv(output_path, index=False)
        return output_path
