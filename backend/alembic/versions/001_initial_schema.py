"""Initial schema for analyses and detections

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-10-11 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        'analyses',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('original_filename', sa.String(length=255), nullable=False),
        sa.Column('stored_filename', sa.String(length=255), nullable=False),
        sa.Column('output_video_filename', sa.String(length=255), nullable=True),
        sa.Column('export_csv_filename', sa.String(length=255), nullable=True),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='queued'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('video_width', sa.Integer(), nullable=True),
        sa.Column('video_height', sa.Integer(), nullable=True),
        sa.Column('fps', sa.Float(), nullable=True),
        sa.Column('total_frames', sa.Integer(), nullable=True),
        sa.Column('video_duration', sa.Float(), nullable=True),
        sa.Column('processed_frames', sa.Integer(), server_default='0', nullable=False),
        sa.Column('detection_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('distinct_tracks_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('processing_duration', sa.Float(), nullable=True),
        sa.Column('sample_interval', sa.Integer(), server_default='1', nullable=False),
        sa.Column('conf_threshold', sa.Float(), server_default='0.35', nullable=False),
        sa.Column('model_name', sa.String(length=64), server_default='yolov8n.pt', nullable=False),
        sa.Column('analytics_summary', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_analyses_id'), 'analyses', ['id'], unique=False)
    op.create_index(op.f('ix_analyses_status'), 'analyses', ['status'], unique=False)

    op.create_table(
        'detections',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('analysis_id', sa.String(length=36), nullable=False),
        sa.Column('frame_number', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.Float(), nullable=False),
        sa.Column('track_id', sa.Integer(), nullable=True),
        sa.Column('class_id', sa.Integer(), server_default='0', nullable=False),
        sa.Column('class_label', sa.String(length=32), server_default='player', nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('x1', sa.Float(), nullable=False),
        sa.Column('y1', sa.Float(), nullable=False),
        sa.Column('x2', sa.Float(), nullable=False),
        sa.Column('y2', sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(['analysis_id'], ['analyses.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_detections_analysis_id'), 'detections', ['analysis_id'], unique=False)
    op.create_index(op.f('ix_detections_frame_number'), 'detections', ['frame_number'], unique=False)
    op.create_index(op.f('ix_detections_track_id'), 'detections', ['track_id'], unique=False)
    op.create_index('idx_analysis_frame', 'detections', ['analysis_id', 'frame_number'], unique=False)

def downgrade() -> None:
    op.drop_table('detections')
    op.drop_table('analyses')
