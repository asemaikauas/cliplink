"""
Pydantic schemas for API requests and responses

This module defines the data structures used for API serialization
and validation in the Cliplink backend.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime
from uuid import UUID
import enum


class VideoStatus(str, enum.Enum):
    """Video processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class ClipResponse(BaseModel):
    """Response model for video clip data"""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    video_id: UUID
    s3_url: str
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    duration: float = Field(description="Duration in seconds")
    created_at: datetime


class VideoResponse(BaseModel):
    """Response model for video data"""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    user_id: UUID
    youtube_id: str = Field(description="YouTube video ID")
    title: Optional[str] = None
    status: VideoStatus
    created_at: datetime
    clips: List[ClipResponse] = Field(default_factory=list)


class VideoSummaryResponse(BaseModel):
    """Summary response model for video without clips"""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    user_id: UUID
    youtube_id: str
    title: Optional[str] = None
    status: VideoStatus
    created_at: datetime
    clips_count: int = Field(description="Number of clips generated")


class VideosListResponse(BaseModel):
    """Response model for paginated video list"""
    videos: List[VideoSummaryResponse]
    total: int
    page: int
    per_page: int
    has_next: bool
    has_prev: bool


class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str
    message: str
    details: Optional[dict] = None


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    service: str
    timestamp: datetime
    database_connected: bool 