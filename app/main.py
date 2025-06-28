"""
ClipsAI Backend - FastAPI Application
Advanced video processing with AI speaker tracking, scene detection, and vertical cropping
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from pathlib import Path

# Import routers
from app.routers import workflow
try:
    from app.routers import crop
    CROP_AVAILABLE = True
except ImportError:
    CROP_AVAILABLE = False

# Create FastAPI app
app = FastAPI(
    title="ClipsAI Backend",
    description="Advanced video processing with AI speaker tracking, scene detection, and vertical cropping",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(workflow.router, prefix="/workflow", tags=["workflow"])

# Include crop router if available
if CROP_AVAILABLE:
    app.include_router(crop.router, prefix="/crop", tags=["Video Cropping"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ClipsAI Backend API",
        "version": "1.0.0",
        "features": [
            "Advanced Speaker Diarization (pyannote.audio)",
            "Scene Change Detection (PySceneDetect)",
            "Face Detection (MediaPipe + MTCNN)",
            "Intelligent Transition Management",
            "Vertical Video Cropping (9:16)",
            "YouTube Video Processing",
            "Async Task Management"
        ],
        "docs": "/docs",
        "health": "/workflow/health"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url)
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    print("🚀 ClipsAI Backend starting up...")
    
    # Create necessary directories
    directories = ["clips", "downloads", "temp_uploads", "temp_vertical", "models"]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 Created directory: {dir_name}")
    
    print("✅ ClipsAI Backend ready!")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 ClipsAI Backend shutting down...")
    # Add any cleanup code here
    print("✅ Shutdown complete!")

if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )
