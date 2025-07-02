from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routers import transcript, workflow, subtitles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

app = FastAPI(
    title="ClipLink API",
    description="YouTube transcript extraction, Gemini AI analysis, video clip generation, and subtitle processing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files

@app.get("/")
def read_root():
    return {
        "message": "ClipLink API", 
        "endpoints": {
            "transcript": "/transcript - Extract transcript from YouTube URL",
            "analyze": "/analyze - Analyze transcript with Gemini AI",
            "workflow": {
                "process_complete": "/workflow/process-complete - Full workflow: transcript → analysis → download → clips",
                "process_complete_async": "/workflow/process-complete-async - Async version of full workflow",
                "process_comprehensive_async": "/workflow/process-comprehensive-async - 🚀 ULTIMATE: transcript → gemini → download → vertical crop → burn subtitles (ALL-IN-ONE)",
                "analyze_only": "/workflow/analyze-only - Only transcript + analysis"
            },
            "subtitles": "/subtitles - Generate subtitles and burn them into video",
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "cliplink_api"}

# Include routers
app.include_router(transcript.router)
app.include_router(workflow.router, prefix="/workflow", tags=["workflow"])
app.include_router(subtitles.router, tags=["subtitles"])
