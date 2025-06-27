# Use Python 3.11 slim base image for better performance
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies required for video processing and AI libraries
RUN apt-get update && apt-get install -y \
    # Video processing dependencies
    ffmpeg \
    x264 \
    x265 \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    # Audio processing
    libsndfile1 \
    libasound2-dev \
    portaudio19-dev \
    # System utilities
    wget \
    curl \
    git \
    build-essential \
    # For PyTorch and ML libraries
    libatlas-base-dev \
    liblapack-dev \
    libopenblas-dev \
    # For MediaPipe
    libprotobuf-dev \
    protobuf-compiler \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create directories for the app
RUN mkdir -p /app/clips \
    /app/downloads \
    /app/temp_uploads \
    /app/temp_vertical \
    /app/models

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch first (CPU version for smaller image, change to GPU if needed)
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
RUN pip install -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY models/ ./models/
COPY *.py ./
COPY *.md ./

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Change ownership of app directory
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/workflow/health || exit 1

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 