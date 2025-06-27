# ClipsAI Backend - Docker Deployment

This guide explains how to build and run the ClipsAI Backend using Docker, featuring advanced AI-powered video processing with speaker diarization, scene detection, and intelligent vertical cropping.

## 🚀 Quick Start

### Prerequisites

- Docker (20.10+)
- Docker Compose (2.0+)
- HuggingFace account and token (for AI features)

### 1. Environment Setup

```bash
# Copy environment template
cp env.example .env

# Edit .env file with your configuration
nano .env
```

**Important**: Add your HuggingFace token to the `.env` file:
```env
HF_TOKEN=your_actual_huggingface_token_here
```

Get your token from: https://huggingface.co/settings/tokens

### 2. Build and Run

```bash
# Build and start with Docker Compose
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

### 3. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/workflow/health
- **Root Endpoint**: http://localhost:8000

## 🐳 Docker Commands

### Building the Image

```bash
# Build the Docker image
docker build -t cliplink-backend .

# Build with specific tag
docker build -t cliplink-backend:v1.0.0 .
```

### Running the Container

```bash
# Run basic container
docker run -p 8000:8000 cliplink-backend

# Run with environment variables
docker run -p 8000:8000 \
  -e HF_TOKEN=your_token_here \
  -e MAX_WORKERS=4 \
  -v $(pwd)/clips:/app/clips \
  -v $(pwd)/downloads:/app/downloads \
  cliplink-backend

# Run in background
docker run -d -p 8000:8000 \
  --name cliplink-app \
  -e HF_TOKEN=your_token_here \
  cliplink-backend
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | - | **Required**: HuggingFace API token |
| `MAX_WORKERS` | 4 | Number of worker threads |
| `MAX_CONCURRENT_TASKS` | 10 | Max concurrent video processing tasks |
| `HOST` | 0.0.0.0 | Server host |
| `PORT` | 8000 | Server port |
| `DEBUG` | false | Enable debug mode |

### Volume Mounts

The following directories should be mounted for persistent storage:

```yaml
volumes:
  - ./clips:/app/clips              # Processed video clips
  - ./downloads:/app/downloads      # Downloaded videos
  - ./temp_uploads:/app/temp_uploads # Temporary uploads
  - ./temp_vertical:/app/temp_vertical # Temporary vertical crops
  - ./models:/app/models            # AI model cache
```

## 🎯 API Features

### Advanced Video Processing Endpoints

1. **Advanced Vertical Cropping**
   ```bash
   POST /workflow/create-vertical-crop-advanced
   ```
   - Uses pyannote.audio for speaker diarization
   - MediaPipe + MTCNN for face detection
   - PySceneDetect for scene change detection
   - Intelligent transition management

2. **Standard Vertical Cropping**
   ```bash
   POST /workflow/create-vertical-crop-async
   ```
   - Async processing with progress tracking
   - Basic speaker detection

3. **Complete Workflow**
   ```bash
   POST /workflow/process-complete-async
   ```
   - YouTube download → transcript → AI analysis → clip generation

### Testing the API

```bash
# Test HuggingFace connection
curl -X POST "http://localhost:8000/workflow/test-hf-connection" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "hf_token=your_token_here"

# Health check
curl http://localhost:8000/workflow/health

# Get task status
curl http://localhost:8000/workflow/task-status/{task_id}
```

## 🏗️ Production Deployment

### Docker Compose Production

```yaml
version: '3.8'
services:
  cliplink-backend:
    image: cliplink-backend:latest
    container_name: cliplink-prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - HF_TOKEN=${HF_TOKEN}
    volumes:
      - clips-data:/app/clips
      - downloads-data:/app/downloads
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'

volumes:
  clips-data:
  downloads-data:
```

### GPU Support (Optional)

For faster AI processing with GPU support:

```dockerfile
# Modify Dockerfile to use CUDA base image
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Add to docker-compose.yml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## 🔍 Monitoring and Logs

### View Logs

```bash
# Docker Compose logs
docker-compose logs -f

# Container logs
docker logs -f cliplink-app

# Specific service logs
docker-compose logs -f cliplink-backend
```

### Health Monitoring

The container includes built-in health checks:

```bash
# Check container health
docker ps

# Manual health check
docker exec cliplink-app curl -f http://localhost:8000/workflow/health
```

## 🛠️ Development

### Development with Volume Mounts

```yaml
# docker-compose.override.yml
version: '3.8'
services:
  cliplink-backend:
    volumes:
      - ./app:/app/app  # Mount source code
    environment:
      - DEBUG=true
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Debugging

```bash
# Run container with shell access
docker run -it --entrypoint /bin/bash cliplink-backend

# Execute commands in running container
docker exec -it cliplink-app /bin/bash

# View Python environment
docker exec cliplink-app pip list
```

## 📊 Resource Requirements

### Minimum System Requirements

- **CPU**: 2 cores
- **Memory**: 4GB RAM
- **Storage**: 10GB free space
- **Network**: Internet connection for AI model downloads

### Recommended for Production

- **CPU**: 4+ cores
- **Memory**: 8GB+ RAM
- **Storage**: 50GB+ SSD
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional)

## 🔒 Security Notes

1. **HuggingFace Token**: Keep your token secure, don't commit to version control
2. **Network**: Configure CORS properly for production
3. **User**: Container runs as non-root user for security
4. **Volumes**: Set appropriate permissions on mounted volumes

## 🐛 Troubleshooting

### Common Issues

1. **HuggingFace Token Issues**
   ```bash
   # Test token validity
   curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://huggingface.co/api/whoami
   ```

2. **Memory Issues**
   ```bash
   # Increase Docker memory limit
   # Docker Desktop: Settings → Resources → Memory
   ```

3. **Port Conflicts**
   ```bash
   # Use different port
   docker run -p 8080:8000 cliplink-backend
   ```

4. **Permission Issues**
   ```bash
   # Fix volume permissions
   sudo chown -R $USER:$USER ./clips ./downloads
   ```

### Debug Mode

Enable debug logging:

```bash
docker run -p 8000:8000 \
  -e DEBUG=true \
  -e LOG_LEVEL=DEBUG \
  cliplink-backend
```

## 📝 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyAnnote Audio](https://github.com/pyannote/pyannote-audio)
- [MediaPipe](https://mediapipe.dev/)
- [PySceneDetect](https://scenedetect.com/)
- [Docker Documentation](https://docs.docker.com/)

## 🤝 Support

For issues and questions:
1. Check the logs: `docker-compose logs -f`
2. Verify your HuggingFace token
3. Ensure all required dependencies are installed
4. Check system resources (CPU, memory, disk space) 