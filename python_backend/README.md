# Docker Deployment Guide

## Backend-Only Docker Setup (Recommended)

This setup containerizes the Python API while running Next.js locally for easier camera access.

### Prerequisites

```bash
# Install Docker
# - Windows/Mac: Docker Desktop
# - Linux: Docker Engine

docker --version  # Should be 20.10+
```

### Build and Run

```bash
# Navigate to python_backend directory
cd python_backend

# Build the Docker image
docker build -t face-recognition-api .

# Run the container
docker run -d \
  --name face-api \
  -p 8111:8111 \
  -v $(pwd)/../models:/app/models \
  face-recognition-api
```

**For Windows PowerShell:**
```powershell
docker run -d `
  --name face-api `
  -p 8111:8111 `
  -v ${PWD}/../models:/app/models `
  face-recognition-api
```

### With GPU Support (NVIDIA)

```bash
# Requires nvidia-docker installed
docker run -d \
  --name face-api \
  --gpus all \
  -p 8111:8111 \
  -v $(pwd)/../models:/app/models \
  face-recognition-api
```

### Verify API is Running

```bash
# Check logs
docker logs face-api

# Test health endpoint
curl http://localhost:8111/health

# Or open in browser
# http://localhost:8111/test
```

### Run Frontend Locally

```bash
# In a separate terminal
cd nextjs-frontend

# Install and run
npm install
npm run dev
```

Access the app at `http://localhost:3000`

### Docker Commands

```bash
# Stop container
docker stop face-api

# Start container
docker start face-api

# Restart container
docker restart face-api

# View logs
docker logs -f face-api

# Remove container
docker rm -f face-api

# Remove image
docker rmi face-recognition-api
```

### Environment Variables

Create `.env` file in `python_backend/`:

```env
MODEL_PATH=/app/models/best_facenet_model.pt
CONFIDENCE_THRESHOLD=0.5
HOST=0.0.0.0
PORT=8111
```

Run with environment file:
```bash
docker run -d \
  --name face-api \
  -p 8111:8111 \
  -v $(pwd)/../models:/app/models \
  --env-file .env \
  face-recognition-api
```

### Troubleshooting

**Issue: Model file not found**
```bash
# Verify volume mount
docker exec face-api ls -la /app/models

# Copy model into container (alternative)
docker cp ../models/best_facenet_model.pt face-api:/app/models/
```

**Issue: Permission denied**
```bash
# Fix permissions on host
chmod -R 755 ../models

# Rebuild with correct user
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t face-recognition-api .
```

**Issue: API not responding**
```bash
# Check if container is running
docker ps

# Check container health
docker inspect --format='{{.State.Health.Status}}' face-api

# Enter container to debug
docker exec -it face-api bash
```

### Production Deployment

```bash
# Build optimized image
docker build -t face-recognition-api:prod --target production .

# Run in production mode
docker run -d \
  --name face-api-prod \
  --restart unless-stopped \
  -p 8111:8111 \
  -v /path/to/models:/app/models \
  face-recognition-api:prod
```

### Docker Compose (Optional)

Create `docker-compose.yml` in project root:

```yaml
version: '3.8'

services:
  backend:
    build: ./python_backend
    container_name: face-api
    ports:
      - "8111:8111"
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8111/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Run with Docker Compose:
```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```