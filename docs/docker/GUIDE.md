# 🐳 Docker Deployment Guide - NYC Taxi Prediction App

## 📋 Table of Contents
1. [Understanding Docker](#understanding-docker)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Detailed Instructions](#detailed-instructions)
5. [Troubleshooting](#troubleshooting)

---

## 🎓 Understanding Docker

### What is Docker?
Docker packages your application and all its dependencies into a **container** - a lightweight, standalone package that runs consistently anywhere.

### Why Docker for this project?
- ✅ **Consistency**: Works the same on Windows, Mac, Linux
- ✅ **No dependency issues**: All libraries bundled inside
- ✅ **Easy deployment**: One command to run entire app
- ✅ **Isolation**: Doesn't affect your system's Python/packages

### Our Docker Setup
We use a **multi-container** approach with Docker Compose:
```
┌─────────────────────────────────────────────────────┐
│         Docker Compose Network (taxi-network)       │
│                                                      │
│  ┌────────────────────────┐  ┌───────────────────┐ │
│  │  Backend Container     │  │ Frontend Container│ │
│  │  (nyc-taxi-backend)    │  │ (nyc-taxi-frontend)│ │
│  │                        │  │                   │ │
│  │  FastAPI Server        │◄─┤  Streamlit UI     │ │
│  │  Port: 8000           │  │  Port: 8501       │ │
│  │  - /predict endpoint   │  │  - Interactive Map│ │
│  │  - /health endpoint    │  │  - Trip Predictor │ │
│  │  - /docs (Swagger)     │  │                   │ │
│  └────────────────────────┘  └───────────────────┘ │
│            │                                         │
│  ┌─────────▼──────────────────────────────────────┐ │
│  │  Shared Resources                              │ │
│  │  - Trained Models (duration_model, fare_model)│ │
│  │  - Taxi Zone Data (GeoPandas shapefiles)      │ │
│  │  - Logs Volume (persistent)                   │ │
│  └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘

Your Computer:
  http://localhost:8000  →  Backend API
  http://localhost:8501  →  Frontend UI
```

**Benefits of Multi-Container:**
- ✅ **Independent Scaling**: Scale backend and frontend separately
- ✅ **Isolated Failures**: Frontend crash doesn't affect backend
- ✅ **Easy Updates**: Update one service without rebuilding both
- ✅ **Better Resource Management**: Allocate resources per container
- ✅ **Production-Ready**: Matches real-world deployment patterns

---

## 📦 Prerequisites

### 1. Install Docker Desktop

**Windows:**
1. Download from: https://www.docker.com/products/docker-desktop
2. Run installer (requires Windows 10/11 Pro, Enterprise, or Education)
3. Restart computer when prompted
4. Open Docker Desktop - wait for "Docker is running" message

**Mac:**
1. Download from: https://www.docker.com/products/docker-desktop
2. Drag to Applications folder
3. Launch Docker Desktop

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
```

### 2. Verify Installation
```bash
# Check Docker version
docker --version
# Should show: Docker version 24.x.x or higher

# Check Docker Compose
docker-compose --version
# Should show: Docker Compose version 2.x.x or higher

# Test Docker is running
docker run hello-world
# Should download and run a test container
```

---

## 🚀 Quick Start

### Method 1: Using Docker Compose (Easiest)

```bash
# Navigate to project directory
cd "C:\Users\Julian\OneDrive\Desktop\Taxi Trips ML Final project"

# Build and start the container
docker-compose up --build

# Wait for: "Application is ready!" message
# Then open your browser:
#   - Streamlit UI: http://localhost:8501
#   - API Docs: http://localhost:8000/docs
```

**To stop:**
Press `Ctrl+C` in terminal, then:
```bash
docker-compose down
```

### Method 2: Using Docker Commands Directly

```bash
# Build both images
docker build -f Dockerfile.backend -t nyc-taxi-backend .
docker build -f Dockerfile.frontend -t nyc-taxi-frontend .

# Create a network for inter-container communication
docker network create taxi-network

# Run the backend container
docker run -d \
  --name nyc-taxi-backend \
  --network taxi-network \
  -p 8000:8000 \
  nyc-taxi-backend

# Run the frontend container
docker run -d \
  --name nyc-taxi-frontend \
  --network taxi-network \
  -p 8501:8501 \
  -e API_URL=http://nyc-taxi-backend:8000 \
  nyc-taxi-frontend

# Check logs
docker logs -f nyc-taxi-backend
docker logs -f nyc-taxi-frontend

# Stop containers
docker stop nyc-taxi-backend nyc-taxi-frontend
docker rm nyc-taxi-backend nyc-taxi-frontend
```

---

## 📖 Detailed Instructions

### Step 1: Prepare Your Project

Make sure you have these files in your project root:
```
Taxi Trips ML Final project/
├── Dockerfile              ✅ (just created)
├── .dockerignore          ✅ (just created)
├── docker-compose.yml     ✅ (just created)
├── requirements.txt       ✅ (already exists)
├── start_all.py          ✅ (already exists)
├── src/                   ✅ (your code)
├── models/                ✅ (your trained models)
└── Data/                  ✅ (taxi zones, etc.)
```

### Step 2: Build the Docker Images

This creates the container images for backend and frontend:

```bash
# Using Docker Compose (builds both)
docker-compose build

# OR using Docker directly (build each separately)
docker build -f Dockerfile.backend -t nyc-taxi-backend .
docker build -f Dockerfile.frontend -t nyc-taxi-frontend .
```

**What's happening:**
1. Downloads Python 3.11 base image (~100 MB)
2. Installs system dependencies (gcc, g++, curl)
3. Installs Python packages from requirements.txt
4. Copies your application code
5. Creates two separate images:
   - `nyc-taxi-backend` - FastAPI server (~1.5 GB)
   - `nyc-taxi-frontend` - Streamlit app (~1.5 GB)

**This takes 5-10 minutes first time** (downloads dependencies)

### Step 3: Run the Containers

```bash
# Using Docker Compose (recommended - starts both containers)
docker-compose up

# Add -d flag to run in background
docker-compose up -d

# OR using Docker directly (manual network setup)
docker network create taxi-network
docker run -d --name nyc-taxi-backend --network taxi-network -p 8000:8000 nyc-taxi-backend
docker run -d --name nyc-taxi-frontend --network taxi-network -p 8501:8501 nyc-taxi-frontend
```

**What's happening:**
1. Creates a Docker network for inter-container communication
2. Starts backend container (FastAPI on port 8000)
3. Waits for backend health check to pass
4. Starts frontend container (Streamlit on port 8501)
5. Frontend connects to backend via internal network

**Startup sequence (~30-60 seconds):**
- Backend: 10-20 seconds (loads models)
- Frontend: 15-30 seconds (waits for backend, initializes UI)

### Step 4: Access the Application

Open your browser:
- **Streamlit UI**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

### Step 5: View Logs

```bash
# Using Docker Compose (both containers)
docker-compose logs -f

# View specific container logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Using Docker directly
docker logs -f nyc-taxi-backend
docker logs -f nyc-taxi-frontend
```

### Step 6: Stop the Application

```bash
# Using Docker Compose
docker-compose down

# Using Docker directly
docker stop nyc-taxi-backend nyc-taxi-frontend
docker rm nyc-taxi-backend nyc-taxi-frontend
docker network rm taxi-network
```

---

## 🔧 Common Docker Commands

### Managing Containers
```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Start stopped containers
docker-compose start

# Stop running containers (without removing)
docker-compose stop

# Restart containers
docker-compose restart

# Restart specific service
docker-compose restart backend
docker-compose restart frontend

# Remove containers
docker-compose down

# Remove containers and volumes
docker-compose down -v
```

### Managing Images
```bash
# List images
docker images

# Remove specific images
docker rmi nyc-taxi-backend
docker rmi nyc-taxi-frontend

# Rebuild images (after code changes)
docker-compose build --no-cache

# Rebuild specific service
docker-compose build --no-cache backend
docker-compose build --no-cache frontend

# Clean up unused images
docker image prune

# Remove all unused images, containers, networks
docker system prune -a
```

### Debugging
```bash
# Execute command inside running container
docker exec -it nyc-taxi-backend bash
docker exec -it nyc-taxi-frontend bash

# View real-time logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Inspect container details
docker inspect nyc-taxi-backend
docker inspect nyc-taxi-frontend

# Check resource usage
docker stats

# Test network connectivity between containers
docker exec nyc-taxi-frontend curl http://backend:8000/health

# Check container health status
docker ps --filter "name=nyc-taxi" --format "table {{.Names}}\t{{.Status}}"
```

---

## 🐛 Troubleshooting

### Problem: "Docker daemon is not running"
**Solution:**
- Windows/Mac: Open Docker Desktop and wait for it to start
- Linux: `sudo systemctl start docker`

### Problem: "Port already in use"
**Solution:**
```bash
# Find what's using port 8000 or 8501
netstat -ano | findstr :8000
netstat -ano | findstr :8501

# Stop the process or change ports in docker-compose.yml
ports:
  - "8002:8000"  # Use port 8002 on host instead
  - "8502:8501"  # Use port 8502 on host instead
```

### Problem: "Cannot connect to Docker daemon"
**Solution:**
- Ensure Docker Desktop is running
- Try restarting Docker Desktop
- Windows: Check WSL2 is installed and updated

### Problem: Build fails with "No space left on device"
**Solution:**
```bash
# Clean up Docker
docker system prune -a
docker volume prune
```

### Problem: Container exits immediately
**Solution:**
```bash
# Check logs for errors
docker-compose logs backend
docker-compose logs frontend

# Common issues:
# - Backend: Missing models/ folder, port conflict
# - Frontend: Can't connect to backend (check API_URL)
# - Both: Missing Data/taxi_zones/ folder
```

### Problem: Frontend can't connect to backend
**Solution:**
```bash
# Test backend health from frontend container
docker exec nyc-taxi-frontend curl http://backend:8000/health

# Check if backend is running
docker ps | grep backend

# Verify network connection
docker network inspect taxi-network

# Restart both services
docker-compose restart
```

### Problem: Only one container starts
**Solution:**
```bash
# Check depends_on configuration
docker-compose config

# Start services individually
docker-compose up backend
# Wait for healthy, then:
docker-compose up frontend
```

### Problem: Slow build time
**Solution:**
```bash
# Use BuildKit for faster builds
set DOCKER_BUILDKIT=1  # Windows
export DOCKER_BUILDKIT=1  # Mac/Linux

docker-compose build
```

### Problem: Changes not reflected in container
**Solution:**
```bash
# Rebuild without cache
docker-compose build --no-cache
docker-compose up -d
```

---

## 📊 Image Size Optimization

Our Docker images include:

**Backend Container (~1.5 GB):**
- Python 3.11 base (~100 MB)
- System dependencies (~50 MB)
- Python packages (~500 MB)
- Application code (~50 MB)
- Models (~100 MB)
- Data files (taxi zones, ~20 MB)

**Frontend Container (~1.5 GB):**
- Python 3.11 base (~100 MB)
- System dependencies (~50 MB)
- Python packages (Streamlit, Folium, ~500 MB)
- Application code (~50 MB)
- Models (~100 MB) - needed for zone calculations
- Data files (taxi zones, ~20 MB)

**Total disk space: ~3 GB** (both containers)

**Optimization Strategies:**
1. ✅ Use `.dockerignore` (already configured)
2. ✅ Share base layers between containers
3. ✅ Multi-stage builds for smaller images
4. Consider separate data volume (mount instead of copy)

---

## 🚀 Production Deployment

### Scaling with Docker Compose

**Scale backend for high traffic:**
```bash
# Run multiple backend instances (load balancing)
docker-compose up --scale backend=3

# Docker will create:
# - nyc-taxi-backend-1 (port 8000)
# - nyc-taxi-backend-2 (random port)
# - nyc-taxi-backend-3 (random port)

# Add nginx for load balancing (see nginx.conf example below)
```

**Example nginx configuration for load balancing:**
```nginx
upstream backend {
    server nyc-taxi-backend-1:8000;
    server nyc-taxi-backend-2:8000;
    server nyc-taxi-backend-3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://backend;
    }
}
```

### Deploy to Cloud

**AWS ECS (Elastic Container Service):**
```bash
# Build and push to Amazon ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push backend
docker tag nyc-taxi-backend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/nyc-taxi-backend:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/nyc-taxi-backend:latest

# Tag and push frontend
docker tag nyc-taxi-frontend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/nyc-taxi-frontend:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/nyc-taxi-frontend:latest

# Create ECS task definition with both services
```

**Google Cloud Run:**
```bash
# Build and deploy backend
gcloud builds submit --tag gcr.io/<project-id>/nyc-taxi-backend ./Dockerfile.backend
gcloud run deploy nyc-taxi-backend --image gcr.io/<project-id>/nyc-taxi-backend --platform managed --port 8000

# Build and deploy frontend
gcloud builds submit --tag gcr.io/<project-id>/nyc-taxi-frontend ./Dockerfile.frontend
gcloud run deploy nyc-taxi-frontend --image gcr.io/<project-id>/nyc-taxi-frontend --platform managed --port 8501 --set-env-vars API_URL=<backend-url>
```

**Azure Container Instances:**
```bash
# Create resource group
az group create --name nyc-taxi-rg --location eastus

# Push to Azure Container Registry
az acr build --registry <registry-name> --image nyc-taxi-backend:latest -f Dockerfile.backend .
az acr build --registry <registry-name> --image nyc-taxi-frontend:latest -f Dockerfile.frontend .

# Deploy with Azure Container Instances
az container create --resource-group nyc-taxi-rg --name backend --image <registry-name>.azurecr.io/nyc-taxi-backend:latest --ports 8000
az container create --resource-group nyc-taxi-rg --name frontend --image <registry-name>.azurecr.io/nyc-taxi-frontend:latest --ports 8501
```

### Kubernetes Deployment (Advanced)

For large-scale production, use Kubernetes:

```yaml
# backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nyc-taxi-backend
spec:
  replicas: 3  # Run 3 backend pods
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: nyc-taxi-backend:latest
        ports:
        - containerPort: 8000
---
# frontend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nyc-taxi-frontend
spec:
  replicas: 2  # Run 2 frontend pods
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
      - name: frontend
        image: nyc-taxi-frontend:latest
        ports:
        - containerPort: 8501
        env:
        - name: API_URL
          value: "http://backend-service:8000"
```

### Environment Variables for Production
For production, use environment variables for configuration:

```bash
# In docker-compose.yml
environment:
  - API_HOST=0.0.0.0
  - API_PORT=8000
  - STREAMLIT_PORT=8501
  - LOG_LEVEL=INFO
```

---

## 📚 Next Steps

1. **Test locally**: `docker-compose up`
2. **Share image**: Push to Docker Hub or private registry
3. **Deploy**: Use cloud provider's container service
4. **Monitor**: Add logging and monitoring tools
5. **Scale**: Use Kubernetes for multi-replica deployment

---

## 💡 Tips for Beginners

1. **Always check logs** when something doesn't work: `docker logs -f nyc-taxi`
2. **Rebuild after code changes**: `docker-compose build --no-cache`
3. **Clean up regularly**: `docker system prune` removes unused containers/images
4. **Use Docker Desktop UI**: Visual way to manage containers (easier than CLI)
5. **Port conflicts**: If 8000/8501 are busy, change ports in docker-compose.yml

---

## 🆘 Getting Help

- Docker Documentation: https://docs.docker.com/
- Docker Desktop Forum: https://forums.docker.com/
- Stack Overflow: Tag your questions with `docker`

---

**Your Docker setup is ready! Run `docker-compose up` to start your containerized NYC Taxi app! 🚕🐳**
