# 🐳 Docker Quick Reference - NYC Taxi App (Multi-Container)

## ⚡ Quick Commands (Most Used)

```bash
# Start both backend and frontend
docker-compose up

# Start in background
docker-compose up -d

# Stop both containers
docker-compose down

# View logs (both containers)
docker-compose logs -f

# View logs for specific service
docker-compose logs -f backend
docker-compose logs -f frontend

# Rebuild after code changes
docker-compose build --no-cache
docker-compose up

# Restart specific service
docker-compose restart backend
docker-compose restart frontend
```

---

## 📋 Checklist for First Time

- [ ] Install Docker Desktop
- [ ] Verify: `docker --version`
- [ ] Verify: `docker-compose --version`
- [ ] Navigate to project folder
- [ ] Run: `docker-compose up`
- [ ] Wait for both containers to be healthy
- [ ] Open backend: http://localhost:8000/docs
- [ ] Open frontend: http://localhost:8501

---

## 🏗️ Architecture Overview

```
Frontend (Streamlit)  →  Backend (FastAPI)
    :8501            →      :8000
    
- Frontend depends on backend
- Backend starts first (health check)
- Frontend connects via Docker network
```

---

## 🔥 Common Issues

| Problem | Solution |
|---------|----------|
| Port already in use | Change ports in `docker-compose.yml` |
| Backend exits immediately | Check logs: `docker-compose logs backend` |
| Frontend can't connect | Test: `docker exec nyc-taxi-frontend curl http://backend:8000/health` |
| Changes not showing | Rebuild: `docker-compose build --no-cache` |
| Out of space | Clean up: `docker system prune -a` |
| Only one container starts | Check: `docker-compose ps` |

---

## 🛠️ Useful Commands

```bash
# See running containers
docker-compose ps

# Access container shell
docker exec -it nyc-taxi-backend bash
docker exec -it nyc-taxi-frontend bash

# Check resource usage
docker stats

# Scale backend (multiple instances)
docker-compose up --scale backend=3

# Remove everything and start fresh
docker-compose down -v
docker system prune -a
docker-compose up --build
```

---

## 🔍 Debugging Commands

```bash
# Test backend from frontend container
docker exec nyc-taxi-frontend curl http://backend:8000/health

# Check network connectivity
docker network inspect taxi-network

# View container details
docker inspect nyc-taxi-backend
docker inspect nyc-taxi-frontend

# Follow logs in real-time
docker-compose logs -f --tail=100
```

---

## 📚 Full Documentation

See [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for complete instructions with explanations!
