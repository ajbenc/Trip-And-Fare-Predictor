# 🐳 Docker Documentation

Complete documentation for deploying the NYC Taxi Prediction App with Docker.

## 📚 Documentation Index

### Getting Started
- **[GUIDE.md](GUIDE.md)** - Complete Docker deployment guide
  - Understanding Docker concepts
  - Installation instructions (Windows/Mac/Linux)
  - Step-by-step setup
  - Multi-container architecture explanation

### Quick Reference
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Common Docker commands
  - Container management
  - Image operations
  - Debugging commands
  - Cleanup operations

### Troubleshooting
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Solve common issues
  - Windows-specific problems
  - Port conflicts
  - Network issues
  - Build failures

### Deployment
- **[DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)** - Production readiness
  - Pre-deployment verification
  - Sharing with team members
  - Testing checklist
  - Production deployment notes

---

## 🚀 Quick Start

```bash
# From project root directory
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

Access:
- **Frontend UI**: [http://localhost:8501](http://localhost:8501)
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📁 Project Structure

```
Taxi Trips ML Final project/
├── Dockerfile.backend         # FastAPI backend container (production)
├── Dockerfile.frontend        # Streamlit frontend container (production)
├── docker-compose.yml         # Local development orchestration
├── docker-compose.prod.yml    # Production orchestration (scaling, restart)
├── .dockerignore              # Files excluded from build
├── nginx.conf                 # Load balancer config
├── docs/
│   └── docker/                # Docker documentation (you are here!)
│       ├── README.md          # This file
│       ├── GUIDE.md           # Complete guide
│       ├── QUICK_REFERENCE.md # Command cheat sheet
│       ├── TROUBLESHOOTING.md # Problem solving
│       └── DEPLOYMENT_CHECKLIST.md # Production readiness
├── src/                       # Application code
├── models/                    # Trained ML models
└── Data/                      # Taxi zone data
```

---

## 💡 Architecture Overview

```
┌─────────────────────────────────────────────────┐
│      Docker Compose Network (taxi-network)      │
│                                                 │
│  ┌──────────────────┐  ┌────────────────────┐   │
│  │  backend         │  │  frontend          │   │
│  │  (FastAPI)       │◄─┤  (Streamlit)       │   │
│  │  Port: 8000      │  │  Port: 8501        │   │
│  └──────────────────┘  └────────────────────┘   │
│                                                 │
└─────────────────────────────────────────────────┘
     ↓                        ↓
   localhost:8000          localhost:8501
```

**Multi-Container Benefits:**
- ✅ Independent scaling
- ✅ Isolated failures
- ✅ Easy updates
- ✅ Production-ready

---

## 🆘 Need Help?

1. **Common Issues**: Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. **Commands**: See [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
3. **Full Guide**: Read [GUIDE.md](GUIDE.md)
4. **Deployment**: Review [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)

---

**Status**: ✅ Production Ready | **Last Updated**: November 11, 2025
