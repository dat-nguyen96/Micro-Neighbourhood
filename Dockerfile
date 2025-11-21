# ---------- STAGE 1: build frontend ----------
FROM node:18-bullseye AS frontend-builder

WORKDIR /app

# alleen frontend files kopiëren (sneller cachen)
COPY frontend ./frontend

WORKDIR /app/frontend
RUN npm install
RUN npm run build

# ---------- STAGE 2: backend + serve static ----------
FROM python:3.11-slim

# Zorg dat pip & basic tools aanwezig zijn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# backend code + requirements
COPY backend ./backend

# gekloonde frontend build op juiste plek:
# /app/frontend/dist (zodat jouw main.py het vindt)
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

WORKDIR /app/backend

# Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Externe wereld praat via $PORT (Railway zet die env var)
ENV PORT=8000

# Start FastAPI via uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
