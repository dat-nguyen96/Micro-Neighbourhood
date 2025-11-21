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

# Basis tools + libs voor geopandas (gdal, spatialindex, proj)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    proj-bin \
    proj-data \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend ./backend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

WORKDIR /app/backend

RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
