# Micro-Neighbourhood Insights

 **Live op Railway:** [micro-neighbourhood-production.up.railway.app](https://micro-neighbourhood-production.up.railway.app)

Een moderne webapplicatie die inzichten geeft in Nederlandse wijken en buurten door data te combineren van PDOK, CBS en AI-gestuurde analyses.

## ✨ Features

### 🔍 Adres Zoeken
- PDOK Locatieserver integratie voor nauwkeurige adreszoekfunctie
- Nederlandse adressen met autocomplete ondersteuning

### 📊 Buurt Statistieken
- CBS Kerncijfers Wijken en Buurten
- Bevolkingsaantallen (momenteel beschikbaar)
- Uitbreidbaar naar dichtheid, leeftijd, inkomen
- Nederlandse locale formatting

### 🗺️ Interactieve Kaart
- Leaflet.js integratie met OpenStreetMap
- Markers op exacte adressen
- Zoom en pan functionaliteit

### 🤖 AI Buurtverhaal
- OpenAI GPT-4 integratie
- Persoonlijke buurtbeschrijvingen in markdown
- Mooie rendering met headers, lijsten en opmaak
- Adaptief taalgebruik per doelgroep

## Tech Stack

### Frontend
- React 18 - Moderne UI componenten
- Vite - Snelle development server en build tool
- Leaflet - Interactieve kaarten
- React Markdown - Mooie AI content rendering
- CSS Modules - Scoped styling

### Backend
- FastAPI - Moderne Python web framework
- OpenAI API - AI taalmodel integratie
- Uvicorn - ASGI server voor productie
- Python 3.11 - Docker deployment

### Data Bronnen
- PDOK Locatieserver - Adres geocoding
- CBS StatLine OData - Demografische data

### Deployment
- Docker - Containerized deployment
- Railway - Cloud hosting platform
- GitHub - Version control en CI/CD

## 🛠️ Lokale Development

### Prerequisites
- Python 3.8+
- Node.js 18+
- OpenAI API key

### Quick Start

1. **Clone & Setup**
   ```bash
   git clone https://github.com/dat-nguyen96/Micro-Neighbourhood.git
   cd Micro-Neighbourhood
   ```

2. **Backend Dependencies**
   ```bash
   cd backend
   python3 -m pip install -r requirements.txt
   # Voeg OPENAI_API_KEY toe aan environment
   ```

3. **Frontend Build**
   ```bash
   cd frontend
   npm install
   npm run build
   ```

4. **Start Server**
   ```bash
   cd backend
   python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```

5. **Open:** http://localhost:8000

## API Endpoints

### POST /api/neighbourhood-story
Genereert een AI buurtbeschrijving.

Request:
```json
{
  "data": {
    "address": "Damrak 1, Amsterdam",
    "buildingInfo": {...},
    "cbsStats": {...}
  },
  "persona": "algemeen huishouden"
}
```

### GET /api/health
Health check endpoint.

## 🚀 Live Deployment

**🌐 URL:** [micro-neighbourhood-production.up.railway.app](https://micro-neighbourhood-production.up.railway.app)

**📅 Laatste update:** November 2025

### Docker Deployment op Railway

De applicatie gebruikt Docker voor consistente deployment:

1. **Repository Structure**
   ```
   ├── Dockerfile          # Multi-stage build (Node + Python)
   ├── frontend/           # React app
   ├── backend/            # FastAPI backend
   └── railway.toml        # Railway config
   ```

2. **Docker Build Process**
   - **Stage 1:** Node.js container bouwt React app
   - **Stage 2:** Python container serveert FastAPI + static files
   - Automatische Railway deployment via GitHub

3. **Environment Variables**
   ```
   OPENAI_API_KEY=sk-...jouw_key...
   OPENAI_MODEL=gpt-4o-mini
   ```

### Lokale Development

1. **Clone & Setup**
   ```bash
   git clone https://github.com/dat-nguyen96/Micro-Neighbourhood.git
   cd Micro-Neighbourhood
   ```

2. **Backend**
   ```bash
   cd backend
   python3 -m pip install -r requirements.txt
   ```

3. **Frontend**
   ```bash
   cd frontend
   npm install
   npm run build
   ```

4. **Start**
   ```bash
   cd backend
   python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```

5. **Open:** http://localhost:8000