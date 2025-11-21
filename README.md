# Wat is mijn straat? 🏘️

 **🌐 Live op Railway:** [micro-neighbourhood-production-4237.up.railway.app](https://micro-neighbourhood-production-4237.up.railway.app/)

**In één scherm:** adres, pand, buurtcijfers, AI-buurtverhaal, cluster classificatie en vergelijkbare buurten via machine learning!

Een moderne webapp die Nederlandse wijken analyseert met PDOK adressen, CBS demografische data, KMeans clustering van alle 5000+ buurten, KNN buurtvergelijking en AI-gestuurde verhalen.

## ✨ Features

### 🔍 Adres Zoeken
- PDOK Locatieserver integratie voor nauwkeurige adreszoekfunctie
- Nederlandse adressen met autocomplete ondersteuning
- BAG pand-geometrie visualisatie

### 📊 Uitgebreide Buurt Statistieken
- CBS Kerncijfers Wijken en Buurten (volledige dataset)
- Bevolking, dichtheid, leeftijdsgroepen, inkomen, huishoudens
- Huishoudsamenstelling, woningtypes, voorzieningen afstanden
- Nederlandse locale formatting en visualisaties

### 🗺️ Moderne Interactieve Kaart
- MapLibre GL integratie met OpenStreetMap
- Markers op exacte adressen + BAG pand polygons
- Zoom, pan en geolocatie controls
- Moderne state-of-the-art mapping experience

### 🤖 AI & Machine Learning
- **OpenAI GPT-4 integratie** voor buurtbeschrijvingen
- **KMeans clustering** van alle Nederlandse buurten (8 clusters)
- **KNN recommender** voor vergelijkbare buurten
- **LLM-gegenereerde labels** voor cluster interpretatie
- Persoonlijke verhalen + data-driven inzichten

### 📈 Data Visualisaties
- **Leeftijdsverdelingscharts** (Highcharts)
- **Inkomensverdelingsanalyse** (Highcharts)
- **Cluster classificatie** met begrijpelijke labels
- **Vergelijkbare buurten lijst** met bevolkingsdata

## Tech Stack

### Frontend
- React 18 - Moderne UI componenten
- Vite - Snelle development server en build tool
- MapLibre GL - Moderne interactieve kaarten
- Highcharts - Professionele data visualisaties
- React Markdown - Mooie AI content rendering
- CSS Grid/Flexbox - Moderne responsive layouts

### Backend
- FastAPI - Moderne Python web framework
- OpenAI API - AI taalmodel integratie
- Scikit-learn - Machine learning (KMeans, KNN)
- Pandas/GeoPandas - Data processing
- Uvicorn - ASGI server voor productie
- Python 3.11 - Docker deployment

### Data Bronnen & ML
- **PDOK Locatieserver** - Adres geocoding + BAG pand geometrie
- **CBS StatLine OData** - Uitgebreide demografische data (11 features)
- **KMeans Clustering** - 8 buurt-types gebaseerd op socio-demografische data
- **LLM Labeling** - AI-g gegenereerde begrijpelijke cluster beschrijvingen
- **KNN Recommender** - Vergelijkbare buurten gebaseerd op feature similarity

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

### GET /api/similar-buurten
Vindt vergelijkbare buurten via KNN machine learning.

Parameters:
- `buurt_code`: CBS buurtcode (bijv. BU05990110)
- `k`: Aantal resultaten (1-10)

### GET /api/buurt-cluster
Geeft cluster informatie voor een buurt.

Parameters:
- `buurt_code`: CBS buurtcode

Response:
```json
{
  "buurt_code": "BU05990110",
  "cluster": 3,
  "label_short": "jong & stedelijk",
  "label_long": "Drukke binnenstad met veel jonge volwassenen..."
}
```

### GET /api/health
Health check endpoint.

## 🚀 Live Deployment

**🌐 URL:** [micro-neighbourhood-production-4237.up.railway.app](https://micro-neighbourhood-production-4237.up.railway.app)

**📅 Laatste update:** November 2025

**✨ Wat is nieuw:** Volledige ML integratie met KNN buurtvergelijking en KMeans clustering!

### Docker Deployment op Railway

De applicatie gebruikt Docker voor consistente deployment:

1. **Repository Structure**
   ```
   ├── Dockerfile          # Multi-stage build (Node + Python)
   ├── frontend/           # React app + MapLibre + Highcharts
   ├── backend/            # FastAPI backend + ML
   │   ├── offline/        # ML preprocessing scripts
   │   └── data/           # Precomputed ML data (CSV)
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

2. **ML Data Preprocessing** (Optioneel - gebruikt precomputed data)
   ```bash
   cd backend
   # Zorg voor OPENAI_API_KEY in .env
   python -m offline.build_clusters  # Genereert ML modellen + labels
   ```

3. **Backend**
   ```bash
   cd backend
   python3 -m pip install -r requirements.txt
   ```

4. **Frontend**
   ```bash
   cd frontend
   npm install
   npm run build
   ```

5. **Start**
   ```bash
   cd backend
   python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```

6. **Open:** http://localhost:8000
