# Wat is mijn straat? 🏘️

 **🌐 Live op Railway:** [micro-neighbourhood-production-4237.up.railway.app](https://micro-neighbourhood-production-4237.up.railway.app/)

**In één scherm:** adres, pand, buurtcijfers, AI-buurtverhaal, cluster classificatie, vergelijkbare buurten via machine learning, en directe buurtvergelijking met radar charts!

Een moderne webapp die Nederlandse wijken analyseert met PDOK adressen, CBS demografische data, KMeans clustering van alle 5000+ buurten, KNN buurtvergelijking, buurt-omlijning, en AI-gestuurde verhalen verrijkt met ML-context. Vergelijk twee buurten visueel met interactieve radar charts en sync zoom kaarten.

## ✨ Features

### 🔍 Intelligente Adres Zoeken
- **PDOK Locatieserver** integratie voor nauwkeurige adreszoekfunctie
- **Live autocomplete** met buurt namen in dropdown (bijv. "Damrak 1, Amsterdam • Stadsdriehoek")
- **Keyboard navigatie** (↑↓ Enter) voor snelle selectie
- **Direct zoeken** bij Enter toets of klik op suggestie
- BAG pand-geometrie visualisatie

### 📊 Uitgebreide Buurt Statistieken
- **CBS Kerncijfers Wijken en Buurten** (volledige dataset - 11+ features)
- Bevolking, dichtheid, leeftijdsgroepen, inkomen per persoon/huishouden
- Huishoudsamenstelling, woningtypes, stedelijkheid indicator
- **Voorzieningen afstanden:** Supermarkt, huisarts, school, kinderdagverblijf (in km)
- **CBS Politie Criminaliteit 2024** - Gewelds- en vermogensmisdrijven per 1000 inwoners
- Nederlandse locale formatting en visualisaties

### 🗺️ Moderne Interactieve Kaarten (Dubbel)
- **MapLibre GL** integratie met OpenStreetMap tiles
- **Dubbele kaart layout:** Hoofd buurt + vergelijkingsbuurt
- **Sync zoom & pan** tussen beide kaarten
- **Buurt-omlijning polygons** van PDOK WFS (hele buurtgrenzen)
- Markers op exacte adressen + BAG pand polygons
- Zoom, pan en geolocatie controls

### 🤖 AI & Machine Learning (Geavanceerd)
- **OpenAI GPT-4o-mini** integratie voor buurtbeschrijvingen met ML-context
- **KMeans clustering** van alle Nederlandse buurten (12 clusters)
- **KNN recommender** voor vergelijkbare buurten met buurt namen
- **LLM-gegenereerde labels** voor cluster interpretatie
- **AI verhalen verrijkt met cluster data** en vergelijkbare buurten
- Persoonlijke verhalen + data-driven inzichten

### 📈 Data Visualisaties (Uitgebreid)
- **Radar comparison chart** voor buurtvergelijking (6 dimensies)
- **Leeftijdsverdelingscharts** (Highcharts)
- **Inkomensverdelingsanalyse** (Highcharts)
- **PCA scatter plot** voor ML-ruimte visualisatie
- **Cluster classificatie** met begrijpelijke labels
- **Vergelijkbare buurten lijst** met echte buurt namen en bevolkingsdata

### 🔄 Directe Buurtvergelijking
- **Tweede zoekbalk** voor vergelijkingsbuurt
- **Side-by-side stats vergelijking** met verschil indicatoren
- **Visuele radar chart** voor multi-dimensionele vergelijking
- **Synchrone kaarten** met buurtgrenzen en markers
- **Realtime updates** bij nieuwe zoekopdrachten

## Tech Stack

### Frontend
- **React 18** - Moderne UI componenten met hooks
- **Vite** - Snelle development server met API proxy
- **MapLibre GL** - Moderne interactieve kaarten met sync zoom
- **Highcharts** - Professionele data visualisaties (radar, bar, scatter)
- **React Markdown** - AI content rendering
- **Custom AddressSearchBar** - Live autocomplete met buurt namen
- **CSS Grid/Flexbox** - Moderne responsive layouts

### Backend
- **FastAPI** - Moderne Python web framework met async endpoints
- **OpenAI GPT-4o-mini** - AI taalmodel voor buurtbeschrijvingen
- **Scikit-learn** - Machine learning (KMeans, KNN, StandardScaler)
- **Pandas/GeoPandas** - Data processing + CBS integratie
- **PDOK WFS client** - Buurt-omlijning polygons ophalen
- **Uvicorn** - ASGI server voor productie deployment

### Data Bronnen & ML Pipeline
- **PDOK Locatieserver** - Adres geocoding + BAG pand geometrie
- **PDOK WFS** - Buurt-omlijning polygons voor kaartvisualisatie
- **CBS StatLine OData** - Uitgebreide demografische data (15+ features)
- **CBS Politie 47018NED** - Criminaliteitscijfers 2024 per buurt
- **CBS Wijken&Buurten metadata** - Buurt namen voor gebruiksvriendelijkheid
- **KMeans Clustering** - 12 buurt-types gebaseerd op socio-demografische features
- **LLM Labeling** - AI-gegenereerde begrijpelijke cluster beschrijvingen
- **KNN Recommender** - Vergelijkbare buurten met echte buurt namen
- **PCA Visualisatie** - 2D ML-ruimte voor cluster understanding

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

**Development Note:** Frontend gebruikt Vite proxy om API calls naar backend te routeren.

## API Endpoints

### POST /api/neighbourhood-story
Genereert een AI buurtbeschrijving verrijkt met ML-context.

Request:
```json
{
  "data": {
    "address": "Damrak 1, Amsterdam",
    "buildingInfo": {...},
    "cbsStats": {...},
    "clusterInfo": {
      "label": "Jong & stedelijk",
      "label_long": "Drukke binnenstad met veel jonge volwassenen..."
    },
    "similarBuurten": {
      "neighbours": [...]
    }
  },
  "persona": "algemeen huishouden"
}
```

**Nieuwe feature:** Gebruikt cluster classificatie en vergelijkbare buurten voor contextrijke verhalen.

### GET /api/similar-buurten
Vindt vergelijkbare buurten via KNN machine learning.

Parameters:
- `buurt_code`: CBS buurtcode (bijv. BU05990110)
- `k`: Aantal resultaten (1-10)

Response:
```json
{
  "base_buurt_code": "BU05990110",
  "base_cluster_label_short": "Jong & stedelijk",
  "base_cluster_label_long": "Drukke binnenstad met veel jonge volwassenen...",
  "base_pca_x": 1.299,
  "base_pca_y": 1.779,
  "neighbours": [
    {
      "buurt_code": "BU03630000",
      "naam": "Amsterdam Centrum",
      "gemeente": "Amsterdam",
      "distance": 0.15,
      "cluster": 3,
      "cluster_label_short": "Jong & stedelijk",
      "population": 12500,
      "income_per_person": 42.5,
      "pca_x": 1.245,
      "pca_y": 1.812
    }
  ]
}
```

### GET /api/buurt-cluster
Geeft cluster informatie voor een buurt.

Parameters:
- `buurt_code`: CBS buurtcode

Response:
```json
{
  "buurt_code": "BU05990110",
  "cluster": 3,
  "label": "Jong & stedelijk",
  "label_long": "Drukke binnenstad met veel jonge volwassenen..."
}
```

### GET /api/buurt-geometry
Geeft buurt-omlijning polygon voor kaartvisualisatie.

Parameters:
- `buurt_code`: CBS buurtcode (bijv. BU05990110)

Response:
```json
{
  "type": "Feature",
  "geometry": {
    "type": "MultiPolygon",
    "coordinates": [[[...]]]
  },
  "properties": {...}
}
```

**Gebruikt PDOK WFS voor accurate buurtgrenzen op beide kaarten.**

### GET /api/buurt-crime
Geeft criminaliteitsgegevens voor een buurt (2024 data).

Parameters:
- `buurt_code`: CBS buurtcode

Response:
```json
{
  "buurt_code": "BU05990110",
  "total_crimes": 42,
  "crime_rate_per_1000": 15.7
}
```

### GET /api/health
Health check endpoint.

## 🚀 Live Deployment

**🌐 URL:** [micro-neighbourhood-production-4237.up.railway.app](https://micro-neighbourhood-production-4237.up.railway.app)

**📅 Laatste update:** November 2025

**✨ Laatste updates (November 2025):**
- **🚀 Directe buurtvergelijking** - Tweede zoekbalk met radar chart vergelijking
- **🗺️ Dubbele kaarten** - Sync zoom tussen hoofd- en vergelijkingskaart
- **🏘️ Buurt-omlijning** - PDOK WFS polygons voor accurate buurtgrenzen
- **🔍 Intelligente autocomplete** - Buurt namen in dropdown bij adres zoeken
- **📊 Radar comparison chart** - 6-dimensionele visuele buurtvergelijking
- **🏷️ Echte buurt namen** - KNN resultaten tonen buurt namen i.p.v. CBS codes
- **📏 Meer statistieken** - Voorzieningen afstanden (supermarkt, huisarts, school, kinderdagverblijf)
- **🎯 Verbeterde UX** - Enter toets & suggestie-klik werken beide voor direct zoeken
- **🔄 Verticale layout** - Hoofd kaart bovenaan, vergelijkingssectie eronder
- AI verhalen verrijkt met cluster context en vergelijkbare buurten
- Criminaliteitsdata 2024 geïntegreerd - CBS Politie misdaadcijfers per buurt

### Docker Deployment op Railway

De applicatie gebruikt Docker voor consistente deployment:

1. **Repository Structure**
   ```
   ├── Dockerfile          # Multi-stage build (Node + Python)
   ├── frontend/           # React app + MapLibre + Highcharts
   │   ├── src/
   │   │   ├── components/AddressSearchBar.jsx  # Autocomplete component
   │   │   └── App.jsx       # Main app met buurtvergelijking
   │   └── package.json
   ├── backend/            # FastAPI backend + ML
   │   ├── main.py         # API endpoints (neighbourhood-story, similar-buurten, etc.)
   │   ├── offline/        # ML preprocessing scripts
   │   │   ├── build_clusters.py        # KMeans + LLM labeling
   │   │   ├── fetch_crime_and_merge.py # CBS criminaliteit data
   │   │   ├── fetch_buurt_namen.py     # CBS buurt namen
   │   │   └── merge_buurt_namen.py     # Data merging
   │   └── data/           # Precomputed ML data (CSV files)
   ├── README.md           # Deze documentatie
   └── requirements.txt    # Python dependencies
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
   cd backend/offline
   # Zorg voor OPENAI_API_KEY in environment
   python fetch_buurt_namen.py      # Haal buurt namen op van CBS
   python fetch_crime_and_merge.py  # Haal criminaliteitsdata 2024 op
   python merge_buurt_namen.py      # Voeg buurt namen toe aan dataset
   python build_clusters.py         # Genereert ML modellen + labels
   ```

3. **Backend Setup**
   ```bash
   cd backend
   python3 -m pip install -r requirements.txt
   # OPENAI_API_KEY toevoegen aan environment
   ```

4. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   npm run build  # Of npm run dev voor development
   ```

5. **Start Development Server**
   ```bash
   cd backend
   python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Open:** http://localhost:8000

**Development Notes:**
- Vite proxy routes `/api/*` requests automatisch naar backend
- Frontend hot-reload werkt parallel met backend reload
- Precomputed data staat in `backend/data/` voor snelle startup
