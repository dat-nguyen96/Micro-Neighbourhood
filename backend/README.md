# Neighbourhood AI Backend 🏘️🤖

**FastAPI backend** met OpenAI integratie, machine learning clustering, en Nederlandse buurt data processing.

## 🚀 Features

- **OpenAI GPT-4o-mini** integratie voor buurtbeschrijvingen
- **KMeans clustering** van alle Nederlandse buurten (12 clusters)
- **KNN recommender** voor vergelijkbare buurten
- **CBS data integratie** (demografie, criminaliteit 2024)
- **PDOK WFS client** voor buurt-omlijning polygons
- **Buurt naam lookup** via CBS metadata

## 📦 Setup

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation
```bash
cd backend
pip install -r requirements.txt
export OPENAI_API_KEY=your_openai_api_key_here
```

### Data Preprocessing (Optioneel)
```bash
cd offline
python fetch_buurt_namen.py      # CBS buurt namen ophalen
python fetch_crime_and_merge.py  # Criminaliteit data 2024
python merge_buurt_namen.py      # Data samenvoegen
python build_clusters.py         # ML modellen genereren
```

### Local Development
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🔗 API Endpoints

### 🤖 POST `/api/neighbourhood-story`
Genereert AI buurtverhaal verrijkt met ML-context.

**Request:**
```json
{
  "data": {
    "address": "Damrak 1, Amsterdam",
    "buildingInfo": {...},
    "cbsStats": {...},
    "clusterInfo": {"label": "Jong & stedelijk"},
    "similarBuurten": {...}
  },
  "persona": "algemeen huishouden"
}
```

**Features:**
- Cluster context integratie
- Vergelijkbare buurten als referentie
- Persoonlijke verhalen per doelgroep

### 🧠 GET `/api/similar-buurten`
Vindt vergelijkbare buurten via KNN machine learning.

**Parameters:**
- `buurt_code`: CBS buurtcode (bijv. BU05990110)
- `k`: Aantal resultaten (1-10, default: 5)

**Response:**
```json
{
  "base_buurt_code": "BU05990110",
  "neighbours": [
    {
      "buurt_code": "BU03630000",
      "naam": "Amsterdam Centrum",
      "gemeente": "Amsterdam",
      "distance": 0.15,
      "cluster_label_short": "Jong & stedelijk",
      "population": 12500
    }
  ]
}
```

### 🏷️ GET `/api/buurt-cluster`
Geeft cluster classificatie voor een buurt.

**Parameters:**
- `buurt_code`: CBS buurtcode

**Response:**
```json
{
  "buurt_code": "BU05990110",
  "buurt_naam": "Stadsdriehoek",
  "cluster": 3,
  "label": "Jong & stedelijk",
  "label_long": "Drukke binnenstad met veel jonge volwassenen..."
}
```

### 🗺️ GET `/api/buurt-geometry`
Haalt buurt-omlijning polygon op via PDOK WFS.

**Parameters:**
- `buurt_code`: CBS buurtcode

**Response:**
```json
{
  "type": "Feature",
  "geometry": {
    "type": "MultiPolygon",
    "coordinates": [[[...]]]
  }
}
```

### 👮 GET `/api/buurt-crime`
Geeft criminaliteitscijfers 2024 voor een buurt.

**Parameters:**
- `buurt_code`: CBS buurtcode

**Response:**
```json
{
  "buurt_code": "BU05990110",
  "total_crimes": 42,
  "crime_rate_per_1000": 15.7
}
```

### 🏥 GET `/api/health`
Health check endpoint.

## 🗃️ Data Pipeline

### CBS Data Sources
- **83765NED**: Kerncijfers wijken en buurten (demografie, inkomen, voorzieningen)
- **47018NED**: Politie criminaliteitscijfers 2024
- **WijkenEnBuurten metadata**: Buurt namen voor gebruiksvriendelijkheid

### ML Pipeline
1. **Data Collection**: CBS OData API calls
2. **Preprocessing**: Pandas cleaning, feature engineering
3. **Clustering**: KMeans op 15+ socio-demografische features
4. **Labeling**: OpenAI GPT voor begrijpelijke cluster beschrijvingen
5. **Storage**: Precomputed CSV voor snelle API responses

### Precomputed Data
```
backend/data/
├── buurten_features_clusters_with_crime_2024.csv  # ML features + clusters
├── cbs_buurt_namen_83765.csv                     # Buurt namen mapping
├── cbs_crime_2024JJ00_buurten.csv               # Criminaliteit per buurt
└── cbs_buurten_raw.csv                          # Raw CBS data
```

## 🚀 Deployment

### Railway (Cloud)
```bash
# Railway.toml configuratie
[build]
builder = "dockerfile"

[deploy]
startCommand = "uvicorn main:app --host 0.0.0.0 --port $PORT"
```

### Docker (Local)
```bash
docker build -t neighbourhood-backend .
docker run -p 8000:8000 -e OPENAI_API_KEY=... neighbourhood-backend
```

### Environment Variables
```bash
OPENAI_API_KEY=sk-...          # Vereist voor AI features
OPENAI_MODEL=gpt-4o-mini       # Default model
DEBUG=true                     # Development mode
```

## 🔧 Development

### ML Model Updates
```bash
cd offline
python build_clusters.py  # Regenerate clusters + labels
```

### API Testing
```bash
# Health check
curl http://localhost:8000/api/health

# Cluster info
curl "http://localhost:8000/api/buurt-cluster?buurt_code=BU05990110"
```

### Logging
- API requests worden gelogd naar console
- ML preprocessing toont progress indicators
- OpenAI calls worden getraceerd voor debugging
