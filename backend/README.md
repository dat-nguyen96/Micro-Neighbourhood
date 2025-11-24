# Neighbourhood AI Backend 🏘️🤖

**FastAPI backend** met OpenAI integratie, machine learning clustering, en Nederlandse buurt data processing.

## 🚀 Features

- **OpenAI GPT-4o-mini** integratie voor buurtbeschrijvingen
- **KMeans clustering** van alle Nederlandse buurten (12 clusters gebaseerd op demografie + 3 soorten criminaliteit)
- **KNN recommender** voor vergelijkbare buurten
- **CBS data integratie** (demografie 85984NED, gedetailleerde criminaliteit 47018NED)
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

### Data Setup & Preprocessing

**⚠️ Vereist voor productie** - Deze stappen genereren de ML modellen en buurt data die de backend gebruikt.

#### 📋 Vereiste Bestanden (na preprocessing)
- `data/buurten_features_clusters_with_crime_2024.csv` - Hoofddata voor ML
- `data/cbs_buurt_namen_85984.csv` - Buurt naam lookups

#### 🚀 Stap-voor-stap Data Pipeline

**Stap 1: Buurt Namen Ophalen**
```bash
cd offline
python fetch_buurt_namen.py
```
- Haalt alle buurt codes + namen op van CBS 85984NED
- Output: `data/cbs_buurt_namen_85984.csv`

**Stap 2: CBS Demografische Data Ophalen**
```bash
python build_clusters.py --fetch-only
```
- Haalt bevolkingsdata van CBS 85984NED (nieuwere dataset)
- Output: `data/cbs_buurten_raw.csv`

**Stap 3: Criminaliteit Data Ophalen & Verwerken**
```bash
python fetch_crime_and_merge.py
```
- Haalt gedetailleerde criminaliteit van CBS 47018NED
- Verwerkt 59 soorten misdrijven naar 3 categorieën
- Output: `data/cbs_crime_2024JJ00_buurten.csv`

**Stap 4: Data Samenvoegen**
```bash
python process_crime_data.py  # Verwerkt ruwe crime data
python merge_buurt_namen.py   # Voegt buurt namen toe
```

**Stap 5: ML Clustering Uitvoeren**
```bash
python build_clusters.py --recluster
```
- Voert KMeans clustering uit met 12 features (9 demo + 3 crime types)
- Genereert cluster labels met OpenAI
- Output: `data/buurten_features_clusters_with_crime_2024.csv`

#### 🎯 Eindresultaat
Backend gebruikt `buurten_features_clusters_with_crime_2024.csv` met:
- 13,817 Nederlandse buurten
- 12 ML features (demografie + 3 soorten criminaliteit)
- 12 clusters met AI-gegenereerde labels
- KNN recommender voor vergelijkbare buurten

#### 🔧 Troubleshooting

**"Missing feature columns" error:**
- Controleer of je alle preprocessing stappen hebt uitgevoerd
- Dataset is bijgewerkt naar CBS 85984NED (2024)

**OpenAI API fouten:**
- Stel `OPENAI_API_KEY` environment variable in
- Gebruik GPT-4o-mini voor consistente resultaten

**Geheugen problemen:**
- CBS datasets zijn groot (>80MB), zorg voor voldoende RAM
- Clustering gebeurt offline, niet tijdens runtime

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
- **85984NED**: Kerncijfers wijken en buurten (demografie, bevolking 2024)
- **47018NED**: Politie criminaliteitscijfers 2024 (59 soorten misdrijven → 3 categorieën)
- **85984NED/WijkenEnBuurtenCodes**: Buurt naam mapping voor gebruiksvriendelijkheid

### ML Pipeline
1. **Data Collection**: CBS OData API calls (85984NED + 47018NED)
2. **Preprocessing**: Pandas cleaning, crime aggregation (59 → 3 types)
3. **Feature Engineering**: 12 features (9 demo + 3 crime categories)
4. **Clustering**: KMeans with 12 clusters
5. **Labeling**: OpenAI GPT-4o-mini cluster descriptions
6. **Storage**: Precomputed CSV for fast inference

### 📁 Bestandsstructuur
```
data/
├── buurten_features_clusters_with_crime_2024.csv  # 🎯 Backend gebruikt dit
└── cbs_buurt_namen_85984.csv                      # Naam lookups
```

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
