# backend/main.py
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import httpx

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# ---------- paths & env ----------
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"

env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BACKEND_ENV = os.getenv("BACKEND_ENV", "local")  # bv. "local", "production"

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is niet gezet. "
        "Zet deze in .env (lokaal) of als Railway environment variable."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- CBS config for ML ----------

# Path to precomputed CSV with features + clusters
DATA_DIR = BASE_DIR / "data"
CLUSTERS_CSV = DATA_DIR / "buurten_features_clusters.csv"

FEATURE_COLUMNS = [
    "AantalInwoners_5",
    "Bevolkingsdichtheid_33",
    "GemiddeldeHuishoudensgrootte_32",
    "GemiddeldInkomenPerInwoner_66",
    "HuishoudensMetKinderen_31",
    "HuishoudensTotaal_28",
    "HuishOnderOfRondSociaalMinimum_73",
    "MateVanStedelijkheid_104",
    "k_0Tot15Jaar_8",
    "k_25Tot45Jaar_10",
    "k_65JaarOfOuder_12",
]

# Globale in-memory data (lazy geladen uit CSV)
CBS_DF: Optional[pd.DataFrame] = None
FEATURE_DF: Optional[pd.DataFrame] = None
SCALER: Optional[StandardScaler] = None
KNN: Optional[NearestNeighbors] = None

# ---------- FastAPI app ----------

app = FastAPI(
    title="Neighbourhood Explorer API",
    version="0.3.1",
    description="API voor NL buurtverhalen met AI, pandas & geopandas.",
)

# CORS: lokale dev + optionele frontend origin uit env
origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
frontend_origin = os.getenv("FRONTEND_ORIGIN")
if frontend_origin:
    origins.append(frontend_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic modellen ----------


class NeighbourhoodMetrics(BaseModel):
    area_ha: Optional[float] = None
    centroid_lat: Optional[float] = None
    centroid_lon: Optional[float] = None


class NeighbourhoodStoryRequest(BaseModel):
    data: Dict[str, Any] = Field(
        ...,
        description=(
            "Gestructureerde data over de buurt (JSON). "
            "Moet minimaal een 'address' bevatten. "
            "Optioneel kan er ook een 'geometry' (GeoJSON) in zitten."
        ),
    )
    persona: Optional[str] = Field(
        default=None,
        description="Optionele persona, bv. 'jong stel', 'gezin met kinderen'.",
    )


class NeighbourhoodStoryResponse(BaseModel):
    story: str
    area_ha: Optional[float] = None


class SimilarBuurt(BaseModel):
    buurt_code: str
    naam: Optional[str] = None
    gemeente: str
    distance: float
    cluster: int
    cluster_label_short: str
    population: Optional[float] = None
    income_per_person: Optional[float] = None
    pca_x: Optional[float] = None
    pca_y: Optional[float] = None


class SimilarBuurtenResponse(BaseModel):
    base_buurt_code: str
    base_cluster_label_short: Optional[str] = None
    base_cluster_label_long: Optional[str] = None
    base_pca_x: Optional[float] = None
    base_pca_y: Optional[float] = None
    neighbours: List[SimilarBuurt]


class ClusterInfoResponse(BaseModel):
    buurt_code: str
    cluster: int
    label: str          # korte label (voor badge)
    label_long: str     # lange uitleg (voor panel)


# ---------- Data-analyse helpers (pandas & geopandas) ----------


def analyse_neighbourhood_data(data: Dict[str, Any]) -> Tuple[str, Dict[str, Optional[float]]]:
    """
    Gebruik pandas/geopandas om een compacte samenvatting van de data te maken
    voor de LLM + enkele numerieke metrics voor de frontend (zoals oppervlakte).
    """
    print("[ANALYSE] Start analyse_neighbourhood_data")
    df = pd.json_normalize(data)
    print("[ANALYSE] Data columns:", list(df.columns))

    summary_lines: list[str] = []
    metrics: Dict[str, Optional[float]] = {"area_ha": None}

    # 1) Numerieke indicatoren
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    print("[ANALYSE] Numeric columns:", numeric_cols)
    if numeric_cols:
        summary_lines.append("Enkele numerieke indicatoren uit de data:")
        for col in numeric_cols[:10]:
            val = df[col].iloc[0]
            summary_lines.append(f"- {col}: {val}")
    else:
        summary_lines.append("Geen numerieke indicatoren gevonden in de data.")

    # 2) Geometrie / oppervlakte / centroid (optioneel)
    if "geometry" in data and data["geometry"]:
        print("[ANALYSE] Geometry key present, attempting GeoPandas analysis...")
        try:
            geom = shape(data["geometry"])  # verwacht GeoJSON-achtige dict
            gdf = gpd.GeoDataFrame(df, geometry=[geom], crs="EPSG:4326")

            # Approx-oppervlakte in hectare (projectie naar WebMercator)
            gdf_m = gdf.to_crs(3857)
            area_m2 = float(gdf_m.area.iloc[0])
            area_ha = area_m2 / 10_000
            metrics["area_ha"] = area_ha
            summary_lines.append(f"Benaderde oppervlakte: {area_ha:.1f} hectare.")
            print(f"[ANALYSE] Computed area_ha={area_ha:.3f}")

            # Centroid terug in WGS84
            centroid = gdf.to_crs(4326).geometry.iloc[0].centroid
            summary_lines.append(
                f"Ongeveer centrum op (lat, lon): "
                f"{centroid.y:.5f}, {centroid.x:.5f}."
            )
            print(
                f"[ANALYSE] Centroid lat/lon = {centroid.y:.5f}, {centroid.x:.5f}"
            )
        except Exception as exc:
            print("[ANALYSE][ERROR] Geometrie-analyse mislukt:", repr(exc))
            traceback.print_exc()
            summary_lines.append(
                "Geometrie aanwezig maar kon niet worden geanalyseerd."
            )
    else:
        print("[ANALYSE] No geometry field in data; skipping GeoPandas.")

    return "\n".join(summary_lines), metrics


# ---------- CBS ML helpers (KMeans + KNN) ----------


def _ensure_cbs_data_loaded():
    """
    Lazy load van precomputed CBS data uit CSV.
    Wordt aangeroepen bij de ML-endpoints.
    """
    global CBS_DF, FEATURE_DF, SCALER, KNN

    if CBS_DF is not None:
        return

    if not CLUSTERS_CSV.exists():
        raise RuntimeError(
            f"Precomputed clusters CSV niet gevonden: {CLUSTERS_CSV}. "
            "Run eerst: python -m offline.build_clusters"
        )

    print(f"[ML] Loading precomputed clusters from {CLUSTERS_CSV} ...")
    df = pd.read_csv(CLUSTERS_CSV)

    # Check dat we de benodigde kolommen hebben
    required_cols = ["WijkenEnBuurten"] + FEATURE_COLUMNS + ["cluster_id", "cluster_label_short", "cluster_label_long"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Ontbrekende kolommen in CSV: {missing}")

    CBS_DF = df

    # Feature matrix voor KNN
    feat_df = df[FEATURE_COLUMNS].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feat_df.values)

    # KNN op de geschaalde features
    knn = NearestNeighbors(n_neighbors=11, metric="euclidean")
    knn.fit(X_scaled)

    FEATURE_DF = feat_df
    SCALER = scaler
    KNN = knn

    print("[ML] Precomputed data loaded: "
          f"{len(df)} buurten, {len(FEATURE_COLUMNS)} features.")


def _find_buurt_index(buurt_code: str) -> int:
    """Zoek index van buurt in CBS_DF op basis van buurtcode."""
    if CBS_DF is None:
        raise RuntimeError("CBS_DF is niet geladen.")
    matches = CBS_DF.index[CBS_DF["WijkenEnBuurten"].str.strip() == buurt_code.strip()]
    if len(matches) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Buurtcode {buurt_code} niet gevonden in CBS-data.",
        )
    return int(matches[0])


def _get_cluster_info(cluster_id: int) -> Dict[str, str]:
    """
    Haal precomputed cluster labels op uit de geladen data.
    """
    if CBS_DF is None:
        return {"label_short": f"Cluster {cluster_id}", "label_long": "Data niet geladen"}

    # Vind een voorbeeldrij voor dit cluster om de labels te krijgen
    sample_row = CBS_DF[CBS_DF["cluster_id"] == cluster_id].iloc[0] if len(CBS_DF[CBS_DF["cluster_id"] == cluster_id]) > 0 else None

    if sample_row is not None:
        return {
            "label_short": str(sample_row.get("cluster_label_short", f"Cluster {cluster_id}")),
            "label_long": str(sample_row.get("cluster_label_long", ""))
        }
    else:
        return {"label_short": f"Cluster {cluster_id}", "label_long": "Geen data beschikbaar"}


# ---------- OpenAI helper ----------


def call_openai(prompt: str, max_tokens: int = 800) -> str:
    """
    Eenvoudige helper om een chat completion aan te roepen.
    Gooit een exception door bij fouten, zodat de endpoint dat kan afhandelen.
    """
    if not client:
        raise RuntimeError("OpenAI-client niet geconfigureerd (geen API key).")

    print("[OPENAI] Calling model:", OPENAI_MODEL)
    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Je bent een behulpzame, neutrale Nederlandse tekstschrijver. "
                        "Geef geen juridisch, financieel of veiligheidsadvies."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=max_tokens,
        )
    except Exception as exc:
        print("[OPENAI][ERROR]", repr(exc))
        traceback.print_exc()
        raise

    content = completion.choices[0].message.content
    print("[OPENAI] Received completion (length", len(content or ""), ")")
    return content or ""


# ---------- API endpoints ----------


@app.post("/api/neighbourhood-story", response_model=NeighbourhoodStoryResponse)
async def neighbourhood_story(req: NeighbourhoodStoryRequest):
    print("[API] /api/neighbourhood-story called")
    print("[API] persona =", req.persona)
    print("[API] keys in data:", list(req.data.keys()))

    if "address" not in req.data:
        print("[API][ERROR] Missing address in data")
        raise HTTPException(
            status_code=400,
            detail="Missing 'address' in data",
        )

    if not client:
        print("[API][ERROR] OpenAI client not configured")
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key ontbreekt op de server.",
        )

    persona = req.persona or "algemeen huishouden"

    analysis_summary, metrics = analyse_neighbourhood_data(req.data)

    prompt = f"""
Acteer als een neutrale Nederlandse buurtuitlegger.

Je krijgt gestructureerde data over één klein gebied in Nederland
(en eventueel een persona) en je schrijft een korte, vriendelijke
uitleg voor iemand die overweegt daar te wonen.

Regels:
- Schrijf in het Nederlands.
- Maximaal 5 korte alinea's.
- Geen juridisch, financieel of veiligheidsadvies.
- Wees beschrijvend maar neutraal.
- Focus op vibe: druk/rustig, jong/oud, voorzieningen, woningtype.

Persona van de lezer: {persona}

Samenvatting van de ruwe buurtdata (uit een pandas/geopandas-analyse):

{analysis_summary}

Originele data (JSON):
{req.data}

Schrijf nu:

1) Een titel van max 60 tekens.
2) Een korte samenvattende intro (1–2 zinnen).
3) Kopje "Pluspunten" met 3 bullets.
4) Kopje "Let op" met 3 bullets.
"""

    try:
        story = call_openai(prompt)
    except Exception:
        print("[API][ERROR] call_openai failed")
        raise HTTPException(
            status_code=502,
            detail="AI-fout bij genereren buurtverhaal",
        )

    print("[API] Returning story + area_ha:", metrics.get("area_ha"))
    return NeighbourhoodStoryResponse(
        story=story,
        area_ha=metrics.get("area_ha"),
    )


@app.get("/api/similar-buurten", response_model=SimilarBuurtenResponse)
async def similar_buurten(
    buurt_code: str = Query(..., description="CBS buurtcode, bv. BU05990110"),
    k: int = Query(5, ge=1, le=10, description="Aantal vergelijkbare buurten"),
):
    _ensure_cbs_data_loaded()
    assert CBS_DF is not None and FEATURE_DF is not None
    assert SCALER is not None and KNN is not None

    idx = _find_buurt_index(buurt_code)
    base_row = CBS_DF.iloc[idx]

    x = FEATURE_DF.iloc[idx:idx + 1].values
    x_scaled = SCALER.transform(x)

    distances, indices = KNN.kneighbors(x_scaled, n_neighbors=k + 1)
    distances = distances[0]
    indices = indices[0]

    neighbours: List[SimilarBuurt] = []

    for dist, i in zip(distances, indices):
        if i == idx:
            continue  # sla de buurt zelf over

        row = CBS_DF.iloc[i]
        feat_row = FEATURE_DF.iloc[i]

        neighbours.append(
            SimilarBuurt(
                buurt_code=str(row["WijkenEnBuurten"]).strip(),
                naam=str(row.get("Codering_3", "")).strip()
                or str(row["WijkenEnBuurten"]).strip(),
                gemeente=str(row["Gemeentenaam_1"]).strip(),
                distance=float(dist),
                cluster=int(row["cluster_id"]),
                cluster_label_short=str(row.get("cluster_label_short", f"Cluster {int(row['cluster_id'])}")),
                population=float(feat_row["AantalInwoners_5"])
                if pd.notna(feat_row["AantalInwoners_5"]) else None,
                income_per_person=float(feat_row["GemiddeldInkomenPerInwoner_66"])
                if pd.notna(feat_row["GemiddeldInkomenPerInwoner_66"]) else None,
                pca_x=float(row["pca_x"]) if "pca_x" in row and pd.notna(row["pca_x"]) else None,
                pca_y=float(row["pca_y"]) if "pca_y" in row and pd.notna(row["pca_y"]) else None,
            )
        )
        if len(neighbours) >= k:
            break

    base_short = str(base_row.get("cluster_label_short", ""))
    base_long = str(base_row.get("cluster_label_long", ""))

    print(f"[API] /api/similar-buurten for {buurt_code}: base_cluster_label_short='{base_short}', base_cluster_label_long='{base_long}'")

    return SimilarBuurtenResponse(
        base_buurt_code=str(buurt_code).strip(),
        base_cluster_label_short=base_short,
        base_cluster_label_long=base_long,
        base_pca_x=float(base_row["pca_x"]) if "pca_x" in base_row and pd.notna(base_row["pca_x"]) else None,
        base_pca_y=float(base_row["pca_y"]) if "pca_y" in base_row and pd.notna(base_row["pca_y"]) else None,
        neighbours=neighbours,
    )


@app.get("/api/buurt-cluster", response_model=ClusterInfoResponse)
async def buurt_cluster(
    buurt_code: str = Query(..., description="CBS buurtcode, bv. BU05990110")
):
    """
    ML endpoint: geeft het precomputed KMeans-cluster + LLM label voor een buurt.
    """
    _ensure_cbs_data_loaded()
    assert CBS_DF is not None

    idx = _find_buurt_index(buurt_code)
    row = CBS_DF.iloc[idx]
    cluster_id = int(row["cluster_id"])
    cluster_info = _get_cluster_info(cluster_id)

    return ClusterInfoResponse(
        buurt_code=buurt_code.strip(),
        cluster=cluster_id,
        label=cluster_info["label_short"],
        label_long=cluster_info["label_long"],
    )


@app.get("/api/health")
async def health():
    print("[API] /api/health called")
    return {
        "status": "ok",
        "env": BACKEND_ENV,
        "has_openai_key": bool(OPENAI_API_KEY),
    }


# ---------- Startup hook ----------


@app.on_event("startup")
async def on_startup():
    print("[STARTUP] FastAPI startup complete.")
    print("[STARTUP] FRONTEND_DIST exists?", FRONTEND_DIST.exists())


# ---------- Static React build ----------

if FRONTEND_DIST.exists():
    print("[BOOT] Mounting static frontend at /")
    app.mount(
        "/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="static"
    )
else:
    print("[BOOT][WARN] frontend/dist bestaat niet, alleen API beschikbaar.")
