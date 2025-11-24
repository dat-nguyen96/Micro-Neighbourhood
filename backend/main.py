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
CLUSTERS_CSV = DATA_DIR / "buurten_features_clusters_with_crime_2024.csv"

FEATURE_COLUMNS = [
    "AantalInwoners_5",
    "Bevolkingsdichtheid_34",
    "GemiddeldeHuishoudensgrootte_33",
    # "GemiddeldInkomenPerInwoner_78",  # Not available in 85984NED dataset
    "HuishoudensMetKinderen_32",
    "HuishoudensTotaal_29",
    # "HuishOnderOfRondSociaalMinimum_85",  # Not available in 85984NED dataset
    "MateVanStedelijkheid_122",
    "k_0Tot15Jaar_8",
    "k_25Tot45Jaar_10",
    "k_65JaarOfOuder_12",
    "crime_property",    # Vermogensmisdrijven (diefstal)
    "crime_violence",    # Geweldsmisdrijven
    "crime_vandalism",   # Vernieling/openbare orde
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

# Serve static files from data directory
app.mount("/data", StaticFiles(directory="data"), name="data")

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
    buurt_naam: str     # echte buurt naam
    gemeente_naam: str  # gemeente naam
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

    # 0) Buurt identificatie
    if "address" in data and data["address"]:
        summary_lines.append(f"Locatie: {data['address']}")

    # Probeer buurt naam en gemeente te vinden
    buurt_naam = None
    gemeente_naam = None

    if "clusterInfo" in data and data["clusterInfo"] and "buurt_naam" in data["clusterInfo"]:
        buurt_naam = data["clusterInfo"]["buurt_naam"]
        print(f"[ANALYSE] Buurt naam gevonden in clusterInfo: {buurt_naam}")

    if "cbsStats" in data and data["cbsStats"]:
        # Gebruik gemeente naam uit cbsStats als deze beschikbaar is
        if "gemeenteNaam" in data["cbsStats"] and data["cbsStats"]["gemeenteNaam"]:
            gemeente_naam = str(data["cbsStats"]["gemeenteNaam"]).strip()
            print(f"[ANALYSE] Gemeente naam gevonden in cbsStats: {gemeente_naam}")

        # Zoek buurt naam en gemeente in onze data als fallback
        if "buurtCode" in data["cbsStats"]:
            buurt_code = data["cbsStats"]["buurtCode"]
            try:
                _ensure_cbs_data_loaded()
                if CBS_DF is not None:
                    matches = CBS_DF[CBS_DF["WijkenEnBuurten"].str.strip() == buurt_code.strip()]
                    if not matches.empty:
                        row = matches.iloc[0]
                        # Gebruik buurt naam uit clusterInfo, of fallback naar CBS data
                        if not buurt_naam:
                            buurt_naam = str(row.get("buurt_naam", "")).strip()
                            if buurt_naam:
                                print(f"[ANALYSE] Buurt naam gevonden in CBS data: {buurt_naam}")

                        # Gebruik gemeente naam uit cbsStats, of fallback naar CBS data
                        if not gemeente_naam:
                            gemeente_naam = str(row.get("Gemeentenaam_1", "")).strip()
                            if gemeente_naam:
                                print(f"[ANALYSE] Gemeente naam gevonden in CBS data: {gemeente_naam}")
            except Exception as e:
                print(f"[ANALYSE] Error looking up buurt/gemeente data: {e}")

    if buurt_naam:
        summary_lines.append(f"Buurt naam: {buurt_naam}")
    else:
        print(f"[ANALYSE] Buurt naam NIET gevonden")

    if gemeente_naam:
        summary_lines.append(f"Gemeente: {gemeente_naam}")

    if "cbsStats" in data and data["cbsStats"] and "buurtCode" in data["cbsStats"]:
        summary_lines.append(f"CBS buurtcode: {data['cbsStats']['buurtCode']}")

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

    # 2) Leefbaarheid & voorzieningen
    if "cbsStats" in data and data["cbsStats"]:
        cbs = data["cbsStats"]

        # Stedelijkheid
        if "stedelijkheid" in cbs and cbs["stedelijkheid"] is not None:
            stedelijkheid = int(cbs["stedelijkheid"])
            stedelijkheid_labels = {
                1: "zeer sterk stedelijk",
                2: "sterk stedelijk",
                3: "matig stedelijk",
                4: "weinig stedelijk",
                5: "niet stedelijk"
            }
            label = stedelijkheid_labels.get(stedelijkheid, f"stedelijkheid niveau {stedelijkheid}")
            summary_lines.append(f"Deze buurt is {label} (CBS stedelijkheidsschaal 1-5).")

        # Woningen
        if "pctAppartementen" in cbs and cbs["pctAppartementen"] is not None:
            pct_app = float(cbs["pctAppartementen"])
            if pct_app > 50:
                woning_type = "voornamelijk appartementen"
            elif pct_app > 25:
                woning_type = "veel appartementen"
            else:
                woning_type = "voornamelijk eengezinswoningen"
            summary_lines.append(f"Woontype: {woning_type} ({pct_app:.1f}% appartementen).")

        # Voorzieningen afstanden
        if "amenities" in cbs and cbs["amenities"]:
            amenities = cbs["amenities"]
            voorzieningen = []

            if "supermarket_km" in amenities and amenities["supermarket_km"] is not None:
                km = float(amenities["supermarket_km"])
                if km < 0.5:
                    voorzieningen.append("zeer dichtbij supermarkt")
                elif km < 1:
                    voorzieningen.append("dichtbij supermarkt")
                else:
                    voorzieningen.append(f"supermarkt op {km:.1f}km")

            if "huisarts_km" in amenities and amenities["huisarts_km"] is not None:
                km = float(amenities["huisarts_km"])
                if km < 0.5:
                    voorzieningen.append("zeer dichtbij huisarts")
                elif km < 1:
                    voorzieningen.append("dichtbij huisarts")
                else:
                    voorzieningen.append(f"huisarts op {km:.1f}km")

            if voorzieningen:
                summary_lines.append(f"Voorzieningen: {', '.join(voorzieningen)}.")

        # Auto bezit
        if "carsPerHousehold" in cbs and cbs["carsPerHousehold"] is not None:
            cars = float(cbs["carsPerHousehold"])
            if cars < 0.8:
                mobiliteit = "relatief lage autobezit"
            elif cars > 1.2:
                mobiliteit = "hoge autobezit"
            else:
                mobiliteit = "gemiddelde autobezit"
            summary_lines.append(f"Auto's per huishouden: {cars:.1f} ({mobiliteit}).")

    # 3) Geometrie / oppervlakte / centroid (optioneel)
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

    # 3) Cluster informatie (ML-gebaseerde classificatie)
    if "clusterInfo" in data and data["clusterInfo"]:
        cluster_info = data["clusterInfo"]
        print(f"[ANALYSE] Cluster info found: {cluster_info}")
        summary_lines.append("\nMachine Learning cluster analyse:")
        if "label" in cluster_info:
            summary_lines.append(f"- Buurt type: {cluster_info['label']}")
        if "label_long" in cluster_info:
            summary_lines.append(f"- Beschrijving: {cluster_info['label_long']}")

    # 4) Criminaliteitsgegevens
    crime_info_added = False

    if "crimeData" in data and data["crimeData"]:
        crime_info = data["crimeData"]
        print(f"[ANALYSE] Crime data found: {crime_info}")
        summary_lines.append("\nCriminaliteitsgegevens:")
        crime_info_added = True

        if "total_crimes" in crime_info and crime_info["total_crimes"] is not None:
            summary_lines.append(f"- Totaal geregistreerde misdrijven: {crime_info['total_crimes']}")
        if "crime_rate_per_1000" in crime_info and crime_info["crime_rate_per_1000"] is not None:
            rate = crime_info["crime_rate_per_1000"]
            if rate < 30:
                safety = "relatief veilig"
            elif rate < 60:
                safety = "gemiddeld veiligheidsniveau"
            else:
                safety = "hoger criminaliteitsniveau"
            summary_lines.append(f"- Misdaad per 1000 inwoners: {rate:.1f} ({safety})")

    # Gedetailleerde criminaliteitscijfers uit CBS 85984NED
    if "cbsStats" in data and data["cbsStats"]:
        cbs_stats = data["cbsStats"]
        detailed_crime = []

        if "geweldsMisdrijven" in cbs_stats and cbs_stats["geweldsMisdrijven"] is not None:
            geweld_rate = float(cbs_stats["geweldsMisdrijven"])
            if geweld_rate < 2:
                geweld_level = "zeer laag geweldsniveau"
            elif geweld_rate < 5:
                geweld_level = "laag geweldsniveau"
            elif geweld_rate < 10:
                geweld_level = "gemiddeld geweldsniveau"
            else:
                geweld_level = "hoog geweldsniveau"
            detailed_crime.append(f"- Gewelds- en seksuele misdrijven: {geweld_rate:.1f} per 1.000 inw. ({geweld_level})")

        if "vermogensMisdrijven" in cbs_stats and cbs_stats["vermogensMisdrijven"] is not None:
            vermogen_rate = float(cbs_stats["vermogensMisdrijven"])
            if vermogen_rate < 5:
                vermogen_level = "zeer laag inbraakrisico"
            elif vermogen_rate < 15:
                vermogen_level = "laag inbraakrisico"
            elif vermogen_rate < 30:
                vermogen_level = "gemiddeld inbraakrisico"
            else:
                vermogen_level = "hoog inbraakrisico"
            detailed_crime.append(f"- Vermogensmisdrijven (inbraak/diefstal woning): {vermogen_rate:.1f} per 1.000 inw. ({vermogen_level})")

        if detailed_crime:
            if not crime_info_added:
                summary_lines.append("\nCriminaliteitsgegevens:")
                crime_info_added = True
            summary_lines.extend(detailed_crime)
            print(f"[ANALYSE] Detailed crime stats added: {len(detailed_crime)} categories")

    # 5) Vergelijkbare buurten (KNN resultaten)
    if "similarBuurten" in data and data["similarBuurten"] and "neighbours" in data["similarBuurten"]:
        neighbours = data["similarBuurten"]["neighbours"]
        if neighbours and len(neighbours) > 0:
            print(f"[ANALYSE] Found {len(neighbours)} similar neighbourhoods")
            summary_lines.append(f"\nVergelijkbare buurten gevonden via KNN ({len(neighbours)} resultaten):")
            for i, nb in enumerate(neighbours[:3]):  # Toon max 3 voorbeelden
                buurt_info = f"- {nb.get('naam', nb.get('buurt_code', 'Onbekend'))}"
                if 'gemeente' in nb:
                    buurt_info += f" ({nb['gemeente']})"
                if 'cluster_label_short' in nb:
                    buurt_info += f" - type: {nb['cluster_label_short']}"
                summary_lines.append(buurt_info)

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

    print("[API] clusterInfo in data:", req.data.get("clusterInfo"))
    print("[API] cbsStats in data:", req.data.get("cbsStats"))
    analysis_summary, metrics = analyse_neighbourhood_data(req.data)

    # Extract gemeente en buurt naam voor duidelijke instructies
    gemeente_naam = "Nederland"  # fallback
    buurt_naam = "deze buurt"    # fallback

    if "cbsStats" in req.data and req.data["cbsStats"] and "gemeenteNaam" in req.data["cbsStats"]:
        gemeente_naam = req.data["cbsStats"]["gemeenteNaam"]
    if "clusterInfo" in req.data and req.data["clusterInfo"] and "buurt_naam" in req.data["clusterInfo"]:
        buurt_naam = req.data["clusterInfo"]["buurt_naam"]

    print(f"[API] AI story voor buurt '{buurt_naam}' in gemeente '{gemeente_naam}'")

    prompt = f"""
Acteer als een neutrale Nederlandse buurtuitlegger.

Je gaat een verhaal schrijven over de buurt "{buurt_naam}" in de gemeente {gemeente_naam}.

Je krijgt gestructureerde data over deze specifieke buurt in {gemeente_naam}
(en eventueel een persona) en je schrijft een korte, vriendelijke
uitleg voor iemand die overweegt in de buurt "{buurt_naam}" te wonen.

BELANGRIJK CONTEXT:
- Dit verhaal gaat SPECIFIEK over de buurt "{buurt_naam}" in {gemeente_naam}
- Uitgebreide leefbaarheidsdata beschikbaar:
  * Criminaliteit: gewelds- en vermogensmisdrijven per 1.000 inwoners
  * Voorzieningen: afstanden tot supermarkt, huisarts, school, kinderdagverblijf
  * Woningtypes: percentage appartementen vs eengezinswoningen
  * Stedelijkheid: mate van stedelijkheid (1-5 schaal)
  * Mobiliteit: autobezit per huishouden
- Plus machine learning informatie:
  * Cluster classificatie (buurt type gebaseerd op socio-demografische data)
  * Vergelijkbare buurten gevonden via KNN algoritme
- Gebruik alle veiligheids-, leefbaarheids- en voorzieningengegevens voor een compleet buurtbeeld.

Regels:
- Schrijf in het Nederlands.
- Maximaal 5 korte alinea's.
- Begin altijd met de naam van de buurt "{buurt_naam}" in de gemeente {gemeente_naam}.
- Geen juridisch, financieel of veiligheidsadvies.
- Wees beschrijvend maar neutraal.
- Focus op vibe: druk/rustig, jong/oud, voorzieningen, woningtype.
- Gebruik de cluster informatie en criminaliteitscijfers om context te geven over de veiligheid en leefbaarheid van "{buurt_naam}".

Persona van de lezer: {persona}

Samenvatting van de ruwe buurtdata (uit een pandas/geopandas-analyse inclusief ML-cluster analyse):

{analysis_summary}

Originele data (JSON):
{req.data}

Schrijf nu over de buurt "{buurt_naam}" in {gemeente_naam}:

1) Een titel van max 60 tekens die "{buurt_naam}" vermeldt.
2) Een korte samenvattende intro (1–2 zinnen) die het karakter van "{buurt_naam}" benadrukt.
3) Kopje "Pluspunten" met 3 bullets over wonen in "{buurt_naam}".
4) Kopje "Let op" met 3 bullets over wonen in "{buurt_naam}".
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
                naam=str(row.get("buurt_naam", "")).strip()
                or str(row["WijkenEnBuurten"]).strip(),
                gemeente=str(row["Gemeentenaam_1"]).strip(),
                distance=float(dist),
                cluster=int(row["cluster_id"]),
                cluster_label_short=str(row.get("cluster_label_short", f"Cluster {int(row['cluster_id'])}")),
                population=float(feat_row["AantalInwoners_5"])
                if pd.notna(feat_row["AantalInwoners_5"]) else None,
                income_per_person=None,  # Not available in 85984NED dataset
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
        buurt_naam=str(row.get("buurt_naam", buurt_code.strip())),
        gemeente_naam=str(row.get("Gemeentenaam_1", "")).strip(),
        cluster=cluster_id,
        label=cluster_info["label_short"],
        label_long=cluster_info["label_long"],
    )


@app.get("/api/buurt-geometry")
async def buurt_geometry(buurt_code: str = Query(..., description="CBS buurtcode, bv. BU05990110")):
    """
    Haal buurt-omlijning op via PDOK WFS.
    Retourneert GeoJSON geometry van de buurt.
    """
    try:
        # PDOK WFS voor wijken/buurten
        wfs_url = f"https://geodata.nationaalgeoregister.nl/wijkenbuurten2023/wfs?service=WFS&request=GetFeature&typeName=buurt_2023&outputFormat=application/json&CQL_FILTER=buurtcode='{buurt_code}'"

        async with httpx.AsyncClient() as client:
            response = await client.get(wfs_url)
            response.raise_for_status()

            geojson = response.json()

            # Controleer of we features hebben
            if not geojson.get("features"):
                raise HTTPException(status_code=404, detail=f"Geen buurt-geometry gevonden voor {buurt_code}")

            # Retourneer de eerste feature (moet er maar één zijn)
            feature = geojson["features"][0]
            return feature

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"PDOK WFS fout: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geometry ophaal fout: {str(e)}")


@app.get("/api/buurt-crime")
async def buurt_crime(
    buurt_code: str = Query(..., description="CBS buurtcode, bv. BU05990110")
):
    """
    Crime data endpoint: geeft criminaliteitsgegevens voor een buurt.
    """
    _ensure_cbs_data_loaded()
    assert CBS_DF is not None

    try:
        idx = _find_buurt_index(buurt_code)
        row = CBS_DF.iloc[idx]

        crime_data = {}
        if "total_crimes" in row and pd.notna(row["total_crimes"]):
            crime_data["total_crimes"] = float(row["total_crimes"])
            # Calculate crime rate per 1000 inhabitants if we have population data
            if "AantalInwoners_5" in row and pd.notna(row["AantalInwoners_5"]) and row["AantalInwoners_5"] > 0:
                crime_data["crime_rate_per_1000"] = (row["total_crimes"] / row["AantalInwoners_5"]) * 1000

        return {
            "buurt_code": buurt_code.strip(),
            "crime_data": crime_data if crime_data else None,
        }
    except HTTPException:
        # Buurt niet gevonden
        return {
            "buurt_code": buurt_code.strip(),
            "crime_data": None,
        }


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

@app.get("/api/buurt-stats", response_model=Dict)
async def buurt_stats(buurt_code: str = Query(..., description="CBS buurtcode, bv. BU05990110")):
    """
    Geef gedetailleerde buurt statistieken terug uit onze voorbewerkte CBS data.
    Inclusief demografie en criminaliteit (3 categorieën).
    """
    _ensure_cbs_data_loaded()
    assert CBS_DF is not None

    # Zoek de buurt in onze data
    mask = CBS_DF["WijkenEnBuurten"] == buurt_code
    if not mask.any():
        raise HTTPException(
            status_code=404,
            detail=f"Buurtcode {buurt_code} niet gevonden in onze data."
        )

    row = CBS_DF[mask].iloc[0]

    # Bouw statistieken object
    stats = {
        "buurtCode": buurt_code,
        "gemeenteNaam": str(row.get("Gemeentenaam_1", "")).strip(),

        # Demografie
        "population": int(row.get("AantalInwoners_5", 0)) if pd.notna(row.get("AantalInwoners_5")) else None,
        "density": float(row.get("Bevolkingsdichtheid_34", 0)) if pd.notna(row.get("Bevolkingsdichtheid_34")) else None,
        "pct65Plus": None,  # Wordt berekend uit leeftijdsgroepen

        # Leeftijdsgroepen
        "ageGroups": {
            "0–15": int(row.get("k_0Tot15Jaar_8", 0)) if pd.notna(row.get("k_0Tot15Jaar_8")) else 0,
            "15–25": int(row.get("k_15Tot25Jaar_9", 0)) if pd.notna(row.get("k_15Tot25Jaar_9")) else 0,
            "25–45": int(row.get("k_25Tot45Jaar_10", 0)) if pd.notna(row.get("k_25Tot45Jaar_10")) else 0,
            "45–65": int(row.get("k_45Tot65Jaar_11", 0)) if pd.notna(row.get("k_45Tot65Jaar_11")) else 0,
            "65+": int(row.get("k_65JaarOfOuder_12", 0)) if pd.notna(row.get("k_65JaarOfOuder_12")) else 0,
        },

        # Inkomen - niet beschikbaar in 85984NED
        "incomePerPerson": None,

        # Criminaliteit (uit onze voorbewerkte data)
        "vermogensMisdrijven": float(row.get("crime_property", 0)) if pd.notna(row.get("crime_property")) else None,
        "geweldsMisdrijven": float(row.get("crime_violence", 0)) if pd.notna(row.get("crime_violence")) else None,
        "vernielingsMisdrijven": float(row.get("crime_vandalism", 0)) if pd.notna(row.get("crime_vandalism")) else None,

        # Bereken percentage 65+ uit leeftijdsgroepen
        "pct65Plus": None,
    }

    # Bereken percentage 65+
    if stats["population"] and stats["population"] > 0:
        over65 = stats["ageGroups"]["65+"]
        stats["pct65Plus"] = round((over65 / stats["population"]) * 100, 1)

    return stats

if FRONTEND_DIST.exists():
    print("[BOOT] Mounting static frontend at /")
    app.mount(
        "/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="static"
    )
else:
    print("[BOOT][WARN] frontend/dist bestaat niet, alleen API beschikbaar.")
