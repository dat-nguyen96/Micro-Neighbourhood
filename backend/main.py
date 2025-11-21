# backend/main.py
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import traceback

print("[BOOT] Importing main.py...")

# ---------- paths & env ----------
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"
print(f"[BOOT] BASE_DIR={BASE_DIR}")
print(f"[BOOT] FRONTEND_DIST exists? {FRONTEND_DIST.exists()}")

env_path = BASE_DIR / ".env"
if env_path.exists():
    print(f"[BOOT] Loading .env from {env_path}")
    load_dotenv(env_path)
else:
    print("[BOOT] No local .env found (this is normal on Railway).")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BACKEND_ENV = os.getenv("BACKEND_ENV", "local")  # bv. "local", "production"

if OPENAI_API_KEY:
    print("[BOOT] OPENAI_API_KEY is present (length =", len(OPENAI_API_KEY), ")")
else:
    print(
        "[BOOT][WARN] OPENAI_API_KEY is NOT set. "
        "The /api/neighbourhood-story endpoint will return a 500 until you set it."
    )

client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

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
