# backend/main.py
import os
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

import pandas as pd
import geopandas as gpd
from shapely.geometry import shape

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

# ---------- FastAPI app ----------

app = FastAPI(
    title="Neighbourhood Explorer API",
    version="0.3.0",
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
    centroid_lat: Optional[float] = None
    centroid_lon: Optional[float] = None


# ---------- Data-analyse helpers (pandas & geopandas) ----------


def analyse_neighbourhood_data(data: Dict[str, Any]) -> dict:
    """
    Gebruik pandas/geopandas om een compacte samenvatting van de data te maken
    voor de LLM én wat ruwe stats terug te geven.
    """
    df = pd.json_normalize(data)

    summary_lines: list[str] = []

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        summary_lines.append("Enkele numerieke indicatoren uit de data:")
        for col in numeric_cols[:10]:
            val = df[col].iloc[0]
            summary_lines.append(f"- {col}: {val}")
    else:
        summary_lines.append("Geen numerieke indicatoren gevonden in de data.")

    area_ha = None
    centroid_lat = None
    centroid_lon = None

    if "geometry" in data:
        try:
            geom = shape(data["geometry"])
            gdf = gpd.GeoDataFrame(df, geometry=[geom], crs="EPSG:4326")

            gdf_m = gdf.to_crs(3857)
            area_m2 = float(gdf_m.area.iloc[0])
            area_ha = area_m2 / 10_000
            summary_lines.append(f"Benaderde oppervlakte: {area_ha:.1f} hectare.")

            centroid = gdf.to_crs(4326).geometry.iloc[0].centroid
            centroid_lat = float(centroid.y)
            centroid_lon = float(centroid.x)
            summary_lines.append(
                f"Ongeveer centrum op (lat, lon): "
                f"{centroid_lat:.5f}, {centroid_lon:.5f}."
            )
        except Exception as exc:
            print("Geometrie-analyse mislukt:", repr(exc))
            summary_lines.append("Geometrie aanwezig maar kon niet worden geanalyseerd.")

    return {
        "summary_text": "\n".join(summary_lines),
        "area_ha": area_ha,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
    }


# ---------- OpenAI helper ----------


def call_openai(prompt: str, max_tokens: int = 800) -> str:
    """
    Eenvoudige helper om een chat completion aan te roepen.
    Gooit een exception door bij fouten, zodat de endpoint dat kan afhandelen.
    """
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
        print("OpenAI fout:", repr(exc))
        raise

    content = completion.choices[0].message.content
    return content or ""


# ---------- API endpoints ----------


@app.post("/api/neighbourhood-story", response_model=NeighbourhoodStoryResponse)
async def neighbourhood_story(req: NeighbourhoodStoryRequest):
    if "address" not in req.data:
        raise HTTPException(
            status_code=400,
            detail="Missing 'address' in data",
        )

    persona = req.persona or "algemeen huishouden"

    analysis = analyse_neighbourhood_data(req.data)
    analysis_summary = analysis["summary_text"]

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
        raise HTTPException(
            status_code=502,
            detail="AI-fout bij genereren buurtverhaal",
        )

    return NeighbourhoodStoryResponse(
        story=story,
        area_ha=analysis["area_ha"],
        centroid_lat=analysis["centroid_lat"],
        centroid_lon=analysis["centroid_lon"],
    )


@app.get("/api/health")
async def health():
    return {"status": "ok", "env": BACKEND_ENV}


# ---------- Static React build ----------

if FRONTEND_DIST.exists():
    app.mount(
        "/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="static"
    )
else:
    print("⚠️ Let op: frontend/dist bestaat niet, alleen API beschikbaar.")
