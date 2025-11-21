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

# ---------- paths & env ----------
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"

env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BACKEND_ENV = os.getenv("BACKEND_ENV", "local")  # bv. "local", "production"

# Maak client optioneel; geen harde crash bij import
client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    # Niet crashen; alleen een waarschuwing loggen
    print(
        "⚠️  OPENAI_API_KEY is niet gezet. "
        "AI-endpoints zullen een fout geven totdat deze env var is geconfigureerd."
    )

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
    df = pd.json_normalize(data)

    summary_lines: list[str] = []
    metrics: Dict[str, float] = {"area_ha": None}

    # 1) Numerieke indicatoren
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        summary_lines.append("Enkele numerieke indicatoren uit de data:")
        for col in numeric_cols[:10]:
            val = df[col].iloc[0]
            summary_lines.append(f"- {col}: {val}")
    else:
        summary_lines.append("Geen numerieke indicatoren gevonden in de data.")

    # 2) Geometrie / oppervlakte / centroid (optioneel)
    if "geometry" in data and data["geometry"]:
        try:
            geom = shape(data["geometry"])  # verwacht GeoJSON-achtige dict
            gdf = gpd.GeoDataFrame(df, geometry=[geom], crs="EPSG:4326")

            # Approx-oppervlakte in hectare (projectie naar WebMercator)
            gdf_m = gdf.to_crs(3857)
            area_m2 = float(gdf_m.area.iloc[0])
            area_ha = area_m2 / 10_000
            metrics["area_ha"] = area_ha
            summary_lines.append(f"Benaderde oppervlakte: {area_ha:.1f} hectare.")

            # Centroid terug in WGS84
            centroid = gdf.to_crs(4326).geometry.iloc[0].centroid
            summary_lines.append(
                f"Ongeveer centrum op (lat, lon): "
                f"{centroid.y:.5f}, {centroid.x:.5f}."
            )
        except Exception as exc:
            print("Geometrie-analyse mislukt:", repr(exc))
            summary_lines.append("Geometrie aanwezig maar kon niet worden geanalyseerd.")

    return "\n".join(summary_lines), metrics


# ---------- OpenAI helper ----------


def call_openai(prompt: str, max_tokens: int = 800) -> str:
    """
    Eenvoudige helper om een chat completion aan te roepen.
    Gooit een exception door bij fouten, zodat de endpoint dat kan afhandelen.
    """
    if client is None:
        # Duidelijke fout als iemand de endpoint aanroept zonder key
        raise RuntimeError("OPENAI_API_KEY ontbreekt; AI kan niet worden aangeroepen.")

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
    except RuntimeError as e:
        # Specifiek voor ontbrekende key -> 500 met duidelijke boodschap
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="AI-fout bij genereren buurtverhaal",
        )

    return NeighbourhoodStoryResponse(
        story=story,
        area_ha=metrics.get("area_ha"),
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
