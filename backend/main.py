# backend/main.py - FastAPI Application Entry Point
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import httpx

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# Import our modular services
try:
    # Try relative imports (when run as package)
    from .models import (
        NeighbourhoodMetrics, NeighbourhoodStoryRequest, NeighbourhoodStoryResponse,
        SimilarBuurt, SimilarBuurtenResponse, ClusterInfoResponse
    )
    from .ai_service import call_openai
    from .data_service import _ensure_cbs_data_loaded, _find_buurt_index, _get_cluster_info
    from .search_service import analyse_neighbourhood_data, find_similar_neighbourhoods
except ImportError:
    # Fallback to absolute imports (when run as script)
    from models import (
        NeighbourhoodMetrics, NeighbourhoodStoryRequest, NeighbourhoodStoryResponse,
        SimilarBuurt, SimilarBuurtenResponse, ClusterInfoResponse
    )
    from ai_service import call_openai
    from data_service import _ensure_cbs_data_loaded, _find_buurt_index, _get_cluster_info
    from search_service import analyse_neighbourhood_data, find_similar_neighbourhoods

# ---------- App Setup ----------
app = FastAPI(
    title="Micro-Neighbourhood API",
    description="AI-powered neighbourhood analysis and clustering",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In productie: specificeer domeinen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mounts - deze moeten NA de API routes komen
# anders overschrijven ze de API endpoints


# ---------- API Endpoints ----------

@app.post("/api/neighbourhood-story", response_model=NeighbourhoodStoryResponse)
async def neighbourhood_story(req: NeighbourhoodStoryRequest):
    """Genereer een verhaal over een buurt met AI."""
    try:
        # Analyseer de buurt data
        summary, metrics = analyse_neighbourhood_data(req.data)

        # Bouw prompt voor AI
        prompt_parts = [summary]

        if req.persona:
            prompt_parts.append(f"\nSchrijf dit verhaal voor: {req.persona}")

        prompt_parts.append(
            "\nSchrijf een boeiend, informatief verhaal van 150-200 woorden over deze buurt. "
            "Gebruik een natuurlijke, Nederlandse schrijfstijl. "
            "Focus op leefbaarheid, voorzieningen en sfeer."
        )

        prompt = "\n".join(prompt_parts)

        # Genereer verhaal met AI
        story = call_openai(prompt, max_tokens=500)

        return NeighbourhoodStoryResponse(story=story)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI story generation failed: {str(e)}")


@app.get("/api/similar-buurten", response_model=SimilarBuurtenResponse)
async def get_similar_neighbourhoods(buurt_code: str = Query(..., description="CBS buurt code")):
    """Vind vergelijkbare buurten op basis van ML features."""
    print(f"API: /api/similar-buurten called with buurt_code={buurt_code}")
    try:
        similar_buurten = find_similar_neighbourhoods(buurt_code, n_similar=5)
        print(f"API: Found {len(similar_buurten)} similar buurten")

        # Haal base cluster info op
        base_idx = _find_buurt_index(buurt_code)
        base_row = _ensure_cbs_data_loaded().iloc[base_idx]
        base_cluster_info = _get_cluster_info(int(base_row.get("cluster_id", 0)))

        try:
            from .models import SimilarBuurt, SimilarBuurtenResponse
        except ImportError:
            from models import SimilarBuurt, SimilarBuurtenResponse

        neighbours = []
        for b in similar_buurten:
            neighbours.append(
                SimilarBuurt(
                    buurt_code=b["buurt_code"],
                    naam=b["naam"],
                    gemeente=b["gemeente"],
                    distance=b["distance"],
                    cluster=int(base_row.get("cluster_id", 0)),  # Gebruik base cluster voor nu
                    cluster_label_short=b["cluster_label"],
                    population=None,  # Wordt niet gebruikt in huidige data
                    income_per_person=None,  # Niet beschikbaar
                    pca_x=float(base_row.get("pca_x", 0)) if pd.notna(base_row.get("pca_x")) else None,
                    pca_y=float(base_row.get("pca_y", 0)) if pd.notna(base_row.get("pca_y")) else None,
                )
            )

        response = SimilarBuurtenResponse(
            base_buurt_code=buurt_code,
            base_cluster_label_short=base_cluster_info["label_short"],
            base_cluster_label_long=base_cluster_info["label_long"],
            base_pca_x=float(base_row.get("pca_x", 0)) if pd.notna(base_row.get("pca_x")) else None,
            base_pca_y=float(base_row.get("pca_y", 0)) if pd.notna(base_row.get("pca_y")) else None,
            neighbours=neighbours
        )
        print(f"API: Returning response with {len(response.neighbours)} neighbours")
        return response

    except ValueError as e:
        print(f"API: ValueError in similar-buurten: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"API: Exception in similar-buurten: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Similar neighbourhoods search failed: {str(e)}")


@app.get("/api/buurt-cluster", response_model=ClusterInfoResponse)
async def get_buurt_cluster(buurt_code: str = Query(..., description="CBS buurt code")):
    """Haal cluster informatie op voor een buurt."""
    print(f"API: /api/buurt-cluster called with buurt_code={buurt_code}")
    try:
        # Vind buurt in data
        idx = _find_buurt_index(buurt_code)
        df = _ensure_cbs_data_loaded()
        row = df.iloc[idx]
        print(f"API: Found buurt at index {idx}")

        # Cluster info
        cluster_id = int(row.get("cluster_id", 0))
        cluster_info = _get_cluster_info(cluster_id)
        print(f"API: Cluster {cluster_id} = '{cluster_info['label_short']}'")

        # Haal buurt naam op voor response
        df = _ensure_cbs_data_loaded()
        row = df.iloc[idx]

        try:
            from .models import ClusterInfoResponse
        except ImportError:
            from models import ClusterInfoResponse

        response = ClusterInfoResponse(
            buurt_code=buurt_code,
            buurt_naam=str(buurt_code),  # Gebruik buurt_code als naam fallback
        gemeente_naam=str(row.get("Gemeentenaam_1", "")).strip(),
        cluster=cluster_id,
        label=cluster_info["label_short"],
            label_long=cluster_info["label_long"]
        )
        print(f"API: Returning cluster response: {response.dict()}")
        return response

    except ValueError as e:
        print(f"API: ValueError in buurt-cluster: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"API: Exception in buurt-cluster: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Cluster info retrieval failed: {str(e)}")


@app.get("/api/buurt-geometry")
async def buurt_geometry(buurt_code: str = Query(..., description="CBS buurtcode")):
    """Placeholder endpoint voor buurt geometrie."""
    # Voor nu retourneren we een lege response
    # In de toekomst kan dit GeoJSON geometrie teruggeven
    return {
        "buurt_code": buurt_code,
        "geometry": None,  # Placeholder
        "message": "Geometry endpoint not yet implemented"
    }


@app.get("/api/buurt-crime")
async def buurt_crime(buurt_code: str = Query(..., description="CBS buurtcode")):
    """Geef criminaliteit statistieken voor een buurt."""
    try:
        # Vind buurt in data
        idx = _find_buurt_index(buurt_code)
        df = _ensure_cbs_data_loaded()
        row = df.iloc[idx]

        crime_data = {}

        # Gedetailleerde criminaliteitsdata
        if pd.notna(row.get("total_crimes")):
            crime_data["total_crimes"] = float(row["total_crimes"])

        # Calculate crime rate per 1000 inhabitants if we have population data
        if crime_data and pd.notna(row.get("AantalInwoners_5")) and row["AantalInwoners_5"] > 0:
            population = float(row["AantalInwoners_5"])
            crime_data["total_crime_rate_per_1000"] = (crime_data["total_crimes"] / population) * 1000

        return {
            "buurt_code": buurt_code.strip(),
            "crime_data": crime_data if crime_data else None,
        }

    except ValueError as e:
        return {
            "buurt_code": buurt_code.strip(),
            "crime_data": None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crime data retrieval failed: {str(e)}")


@app.get("/api/buurt-stats")
async def buurt_stats(buurt_code: str = Query(..., description="CBS buurtcode, bv. BU05990110")):
    """Geef gedetailleerde buurt statistieken terug uit onze voorbewerkte CBS data."""
    try:
        # Vind buurt in data
        idx = _find_buurt_index(buurt_code)
        df = _ensure_cbs_data_loaded()
        row = df.iloc[idx]

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

        # Huishoudens
        "totalHouseholds": int(row.get("HuishoudensTotaal_29", 0)) if pd.notna(row.get("HuishoudensTotaal_29")) else None,
        "avgHouseholdSize": float(row.get("GemiddeldeHuishoudensgrootte_33", 0)) if pd.notna(row.get("GemiddeldeHuishoudensgrootte_33")) else None,

        # Stedelijkheid (1=zeer sterk stedelijk, 5=niet stedelijk)
        "stedelijkheid": int(row.get("MateVanStedelijkheid_122", 0)) if pd.notna(row.get("MateVanStedelijkheid_122")) else None,

        # Inkomen - niet beschikbaar in 85984NED dataset
        "incomePerPerson": None,

        # Criminaliteit (uit onze voorbewerkte data) - GEDTAILLEERD
        "totaalMisdrijven": float(row.get("total_crimes", 0)) if pd.notna(row.get("total_crimes")) else None,
        "seksueelGeweld": float(row.get("crime_sexual_violence", 0)) if pd.notna(row.get("crime_sexual_violence")) else None,
        "geweldsMisdrijven": float(row.get("crime_violence_total", 0)) if pd.notna(row.get("crime_violence_total")) else None,
        "vermogensMisdrijven": float(row.get("crime_property", 0)) if pd.notna(row.get("crime_property")) else None,
        "vernielingsMisdrijven": float(row.get("crime_vandalism", 0)) if pd.notna(row.get("crime_vandalism")) else None,

        # GEDTAILLEERDE ONDERLIGGENDE DATA voor tooltips
        "criminaliteitDetail": {
            # Seksueel geweld breakdown
            "seksueelGeweld_1_4_1": float(row.get("crime_1_4_1", 0)) if pd.notna(row.get("crime_1_4_1")) else None,  # Zedenmisdrijven tegen jeugdigen
            "seksueelGeweld_1_4_2": float(row.get("crime_1_4_2", 0)) if pd.notna(row.get("crime_1_4_2")) else None,  # Overige zedenmisdrijven

            # Geweldmisdrijven breakdown
            "geweld_1_4_3": float(row.get("crime_1_4_3", 0)) if pd.notna(row.get("crime_1_4_3")) else None,  # Bedreiging
            "geweld_1_4_4": float(row.get("crime_1_4_4", 0)) if pd.notna(row.get("crime_1_4_4")) else None,  # Mishandeling
            "geweld_1_4_5": float(row.get("crime_1_4_5", 0)) if pd.notna(row.get("crime_1_4_5")) else None,  # Moord/doodslag
            "geweld_1_4_6": float(row.get("crime_1_4_6", 0)) if pd.notna(row.get("crime_1_4_6")) else None,  # Straatroof
            "geweld_1_4_7": float(row.get("crime_1_4_7", 0)) if pd.notna(row.get("crime_1_4_7")) else None,  # Overvallen

            # Vermogensmisdrijven breakdown
            "vermogens_1_1_1": float(row.get("crime_1_1_1", 0)) if pd.notna(row.get("crime_1_1_1")) else None,  # Diefstal/inbraak woning
            "vermogens_1_2_1": float(row.get("crime_1_2_1", 0)) if pd.notna(row.get("crime_1_2_1")) else None,  # Diefstal uit/vanaf motorvoertuigen

            # Vernieling breakdown
            "vernieling_2_2_1": float(row.get("crime_2_2_1", 0)) if pd.notna(row.get("crime_2_2_1")) else None,  # Vernieling cq. zaakbeschadiging
            "vernieling_3_6_4": float(row.get("crime_3_6_4", 0)) if pd.notna(row.get("crime_3_6_4")) else None,  # Aantasting openbare orde
        },
        }

        # Bereken percentage 65+ als we de leeftijdsgroepen hebben
        if stats["population"] and stats["population"] > 0:
            age_65_plus = stats["ageGroups"]["65+"]
            stats["pct65Plus"] = round((age_65_plus / stats["population"]) * 100, 1)

        # Voeg cluster informatie toe
        cluster_id = int(row.get("cluster_id", 0))
        cluster_info = _get_cluster_info(cluster_id)
        stats.update({
            "cluster": cluster_id,
            "clusterLabel": cluster_info["label_short"],
            "clusterDescription": cluster_info["label_long"]
        })

        return stats

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Buurt stats retrieval failed: {str(e)}")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Micro-Neighbourhood API is running",
        "services": {
            "cbs_data": "loaded",
            "openai": "configured"
        }
    }


# ---------- Startup Event ----------
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print("🚀 Starting Micro-Neighbourhood API...")
    try:
        # Pre-load CBS data
        df = _ensure_cbs_data_loaded()
        print(f"✅ CBS data loaded on startup: {len(df)} rows")
        print(f"✅ Available columns: {list(df.columns[:10])}...")

        # Test cluster data
        sample_clusters = df['cluster_id'].value_counts().head(3)
        print(f"✅ Sample clusters: {dict(sample_clusters)}")

    except Exception as e:
        print(f"❌ Failed to load CBS data on startup: {e}")
        import traceback
        traceback.print_exc()


# ---------- Static File Serving ----------
# IMPORTANT: Deze moeten NA alle API routes worden gemount!
# Anders overschrijven ze de API endpoints.

# Serve static data files
DATA_DIR = Path(__file__).resolve().parent / "data"
if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")
    print(f"📁 Serving static data files from: {DATA_DIR}")

# Serve static frontend files (catch-all route)
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
    print(f"🌐 Serving frontend files from: {FRONTEND_DIST}")
else:
    print(f"⚠️ Frontend dist directory not found: {FRONTEND_DIST}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
