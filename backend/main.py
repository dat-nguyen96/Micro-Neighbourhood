import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional

# ---------- env & OpenAI ----------

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"

# .env alleen lokaal; op Railway zet je env vars in de UI
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
  raise RuntimeError("OPENAI_API_KEY is niet gezet (zie .env of Railway env vars).")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- FastAPI app ----------

app = FastAPI()

# CORS voor lokale dev (Vite op 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic modellen ----------

class NeighbourhoodStoryRequest(BaseModel):
    data: dict
    persona: Optional[str] = None

def call_openai(prompt: str, max_tokens: int = 800) -> str:
    """
    Eenvoudige helper om een chat completion aan te roepen.
    """
    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # kan je ook "gpt-4o" maken als je wilt
        messages=[
            {
                "role": "developer",
                "content": "Je bent een Nederlandse assistent. "
                           "Geef geen juridisch of financieel advies.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=max_tokens,
    )
    content = completion.choices[0].message.content
    return content or ""

@app.post("/api/neighbourhood-story")
def neighbourhood_story(req: NeighbourhoodStoryRequest):
    if "address" not in req.data:
        raise HTTPException(status_code=400, detail="Missing 'address' in data")

    persona = req.persona or "algemeen huishouden"

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

Persona: {persona}

Data (JSON):
{req.data}

Schrijf nu:

1) Een titel van max 60 tekens.
2) Een korte samenvattende intro (1–2 zinnen).
3) Kopje "Pluspunten" met 3 bullets.
4) Kopje "Let op" met 3 bullets.
"""

    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Je bent een behulpzame, neutrale Nederlandse tekstschrijver."},
                {"role": "user", "content": prompt},
            ],
        )
        story = completion.choices[0].message.content
    except Exception as e:
        print("OpenAI error:", e)
        raise HTTPException(status_code=500, detail="AI-fout bij genereren buurtverhaal")

    return {"story": story}

# ---------- Static React build ----------

if FRONTEND_DIST.exists():
    app.mount(
        "/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="static"

    )
else:
    print("⚠️ Let op: frontend/dist bestaat niet, alleen API beschikbaar.")

@app.get("/api/health")
def health():
    return {"status": "ok"}
