# backend/offline/build_clusters.py

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import httpx
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]  # backend/
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_CSV = DATA_DIR / "cbs_buurten_raw.csv"
OUT_CSV = DATA_DIR / "buurten_features_clusters.csv"


# ============================================================
# Load environment & OpenAI client
# ============================================================

env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ============================================================
# Feature list
# ============================================================

FEATURE_COLUMNS: List[str] = [
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


# ============================================================
# CBS Downloader (pagination)
# ============================================================

def fetch_cbs_to_csv():
    """
    Download ALL rows from CBS 83765NED/TypedDataSet with pagination.
    Saves them to RAW_CSV.
    """
    print("Fetching CBS 83765NED to CSV with pagination...")

    base_url = "https://opendata.cbs.nl/ODataApi/OData/83765NED/TypedDataSet"
    all_rows = []
    skip = 0
    batch = 1000

    with httpx.Client(timeout=30.0) as http:
        while True:
            resp = http.get(base_url, params={"$skip": skip, "$top": batch})
            resp.raise_for_status()

            data = resp.json()["value"]
            if not data:
                break

            all_rows.extend(data)
            print(f"Fetched {len(all_rows)} rows...")

            if len(data) < batch:
                break

            skip += batch

    df = pd.DataFrame(all_rows)
    df.to_csv(RAW_CSV, index=False)
    print(f"Saved {len(df)} rows to {RAW_CSV}")


# ============================================================
# Helper to summarize features for the prompt
# ============================================================

def summarize_cluster_for_prompt(df_cluster: pd.DataFrame) -> str:
    desc = df_cluster[FEATURE_COLUMNS].describe(percentiles=[0.5]).T

    lines = []
    for col, row in desc.iterrows():
        lines.append(
            f"- {col}: gemiddelde ≈ {row['mean']:.1f}, mediaan ≈ {row['50%']:.1f}"
        )
    return "\n".join(lines)


# ============================================================
# Robust LLM response cleaner
# ============================================================

def _strip_code_fences(text: str) -> str:
    """
    Remove ``` and ```json fences from the LLM response.
    This makes JSON parsing stable.
    """
    if not text:
        return ""

    cleaned = text.strip()

    # Remove ```json, ```python, etc.
    cleaned = re.sub(r"```[a-zA-Z]*", "", cleaned)
    # Remove ending fences ```
    cleaned = cleaned.replace("```", "")

    return cleaned.strip()


# ============================================================
# Ask LLM for a label
# ============================================================

def llm_label_for_cluster(cluster_id: int, df_cluster: pd.DataFrame) -> Dict[str, str]:
    """
    Uses OpenAI to generate a short label and a one-sentence explanation.
    Results will be cached into the final CSV so this is only run once.
    """

    if client is None:
        # fallback for offline / missing API key
        return {
            "label_short": f"Cluster {cluster_id}",
            "label_long": "Geen LLM-label beschikbaar (OPENAI_API_KEY ontbreekt).",
        }

    summary = summarize_cluster_for_prompt(df_cluster)

    prompt = f"""
Je bent een data-analist in Nederland. Je krijgt gemiddelde cijfers voor een cluster
van Nederlandse buurten. Geef:

1) label_short (max 4 woorden)
2) label_long (max 25 woorden)

Antwoord uitsluitend in JSON met deze velden:
- label_short
- label_long

Cijfers:
{summary}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Je bent een Nederlandse data-analist."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
    )

    raw = resp.choices[0].message.content or "{}"
    cleaned = _strip_code_fences(raw)

    try:
        obj = json.loads(cleaned)
        short = obj.get("label_short", f"Cluster {cluster_id}")
        long = obj.get("label_long", "")
        return {
            "label_short": short.strip(),
            "label_long": long.strip(),
        }
    except Exception:
        # fallback heuristic
        lines = cleaned.splitlines()
        if lines:
            short = lines[0].strip().strip("{} ")
            long = " ".join(lines[1:]).strip()
            return {
                "label_short": short or f"Cluster {cluster_id}",
                "label_long": long or "",
            }

        return {
            "label_short": f"Cluster {cluster_id}",
            "label_long": cleaned[:200],
        }


# ============================================================
# MAIN: build clusters
# ============================================================

def main():
    # --------------------------------------------------------
    # Step 1 — download CBS raw data if missing
    # --------------------------------------------------------
    if not RAW_CSV.exists():
        fetch_cbs_to_csv()

    print(f"Loading {RAW_CSV} ...")
    df = pd.read_csv(RAW_CSV)

    # --------------------------------------------------------
    # Step 2 — filter only buurten
    # --------------------------------------------------------
    df_buurten = df[df["SoortRegio_2"].str.strip() == "Buurt"].copy()
    print(f"Total rows: {len(df)}, buurten: {len(df_buurten)}")

    # --------------------------------------------------------
    # Step 3 — clean missing values
    # --------------------------------------------------------
    missing = [c for c in FEATURE_COLUMNS if c not in df_buurten.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    df_features = df_buurten[FEATURE_COLUMNS]
    mask = df_features.notna().all(axis=1)

    df_clean = df_buurten[mask].reset_index(drop=True)
    features = df_clean[FEATURE_COLUMNS].astype(float)
    print(f"Rows after cleaning: {len(df_clean)}")

    # --------------------------------------------------------
    # Step 4 — scale + KMeans
    # --------------------------------------------------------
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    n_clusters = 8
    print(f"Fitting KMeans (k={n_clusters})...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    df_clean["cluster_id"] = km.fit_predict(X)

    # --------------------------------------------------------
    # Step 5 — LLM labeling (once per cluster)
    # --------------------------------------------------------
    cluster_labels: Dict[int, Dict[str, str]] = {}
    for cid in sorted(df_clean["cluster_id"].unique()):
        df_cluster = df_clean[df_clean["cluster_id"] == cid]
        print(f"Labeling cluster {cid} ({len(df_cluster)} rows)...")
        cluster_labels[cid] = llm_label_for_cluster(cid, df_cluster)

    df_clean["cluster_label_short"] = df_clean["cluster_id"].map(
        lambda cid: cluster_labels[cid]["label_short"]
    )
    df_clean["cluster_label_long"] = df_clean["cluster_id"].map(
        lambda cid: cluster_labels[cid]["label_long"]
    )

    # --------------------------------------------------------
    # Step 6 — export trimmed CSV for backend
    # --------------------------------------------------------
    keep = [
        "WijkenEnBuurten",
        "Gemeentenaam_1",
        "SoortRegio_2",
        "Codering_3",
        *FEATURE_COLUMNS,
        "cluster_id",
        "cluster_label_short",
        "cluster_label_long",
    ]

    out = df_clean[keep].copy()
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(out)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
