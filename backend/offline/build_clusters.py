# backend/offline/build_clusters.py

import os
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import httpx
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---- Paths ----

BASE_DIR = Path(__file__).resolve().parents[1]  # backend/

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_CSV = DATA_DIR / "cbs_buurten_raw.csv"
OUT_CSV = DATA_DIR / "buurten_features_clusters.csv"

# ---- Env / OpenAI ----

env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

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

# ---- Optional helper: fetch CBS once into RAW_CSV ----

def fetch_cbs_to_csv():
    """
    Haal CBS 83765NED (alle buurten) op en schrijf naar cbs_buurten_raw.csv.
    Gebruikt pagination om de 10k limiet te omzeilen.
    """
    print("Fetching CBS 83765NED to CSV with pagination...")

    base_url = "https://opendata.cbs.nl/ODataApi/OData/83765NED/TypedDataSet"
    all_data = []
    skip = 0
    batch_size = 1000  # Kleiner dan 10k limiet

    with httpx.Client(timeout=30.0) as client:
        while True:
            params = {
                "$skip": skip,
                "$top": batch_size,
            }
            resp = client.get(base_url, params=params)
            resp.raise_for_status()
            data = resp.json()["value"]

            if not data:  # Geen data meer
                break

            all_data.extend(data)
            print(f"Fetched {len(all_data)} records so far...")

            if len(data) < batch_size:  # Laatste batch
                break

            skip += batch_size

    df = pd.DataFrame(all_data)
    df.to_csv(RAW_CSV, index=False)
    print(f"Wrote {len(df)} rows to {RAW_CSV}")


def summarize_cluster_for_prompt(df_cluster: pd.DataFrame) -> str:
    """
    Maak een korte numerieke samenvatting per cluster voor in de LLM prompt.
    Houd het compact, anders wordt de prompt te groot.
    """
    desc = df_cluster[FEATURE_COLUMNS].describe(percentiles=[0.5]).T

    lines = []
    for col, row in desc.iterrows():
        mean_val = row["mean"]
        p50 = row["50%"]
        lines.append(f"- {col}: gemiddelde ≈ {mean_val:.1f}, mediaan ≈ {p50:.1f}")

    return "\n".join(lines)


def llm_label_for_cluster(cluster_id: int, df_cluster: pd.DataFrame) -> Dict[str, str]:
    """
    Vraag de LLM om 1 korte label + 1 zin uitleg op basis van cijfers.
    Cache gebeurt door het opslaan in CSV; we roepen dit maar 1x per cluster.
    """
    if client is None:
        # Geen API key, dan fallback labels
        return {
            "label_short": f"Cluster {cluster_id}",
            "label_long": "Geen LLM-label beschikbaar (geen OPENAI_API_KEY gezet).",
        }

    summary = summarize_cluster_for_prompt(df_cluster)

    prompt = f"""
Je bent een data-analist in Nederland. Je krijgt gemiddelde cijfers voor een cluster
van Nederlandse buurten (wijken/buurten). Op basis hiervan geef je:

1) Een KORTE label in maximaal 4 woorden, informeel maar neutraal, bv:
   "jong & stedelijk", "rustig, vergrijzend", "welvarend stadscentrum".
2) Een ENKELE zin (max 25 woorden) die het cluster uitlegt voor een leek.

Gebruik de onderstaande cijfers als context. Antwoord in JSON met exact deze velden:
- label_short
- label_long

Cijfers (per cluster):
{summary}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Je bent een Nederlandse data-analist. Antwoord alleen in JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
    )

    text = resp.choices[0].message.content or "{}"
    # Super-simpele JSON parsing: we proberen eerst via pandas, anders fallback
    import json

    try:
        data = json.loads(text)
        short = str(data.get("label_short", f"Cluster {cluster_id}"))
        long = str(data.get("label_long", ""))
    except Exception:
        short = f"Cluster {cluster_id}"
        long = text[:200]

    return {
        "label_short": short.strip(),
        "label_long": long.strip(),
    }


def main():
    if not RAW_CSV.exists():
        print("RAW_CSV bestaat niet, ik haal eerst CBS-data op...")
        fetch_cbs_to_csv()

    print(f"Loading {RAW_CSV} ...")
    df = pd.read_csv(RAW_CSV)

    # Filter alleen buurten (SoortRegio_2 == "Buurt")
    df_buurten = df[df["SoortRegio_2"].str.strip() == "Buurt"].copy()
    print(f"Total rows: {len(df)}, Buurten only: {len(df_buurten)}")

    # Zorg dat we deze kolommen hebben
    missing = [c for c in FEATURE_COLUMNS if c not in df_buurten.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns in CSV: {missing}")

    # Drop rows met te veel NaN in onze features
    df_features = df_buurten[FEATURE_COLUMNS].copy()
    df_clean = df_buurten.copy()
    mask = df_features.notna().all(axis=1)
    df_clean = df_clean[mask].reset_index(drop=True)
    df_features = df_features[mask].reset_index(drop=True)

    print(f"Rows before cleaning: {len(df_buurten)}, after: {len(df_clean)}")

    # Schalen
    scaler = StandardScaler()
    X = scaler.fit_transform(df_features)

    # KMeans
    n_clusters = 8  # tweakbaar
    print(f"Fitting KMeans with k={n_clusters} ...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    cluster_ids = km.fit_predict(X)

    df_clean["cluster_id"] = cluster_ids

    # LLM labels per cluster
    cluster_labels: Dict[int, Dict[str, str]] = {}

    for cid in sorted(df_clean["cluster_id"].unique()):
        df_cluster = df_clean[df_clean["cluster_id"] == cid]
        print(f"Generating label for cluster {cid} (rows={len(df_cluster)}) ...")
        cluster_labels[cid] = llm_label_for_cluster(cid, df_cluster)

    # Map labels terug naar alle buurten
    df_clean["cluster_label_short"] = df_clean["cluster_id"].map(
        lambda cid: cluster_labels[cid]["label_short"]
    )
    df_clean["cluster_label_long"] = df_clean["cluster_id"].map(
        lambda cid: cluster_labels[cid]["label_long"]
    )

    # Schrijf een dunne CSV met alleen wat we online nodig hebben
    keep_cols = [
        "WijkenEnBuurten",
        "Gemeentenaam_1",
        "SoortRegio_2",
        "Codering_3",
        *FEATURE_COLUMNS,
        "cluster_id",
        "cluster_label_short",
        "cluster_label_long",
    ]

    for col in keep_cols:
        if col not in df_clean.columns:
            print(f"Waarschuwing: kolom {col} ontbreekt, ik sla die over.")
    keep_cols = [c for c in keep_cols if c in df_clean.columns]

    out_df = df_clean[keep_cols].copy()
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out_df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
