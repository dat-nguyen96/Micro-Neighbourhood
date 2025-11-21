# backend/offline/build_clusters.py

import os
from pathlib import Path
from typing import Dict, List

import json
import httpx
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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

# ---- CBS fetch helper (eenmalig offline) ----

def fetch_cbs_to_csv():
  """
  Haal CBS 83765NED (alle buurten) op en schrijf naar cbs_buurten_raw.csv.
  """
  print("Fetching CBS 83765NED to CSV with pagination...")
  base_url = "https://opendata.cbs.nl/ODataApi/OData/83765NED/TypedDataSet"
  all_data = []
  skip = 0
  batch_size = 1000

  with httpx.Client(timeout=60.0) as c:
      while True:
          params = {"$skip": skip, "$top": batch_size}
          r = c.get(base_url, params=params)
          r.raise_for_status()
          batch = r.json()["value"]
          if not batch:
              break
          all_data.extend(batch)
          print(f"Fetched {len(all_data)} rows...")
          if len(batch) < batch_size:
              break
          skip += batch_size

  df = pd.DataFrame(all_data)
  df.to_csv(RAW_CSV, index=False)
  print(f"Wrote {len(df)} rows to {RAW_CSV}")


# ---- LLM helpers ----

def summarize_cluster_for_prompt(df_cluster: pd.DataFrame) -> str:
    """
    Korte numerieke samenvatting voor in de LLM prompt.
    """
    desc = df_cluster[FEATURE_COLUMNS].describe(percentiles=[0.5]).T
    lines = []
    for col, row in desc.iterrows():
        mean_val = row["mean"]
        p50 = row["50%"]
        lines.append(f"- {col}: gemiddelde ≈ {mean_val:.1f}, mediaan ≈ {p50:.1f}")
    return "\n".join(lines)


def parse_label_response(text: str, cluster_id: int) -> Dict[str, str]:
    """
    Verwacht antwoord in twee regels:
      label_short: ...
      label_long: ...
    Haalt eventuele code fences ``` weg en splitst.
    """
    # strip eventuele ```json fences
    text = text.strip()
    if text.startswith("```"):
        # verwijder eerste en laatste ```-blok
        parts = text.split("```")
        # neem alles tussen eerste en laatste fence
        text = "\n".join(p for p in parts[1:-1]).strip() or text

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    label_short = f"Cluster {cluster_id}"
    label_long = ""

    for line in lines:
        if line.lower().startswith("label_short"):
            _, val = line.split(":", 1)
            label_short = val.strip()
        elif line.lower().startswith("label_long"):
            _, val = line.split(":", 1)
            label_long = val.strip()

    return {
        "label_short": label_short or f"Cluster {cluster_id}",
        "label_long": label_long or "",
    }


def llm_label_for_cluster(cluster_id: int, df_cluster: pd.DataFrame) -> Dict[str, str]:
    """
    Vraag de LLM om 1 korte label + 1 zin uitleg per cluster.
    Antwoord wordt netjes opgeschoond (geen ruwe JSON/fences).
    """
    if client is None:
        return {
            "label_short": f"Cluster {cluster_id}",
            "label_long": "Geen LLM-label beschikbaar (geen OPENAI_API_KEY gezet).",
        }

    summary = summarize_cluster_for_prompt(df_cluster)

    prompt = f"""
Je bent een Nederlandse data-analist. Je krijgt cijfers voor een cluster
van vergelijkbare buurten. Geef een korte duiding.

Schrijf je antwoord EXACT in twee regels:

label_short: <max 4 woorden, informeel maar neutraal, bv. "jong & stedelijk", "rustig, vergrijzend">
label_long: <max 25 woorden, één zin uitleg in het Nederlands>

Gebruik onderstaande cijfers als context:

{summary}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Je bent een Nederlandse data-analist.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
    )

    text = resp.choices[0].message.content or ""
    labels = parse_label_response(text, cluster_id)
    return labels


# ---- main pipeline ----

def main():
    if not RAW_CSV.exists():
        print("RAW_CSV bestaat niet, ik haal eerst CBS-data op...")
        fetch_cbs_to_csv()

    print(f"Loading {RAW_CSV} ...")
    df = pd.read_csv(RAW_CSV)

    # Alleen buurten
    df_buurten = df[df["SoortRegio_2"].str.strip() == "Buurt"].copy()
    print(f"Total rows: {len(df)}, buurten only: {len(df_buurten)}")

    missing = [c for c in FEATURE_COLUMNS if c not in df_buurten.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns in CSV: {missing}")

    df_features = df_buurten[FEATURE_COLUMNS].copy()
    mask = df_features.notna().all(axis=1)
    df_clean = df_buurten[mask].reset_index(drop=True)
    df_features = df_features[mask].reset_index(drop=True)

    print(f"Rows before cleaning: {len(df_buurten)}, after: {len(df_clean)}")

    scaler = StandardScaler()
    X = scaler.fit_transform(df_features)

    # KMeans
    n_clusters = 8
    print(f"Fitting KMeans with k={n_clusters} ...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    cluster_ids = km.fit_predict(X)
    df_clean["cluster_id"] = cluster_ids

    # PCA 2D voor visualisatie
    print("Computing PCA(2) for visualisation ...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    df_clean["pca_x"] = X_pca[:, 0]
    df_clean["pca_y"] = X_pca[:, 1]

    # LLM labels per cluster
    cluster_labels: Dict[int, Dict[str, str]] = {}
    for cid in sorted(df_clean["cluster_id"].unique()):
        df_cluster = df_clean[df_clean["cluster_id"] == cid]
        print(f"Generating label for cluster {cid} (rows={len(df_cluster)}) ...")
        cluster_labels[cid] = llm_label_for_cluster(cid, df_cluster)
        print(
            f"  -> {cluster_labels[cid]['label_short']} "
            f" / {cluster_labels[cid]['label_long']}"
        )

    df_clean["cluster_label_short"] = df_clean["cluster_id"].map(
        lambda cid: cluster_labels[cid]["label_short"]
    )
    df_clean["cluster_label_long"] = df_clean["cluster_id"].map(
        lambda cid: cluster_labels[cid]["label_long"]
    )

    # Dunne CSV voor runtime
    keep_cols = [
        "WijkenEnBuurten",
        "Gemeentenaam_1",
        "SoortRegio_2",
        "Codering_3",
        *FEATURE_COLUMNS,
        "cluster_id",
        "cluster_label_short",
        "cluster_label_long",
        "pca_x",
        "pca_y",
    ]
    keep_cols = [c for c in keep_cols if c in df_clean.columns]

    out_df = df_clean[keep_cols].copy()
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out_df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
