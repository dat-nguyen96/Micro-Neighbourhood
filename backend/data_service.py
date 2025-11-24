# backend/data_service.py - CBS Data Loading and Processing
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

# Path to precomputed CSV with features + clusters
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CLUSTERS_CSV = DATA_DIR / "clusters.csv"

FEATURE_COLUMNS = [
    "AantalInwoners_5",
    "Bevolkingsdichtheid_34",
    "GemiddeldeHuishoudensgrootte_33",
    "HuishoudensMetKinderen_32",
    "HuishoudensTotaal_29",
    "MateVanStedelijkheid_122",
    "k_0Tot15Jaar_8",
    "k_25Tot45Jaar_10",
    "k_65JaarOfOuder_12",
    "crime_violence_total"  # Totaal geweld (incl. seksueel)
]

# Global cache for loaded data
_cbs_data = None


def _ensure_cbs_data_loaded():
    """
    Laad de precomputed CBS data met clusters als deze nog niet geladen is.
    Raises RuntimeError bij problemen.
    """
    global _cbs_data

    if _cbs_data is not None:
        return _cbs_data

    if not CLUSTERS_CSV.exists():
        raise RuntimeError(f"Precomputed clusters CSV niet gevonden: {CLUSTERS_CSV}")

    try:
        df = pd.read_csv(CLUSTERS_CSV, low_memory=False)
        print(f"📊 CBS data geladen: {len(df)} rijen")

        # Check required columns
        required_cols = ["WijkenEnBuurten"] + FEATURE_COLUMNS + ["cluster_id", "cluster_label", "cluster_label_long"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Ontbrekende kolommen in CSV: {missing}")

        # Hernoem cluster_label naar cluster_label_short voor consistentie
        if "cluster_label" in df.columns:
            df["cluster_label_short"] = df["cluster_label"]
        else:
            df["cluster_label_short"] = "Onbekend"

        _cbs_data = df
        return df

    except Exception as e:
        raise RuntimeError(f"Fout bij laden CBS data: {e}")


def _find_buurt_index(buurt_code: str) -> int:
    """
    Vind de index van een buurt in de CBS data op basis van buurt_code.
    Raises ValueError als buurt niet gevonden wordt.
    """
    df = _ensure_cbs_data_loaded()

    # Zoek op WijkenEnBuurten kolom (CBS buurt codes)
    mask = df["WijkenEnBuurten"] == buurt_code
    matches = df[mask]

    if len(matches) == 0:
        raise ValueError(f"Buurt code '{buurt_code}' niet gevonden in CBS data")
    elif len(matches) > 1:
        print(f"⚠️ Meerdere matches voor buurt code '{buurt_code}', neem eerste")

    return matches.index[0]


def _get_cluster_info(cluster_id: int) -> Dict[str, str]:
    """
    Haal cluster informatie op voor een gegeven cluster_id.
    """
    df = _ensure_cbs_data_loaded()

    # Vind een sample rij voor dit cluster
    cluster_rows = df[df["cluster_id"] == cluster_id]
    if len(cluster_rows) == 0:
        return {"label_short": f"Cluster {cluster_id}", "label_long": "Geen data beschikbaar"}

    sample_row = cluster_rows.iloc[0]

    # Zorg voor veilige extractie van cluster labels
    label_short = sample_row.get("cluster_label_short")
    if pd.isna(label_short) or label_short is None:
        label_short = sample_row.get("cluster_label", f"Cluster {cluster_id}")

    label_long = sample_row.get("cluster_label_long", "")
    if pd.isna(label_long) or label_long is None:
        label_long = ""

    return {
        "label_short": str(label_short),
        "label_long": str(label_long)
    }
