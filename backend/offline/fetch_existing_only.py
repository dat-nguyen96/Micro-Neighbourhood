#!/usr/bin/env python3
"""
Offline data pipeline - 2 functies:
1. fetch_all_data() - Haalt alle data op en schrijft naar CSV
2. build_clusters() - Gebruikt de data om clusters te bouwen

Belangrijk:
- Criminaliteit: we gebruiken alleen TOTAAL GEWELD (incl. seksueel geweld),
  gedefinieerd als alle SoortMisdrijf codes die beginnen met '1.4.'
- We maken een rate: violent_crimes_per_1000 = totaal geweld / inwoners * 1000
"""

import os
from pathlib import Path
import time

import httpx
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dotenv import load_dotenv

# Optional: OpenAI client, nu niet gebruikt maar kan later voor clusterbeschrijvingen
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Load environment
env_path = Path("../.env")
if env_path.exists():
    load_dotenv(env_path)

# Paths
DATA_DIR = Path("../data")
RAW_DATA_CSV = DATA_DIR / "raw_data.csv"
CLUSTERS_CSV = DATA_DIR / "clusters.csv"

# OpenAI client (nu ongebruikt, maar gelaten voor uitbreidbaarheid)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if (OpenAI and os.getenv("OPENAI_API_KEY")) else None


def fetch_all_data():
    """
    Haal alle benodigde data op van CBS:
    1. CBS 85984NED buurt data (buurten)
    2. CBS crime data (47018NED) -> totaal geweld (incl. seksueel) per buurt
    3. Merge alles samen en schrijf naar CSV
    """
    print("=== Start data ophalen ===")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # =============================
    # 1. CBS 85984NED - buurten
    # =============================
    print("1. CBS 85984NED (buurt data) ophalen...")
    base_url = "https://opendata.cbs.nl/ODataApi/OData/85984NED/TypedDataSet"
    all_cbs_data = []
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

            all_cbs_data.extend(batch)
            print(f"  -> {len(all_cbs_data)} rijen opgehaald...")
            if len(batch) < batch_size:
                break

            skip += batch_size

    cbs_raw_df = pd.DataFrame(all_cbs_data)
    print(f"  Totaal rijen (alle regio's): {len(cbs_raw_df)}")

    # Filter alleen buurten
    print("2. Filter alleen buurten...")
    cbs_df = cbs_raw_df[cbs_raw_df["SoortRegio_2"].str.strip() == "Buurt"].copy()
    print(f"  Na filtering buurten: {len(cbs_df)} rijen")

    # =============================
    # 2. CBS 47018NED - misdrijven
    # =============================
    print("3. CBS 47018NED (crime) ophalen...")

    crime_url = "https://dataderden.cbs.nl/ODataApi/OData/47018NED/TypedDataSet"

    # 2a. Beschikbare perioden
    print("  -> Controleren beschikbare perioden...")
    meta_url = "https://dataderden.cbs.nl/ODataApi/OData/47018NED/Perioden"
    resp_meta = requests.get(meta_url)
    resp_meta.raise_for_status()
    periods = [r["Key"] for r in resp_meta.json()["value"]]
    print(f"  -> Eerste periodes: {periods[:10]}")

    # Kies meest recente jaarcijfer (JJ00) met gedetailleerde data
    print("  -> Zoeken naar periode met gedetailleerde crime data...")
    crime_period = None
    for period in reversed(periods):
        if not period.endswith("JJ00"):
            continue

        print(f"    Testen periode {period}...")
        test_params = {
            "$filter": f"Perioden eq '{period}' and SoortMisdrijf ne '0.0.0 '",
            "$top": 1,
        }
        test_resp = requests.get(crime_url, params=test_params)
        if test_resp.status_code != 200:
            print(f"    ❌ Error {test_resp.status_code}")
            continue

        test_data = test_resp.json().get("value", [])
        if test_data:
            crime_period = period
            print(f"    ✅ Periode {period} heeft gedetailleerde data")
            break
        else:
            print(f"    ❌ Periode {period} heeft alleen totalen")

        time.sleep(0.2)

    if not crime_period:
        print("  -> Geen periode met gedetailleerde data gevonden, gebruik meest recente JJ00")
        crime_period = next((p for p in reversed(periods) if p.endswith("JJ00")), periods[-1])

    print(f"  -> Gekozen crime periode: {crime_period}")

    # 2b. SoortMisdrijf mapping ophalen
    print("  -> SoortMisdrijf mapping ophalen...")
    misdrijf_url = "https://dataderden.cbs.nl/ODataApi/OData/47018NED/SoortMisdrijf"
    misdrijf_resp = requests.get(misdrijf_url)
    misdrijf_resp.raise_for_status()
    misdrijf_data = misdrijf_resp.json()["value"]

    # We definiëren geweld: ALLE 1.4.x (incl. 1.4.1, 1.4.2 -> seksueel geweld)
    violent_keys_raw = []
    for item in misdrijf_data:
        raw_key = item["Key"]      # bevat trailing spatie
        key = raw_key.strip()      # voor prefix check
        if key.startswith("1.4."):
            violent_keys_raw.append(raw_key)

    violent_keys_raw = list(set(violent_keys_raw))
    print(f"  -> {len(violent_keys_raw)} geweld-categorieën gevonden (1.4.x)")
    print(f"     Voorbeelden: {[k.strip() for k in violent_keys_raw[:5]]}")

    # 2c. Data ophalen voor alle geweld categorieën
    print("  -> Ophalen misdrijvendata voor geweldscategorieën...")
    crime_rows = []

    for cat in violent_keys_raw:
        print(f"    Categorie {cat.strip()} ophalen...")
        cat_skip = 0
        while True:
            params = {
                "$filter": f"Perioden eq '{crime_period}' and SoortMisdrijf eq '{cat}'",
                "$skip": cat_skip,
                "$top": batch_size,
            }
            resp = requests.get(crime_url, params=params)
            if resp.status_code != 200:
                print(f"      ❌ HTTP {resp.status_code}, stoppen voor deze categorie.")
                break

            data = resp.json()
            batch = data.get("value", [])
            if not batch:
                break

            crime_rows.extend(batch)
            if len(batch) < batch_size:
                break

            cat_skip += batch_size

        time.sleep(0.3)

    print(f"  -> Totaal misdrijfrijen (geweld): {len(crime_rows)}")

    if not crime_rows:
        print("  ⚠️ Geen misdrijvendata gevonden, pipeline gaat verder zonder crime feature.")
        merged_df = cbs_df.copy()
        merged_df["violent_crimes_total"] = 0
        merged_df["violent_crimes_per_1000"] = 0.0
    else:
        crime_df = pd.DataFrame(crime_rows)
        print(f"  Crime columns: {list(crime_df.columns)}")

        # 2d. Bepaal regiocode-kolom en misdrijf-kolom
        regio_col = None
        for col in ["WijkenEnBuurten", "RegioS", "Regiocode"]:
            if col in crime_df.columns:
                regio_col = col
                break

        if regio_col is None:
            raise RuntimeError("Geen regiokolom gevonden in crime data")

        crime_col = None
        for col in ["GeregistreerdeMisdrijven_1", "TotaalMisdrijven", "MisdrijvenTotaal"]:
            if col in crime_df.columns:
                crime_col = col
                break

        if crime_col is None:
            raise RuntimeError("Geen misdrijf-telkolom gevonden in crime data")

        print(f"  -> Gebruik regio kolom: {regio_col}, misdrijf kolom: {crime_col}")

        # 2e. Totaal geweld per buurt (som over alle 1.4.x)
        crime_agg = (
            crime_df.groupby(regio_col)[crime_col]
            .sum()
            .reset_index()
            .rename(columns={regio_col: "WijkenEnBuurten", crime_col: "violent_crimes_total"})
        )

        print(f"  -> {len(crime_agg)} buurten met geweld-data")

        # 2f. Merge met buurt-data
        merged_df = cbs_df.merge(crime_agg, on="WijkenEnBuurten", how="left")
        merged_df["violent_crimes_total"] = merged_df["violent_crimes_total"].fillna(0)

        # 2g. Maak rate per 1.000 inwoners
        if "AantalInwoners_5" in merged_df.columns:
            merged_df["violent_crimes_per_1000"] = (
                merged_df["violent_crimes_total"] / merged_df["AantalInwoners_5"].replace(0, pd.NA)
            ) * 1000
            merged_df["violent_crimes_per_1000"] = merged_df["violent_crimes_per_1000"].fillna(0)
        else:
            print("  ⚠️ Kolom 'AantalInwoners_5' niet gevonden, geen rate mogelijk.")
            merged_df["violent_crimes_per_1000"] = 0.0

    # =============================
    # 3. Schrijf naar CSV
    # =============================
    print("4. Eindresultaat schrijven naar CSV...")
    merged_df.to_csv(RAW_DATA_CSV, index=False)
    print(f"=== Gedetailleerde data geschreven naar {RAW_DATA_CSV} ({len(merged_df)} rijen) ===")


def build_clusters():
    """
    Gebruik de opgehaalde data om KMeans clusters te bouwen.
    We gebruiken demografische kenmerken + violent_crimes_per_1000 als enige crime feature.
    Clusterlabels worden data-gedreven toegekend (niet hard gekoppeld aan cluster-ID).
    """
    print("=== Start cluster bouw ===")

    df = pd.read_csv(RAW_DATA_CSV)
    print(f"Data geladen: {len(df)} rijen")

    # Basis feature kolommen
    feature_cols = [
        "AantalInwoners_5",
        "Bevolkingsdichtheid_34",
        "GemiddeldeHuishoudensgrootte_33",
        "HuishoudensMetKinderen_32",
        "HuishoudensTotaal_29",
        "MateVanStedelijkheid_122",
        "k_0Tot15Jaar_8",
        "k_15Tot25Jaar_9",
        "k_25Tot45Jaar_10",
        "k_45Tot65Jaar_11",
        "k_65JaarOfOuder_12",
    ]

    # Voeg onze enige crime feature toe
    if "violent_crimes_per_1000" in df.columns:
        feature_cols.append("violent_crimes_per_1000")
        print("Feature 'violent_crimes_per_1000' toegevoegd.")
    else:
        print("⚠️ 'violent_crimes_per_1000' niet gevonden, clustering zonder crime-feature.")
    
    # Filter rijen met complete features
    df_features = df[feature_cols].copy()
    mask = df_features.notna().all(axis=1)
    df_clean = df[mask].reset_index(drop=True)
    df_features = df_features[mask].reset_index(drop=True)

    print(f"Na cleaning: {len(df_clean)} rijen")
    print(f"Features gebruikt: {feature_cols}")

    # Schalen
    scaler = StandardScaler()
    X = scaler.fit_transform(df_features)

    # KMeans
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X)

    # PCA voor visualisatie
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # -----------------------------
    # Data-gedreven clusterlabels
    # -----------------------------
    cluster_stats = df_clean.copy()
    cluster_stats["cluster_id"] = clusters

    # Gemiddelde stedelijkheid en geweld per cluster
    stats = (
        cluster_stats.groupby("cluster_id")[["MateVanStedelijkheid_122", "violent_crimes_per_1000"]]
        .mean()
        .reset_index()
    )

    # Hoog/laag geweld & stedelijkheid bepalen
    # (simpel: sorteren en ranken)
    stats["violence_rank"] = stats["violent_crimes_per_1000"].rank(method="dense")
    stats["urban_rank"] = stats["MateVanStedelijkheid_122"].rank(method="dense")

    # Nu eenvoudige heuristiek voor labels:
    # - hoogste violence & hoogste urban -> "Stedelijk & Risico"
    # - laagste violence & laagste urban -> "Landelijk & Rustig"
    # - hoge urban, lage violence -> "Modern & Veilig"
    # - etc.

    # Cluster id met hoogste violence & urban
    high_risk = stats.sort_values(["violence_rank", "urban_rank"], ascending=False).iloc[0]["cluster_id"]
    # Laagste violence & urban
    rural_safe = stats.sort_values(["violence_rank", "urban_rank"], ascending=True).iloc[0]["cluster_id"]
    # Hoge urban, lage violence
    urban_safe = stats.sort_values(["urban_rank", "violence_rank"], ascending=[False, True]).iloc[0]["cluster_id"]
    # Lage urban, lage violence maar wat ouder
    rural_older = stats.sort_values(["urban_rank", "violence_rank"], ascending=[True, True]).iloc[0]["cluster_id"]

    # Basismapping, overige clusters krijgen generieke namen
    cluster_short_names = {}
    for cid in stats["cluster_id"]:
        cluster_short_names[cid] = f"Cluster {cid}"  # default

    cluster_short_names[high_risk] = "Stedelijk & Risico"
    cluster_short_names[rural_safe] = "Landelijk & Rustig"
    cluster_short_names[urban_safe] = "Modern & Veilig"
    cluster_short_names[rural_older] = "Vergrijzend & Groen"

    print("Clusterlabels (data-gedreven) toegewezen:")
    for _, row in stats.iterrows():
        cid = int(row["cluster_id"])
        print(
            f"  Cluster {cid}: {cluster_short_names[cid]} "
            f"(stedelijkheid ~ {row['MateVanStedelijkheid_122']:.1f}, "
            f"geweld/1000 ~ {row['violent_crimes_per_1000']:.2f})"
        )

    # Resultaat dataframe
    result_df = df_clean.copy()
    result_df["cluster_id"] = clusters
    result_df["cluster_label"] = result_df["cluster_id"].map(cluster_short_names)
    result_df["pca_x"] = X_pca[:, 0]
    result_df["pca_y"] = X_pca[:, 1]

    # Schrijf naar CSV
    result_df.to_csv(CLUSTERS_CSV, index=False)
    print(f"=== Clusters geschreven naar {CLUSTERS_CSV} ({len(result_df)} rijen, {n_clusters} clusters) ===")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "fetch":
            fetch_all_data()
        elif command == "build":
            build_clusters()
        else:
            print("Gebruik: python script.py [fetch|build]")
    else:
        print("Gebruik:")
        print("  python script.py fetch    # Haal data op")
        print("  python script.py build    # Bouw clusters")
