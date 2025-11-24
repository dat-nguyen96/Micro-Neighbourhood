#!/usr/bin/env python3
"""
Offline data pipeline - 2 functies:
1. fetch_all_data() - Haalt alle data op en schrijft naar CSV
2. build_clusters() - Gebruikt de data om clusters te bouwen
"""

import pandas as pd
import httpx
import requests
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment
env_path = Path("../.env")
if env_path.exists():
    load_dotenv(env_path)

# Paths
DATA_DIR = Path("../data")
RAW_DATA_CSV = DATA_DIR / "raw_data.csv"
CLUSTERS_CSV = DATA_DIR / "clusters.csv"

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

def fetch_all_data():
    """
    Haal alle benodigde data op van CBS en schrijf naar één CSV bestand
    """
    print("=== Start data ophalen ===")

    all_data = []

    # 1. CBS 85984NED (buurt data)
    print("1. CBS 85984NED ophalen...")
    base_url = "https://opendata.cbs.nl/ODataApi/OData/85984NED/TypedDataSet"
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
            print(f"  -> {len(all_data)} rijen opgehaald...")
            if len(batch) < batch_size:
                break
            skip += batch_size

    # 2. CBS 47018NED (crime data)
    print("2. CBS 47018NED (crime) ophalen...")
    crime_url = "https://dataderden.cbs.nl/ODataApi/OData/47018NED/TypedDataSet"
    crime_data = []
    skip = 0

    while True:
        params = {"$filter": "Perioden eq '2024JJ00'", "$skip": skip, "$top": batch_size}
        if skip == 0:
            resp = requests.get(crime_url, params=params)
        else:
            next_url = f"{crime_url}?$filter=Perioden eq '2024JJ00'&$skip={skip}&$top={batch_size}"
            resp = requests.get(next_url)

        resp.raise_for_status()
        data = resp.json()
        batch = data.get("value", [])
        if not batch:
            break
        crime_data.extend(batch)
        print(f"  -> {len(crime_data)} crime rijen opgehaald...")
        if len(batch) < batch_size:
            break
        skip += batch_size

    # Combineer data
    print("3. Data combineren...")
    cbs_df = pd.DataFrame(all_data)

    # Alleen buurten (BU codes)
    cbs_df = cbs_df[cbs_df["SoortRegio_2"] == "Buurt"].copy()

    # Voeg crime data toe
    crime_df = pd.DataFrame(crime_data)
    crime_agg = crime_df.groupby("WijkenEnBuurten")["GeregistreerdeMisdrijven_1"].sum().reset_index()
    crime_agg = crime_agg.rename(columns={"GeregistreerdeMisdrijven_1": "total_crimes"})

    # Merge
    merged_df = cbs_df.merge(crime_agg, on="WijkenEnBuurten", how="left")
    merged_df["total_crimes"] = merged_df["total_crimes"].fillna(0)

    # Schrijf naar CSV
    merged_df.to_csv(RAW_DATA_CSV, index=False)
    print(f"=== Data geschreven naar {RAW_DATA_CSV} ({len(merged_df)} rijen) ===")

def build_clusters():
    """
    Gebruik de opgehaalde data om KMeans clusters te bouwen
    """
    print("=== Start cluster bouw ===")

    # Lees data
    df = pd.read_csv(RAW_DATA_CSV)
    print(f"Data geladen: {len(df)} rijen")

    # Feature kolommen
    feature_cols = [
        "AantalInwoners_5", "Bevolkingsdichtheid_34", "GemiddeldeHuishoudensgrootte_33",
        "HuishoudensMetKinderen_32", "HuishoudensTotaal_29", "MateVanStedelijkheid_122",
        "k_0Tot15Jaar_8", "k_15Tot25Jaar_9", "k_25Tot45Jaar_10", "k_45Tot65Jaar_11", "k_65JaarOfOuder_12",
        "total_crimes"
    ]

    # Filter rijen met alle features
    df_features = df[feature_cols].copy()
    mask = df_features.notna().all(axis=1)
    df_clean = df[mask].reset_index(drop=True)
    df_features = df_features[mask].reset_index(drop=True)

    print(f"Na cleaning: {len(df_clean)} rijen")

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(df_features)

    # KMeans
    n_clusters = 12
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X)

    # PCA voor visualisatie
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # Labels genereren
    cluster_labels = {}
    for i in range(n_clusters):
        cluster_data = df_features[clusters == i]
        if client:
            # Genereer korte beschrijving met LLM
            summary = f"Cluster {i}: {len(cluster_data)} buurten, "
            summary += f"gemiddelde inwoners: {cluster_data['AantalInwoners_5'].mean():.0f}, "
            summary += f"stedelijkheid: {cluster_data['MateVanStedelijkheid_122'].mean():.1f}"

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Geef een korte Nederlandse beschrijving (max 20 woorden) van dit buurt cluster."},
                        {"role": "user", "content": summary}
                    ],
                    max_tokens=50
                )
                label = response.choices[0].message.content.strip()
            except:
                label = f"Cluster {i}"
        else:
            label = f"Cluster {i}"

        cluster_labels[i] = label

    # Resultaat dataframe
    result_df = df_clean.copy()
    result_df["cluster_id"] = clusters
    result_df["cluster_label"] = result_df["cluster_id"].map(cluster_labels)
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
            print("Gebruik: python fetch_existing_only.py [fetch|build]")
    else:
        print("Gebruik:")
        print("  python fetch_existing_only.py fetch    # Haal data op")
        print("  python fetch_existing_only.py build    # Bouw clusters")
