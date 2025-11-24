#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

# Test script to debug cluster loading
DATA_DIR = Path(__file__).parent / "data"
CLUSTERS_CSV = DATA_DIR / "buurten_features_clusters_with_crime_2024.csv"

print(f"Looking for file: {CLUSTERS_CSV}")
print(f"File exists: {CLUSTERS_CSV.exists()}")

if CLUSTERS_CSV.exists():
    print("Loading CSV...")
    df = pd.read_csv(CLUSTERS_CSV, nrows=5)
    print("Columns:", list(df.columns))
    print("Sample row:")
    print(df.iloc[0])

    # Check required columns
    required_cols = ["WijkenEnBuurten", "cluster_id", "cluster_label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
    else:
        print("All required columns present")

    # Test cluster label splitting
    if "cluster_label" in df.columns:
        sample_label = df["cluster_label"].iloc[0]
        print(f"Sample cluster_label: {sample_label}")

        # Extract short label
        import re
        match = re.search(r'^(Cluster \d+)', str(sample_label))
        short_label = match.group(1) if match else f"Cluster {df['cluster_id'].iloc[0]}"
        print(f"Extracted short label: {short_label}")

else:
    print("CSV file not found!")
