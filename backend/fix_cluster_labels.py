#!/usr/bin/env python3
"""
Script om cluster labels te repareren als ze niet correct zijn opgeslagen
"""

import pandas as pd
from pathlib import Path

def main():
    """Hoofdprogramma om cluster labels te repareren"""

    data_dir = Path(__file__).parent / "data"
    input_file = data_dir / "clusters_improved.csv"

    print(f"Repareren van {input_file}...")

    # Lees de huidige data
    df = pd.read_csv(input_file)

    print(f"Gevonden {len(df)} rijen")

    # Controleer huidige cluster kolommen
    cluster_cols = [c for c in df.columns if 'cluster' in c.lower()]
    print(f"Bestaande cluster kolommen: {cluster_cols}")

    # Hernoem de nieuwe kolommen naar de verwachte namen (voor de zekerheid)
    if "cluster_label_short_new" in df.columns:
        df["cluster_label_short"] = df["cluster_label_short_new"]
        print("✓ cluster_label_short kolom aangemaakt")
    else:
        print("✗ cluster_label_short_new kolom niet gevonden!")

    if "cluster_label_long_new" in df.columns:
        df["cluster_label_long"] = df["cluster_label_long_new"]
        print("✓ cluster_label_long kolom aangemaakt")
    else:
        print("✗ cluster_label_long_new kolom niet gevonden!")

    # Toon voorbeeld waarden
    if "cluster_label_short" in df.columns:
        sample_short = df["cluster_label_short"].dropna().unique()[:5]
        print(f"Voorbeeld korte namen: {list(sample_short)}")

    if "cluster_label_long" in df.columns:
        sample_long = df["cluster_label_long"].dropna().unique()[:2]
        for i, desc in enumerate(sample_long):
            print(f"Voorbeeld lange beschrijving {i+1}: {desc[:80]}...")

    # Sla het gerepareerde bestand op
    output_file = data_dir / "clusters_fixed.csv"
    df.to_csv(output_file, index=False)

    print(f"\n✅ Gerepareerd bestand opgeslagen als: {output_file}")

    # Hernoem naar de verwachte naam
    import os
    os.rename(output_file, input_file)
    print(f"✅ Bestand hernoemd naar: {input_file}")

if __name__ == "__main__":
    main()
