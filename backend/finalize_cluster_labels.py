#!/usr/bin/env python3
"""
Script om definitief de cluster labels te verbeteren
"""

import pandas as pd
from pathlib import Path

def main():
    """Hoofdprogramma om cluster labels definitief te verbeteren"""

    data_dir = Path(__file__).parent / "data"
    input_file = data_dir / "clusters.csv"

    print(f"Verbeteren van cluster labels in {input_file}...")

    # Lees het originele bestand
    df = pd.read_csv(input_file)

    print(f"Gevonden {len(df)} rijen")

    # Cluster mappings - catchy korte namen en uitgebreide beschrijvingen
    short_names = {
        0: "Rustig & Groen",
        1: "Stedelijk & Druk",
        2: "Stedelijk & Risico",
        3: "Groot & Gemiddeld",
        4: "Klein & Veilig",
        5: "Modern & Veilig",
        6: "Mega & Risico",
        7: "Landelijk & Risico",
        8: "Stedelijk & Veilig",
        9: "Stedelijk & Hoog Risico",
        10: "Modern & Gemiddeld",
        11: "Groot & Vermogensrisico"
    }

    long_descriptions = {
        0: "Rustige, groene woonwijken met relatief lage criminaliteit. Vooral vermogensmisdrijven komen voor, maar geweldsmisdrijven zijn zeldzaam. Ideaal voor gezinnen die waarde hechten aan veiligheid en natuur.",
        1: "Drukke stedelijke gebieden met gemiddelde bevolkingsdichtheid. Geschikt voor mensen die stedelijk leven willen met goede voorzieningen.",
        2: "Drukke stedelijke gebieden met hogere criminaliteitscijfers. Zowel vermogens- als geweldsmisdrijven komen regelmatig voor. Geschikt voor mensen die houden van stedelijk leven maar wel risico's accepteren.",
        3: "Grote, bevolkte buurten met gemiddelde criminaliteit. Goede balans tussen stedelijkheid en leefbaarheid. Veel voorzieningen en sociale activiteiten, maar ook de gebruikelijke stadsproblemen.",
        4: "Kleine, overzichtelijke buurten met lage criminaliteit. Matige stedelijkheid maakt ze prettig leefbaar. Perfect voor mensen die een rustig leven willen zonder te geïsoleerd te raken.",
        5: "Moderne woonwijken met goede voorzieningen en relatief lage criminaliteit. Uitstekend voor mensen die moderne stedelijke voorzieningen waarderen.",
        6: "Zeer grote, dichtbevolkte gebieden met aanzienlijke criminaliteitsproblemen. Voor mensen die stedelijk leven accepteren met alle bijbehorende risico's en drukte.",
        7: "Landelijke gebieden met lage stedelijkheid maar relatief hoge criminaliteit. Weinig stedelijke voorzieningen, maar ook de problemen van dunbevolkte gebieden.",
        8: "Stedelijke gebieden met goede voorzieningen en relatief lage criminaliteit. Uitstekend voor stedelijk leven met veiligheid.",
        9: "Stedelijke gebieden met zeer hoge vermogenscriminaliteit. Vereist extra aandacht voor beveiliging, maar biedt alle voordelen van stedelijk wonen.",
        10: "Moderne gebieden met gemiddelde stedelijke voorzieningen. Goede balans tussen moderniteit en leefbaarheid.",
        11: "Grote wooncomplexen met hoge vermogenscriminaliteit. Moderne stedelijkheid gecombineerd met veiligheidsrisico's die horen bij grote bevolkingsconcentraties."
    }

    # Voeg de nieuwe kolommen toe
    df['cluster_label_short'] = df['cluster_id'].map(short_names).fillna(df['cluster_id'].astype(str).apply(lambda x: f"Cluster {x}"))
    df['cluster_label_long'] = df['cluster_id'].map(long_descriptions).fillna("Een woonomgeving met specifieke karakteristieken die aansluit bij verschillende leefstijlen en behoeften.")

    # Toon samenvatting
    print("\nNieuwe cluster labels:")
    for cluster_id in sorted(df['cluster_id'].unique()):
        short_name = df[df['cluster_id'] == cluster_id]['cluster_label_short'].iloc[0]
        count = len(df[df['cluster_id'] == cluster_id])
        print(f"  Cluster {cluster_id}: {short_name} ({count} buurten)")

    # Sla het verbeterde bestand op
    output_file = data_dir / "clusters_final.csv"
    df.to_csv(output_file, index=False)

    print(f"\n✅ Verbeterd bestand opgeslagen als: {output_file}")

    # Hernoem naar de verwachte naam voor de backend
    import os
    final_name = data_dir / "clusters_improved.csv"
    if final_name.exists():
        os.remove(final_name)
    os.rename(output_file, final_name)

    print(f"✅ Bestand hernoemd naar: {final_name}")

if __name__ == "__main__":
    main()
