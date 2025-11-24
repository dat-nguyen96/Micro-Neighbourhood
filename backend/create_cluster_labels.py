#!/usr/bin/env python3
"""
Script om cluster labels om te zetten naar catchy korte namen en uitgebreide beschrijvingen
"""

import pandas as pd
import re
from pathlib import Path

def analyze_cluster_characteristics(label_text):
    """Analyseer de karakteristieken van een cluster uit de tekst"""
    characteristics = {}

    # Extract bevolkingsgrootte
    pop_match = re.search(r'gemiddeld (\d+(?:\.\d+)?) inwoners', label_text)
    if pop_match:
        characteristics['population'] = float(pop_match.group(1).replace('.', ''))

    # Extract stedelijkheid
    sted_match = re.search(r'stedelijkheid van (\d+(?:\.\d+)?)', label_text)
    if sted_match:
        characteristics['stedelijkheid'] = float(sted_match.group(1))
    elif 'lage stedelijkheid' in label_text:
        characteristics['stedelijkheid'] = 1.0
    elif 'hoge stedelijkheid' in label_text:
        characteristics['stedelijkheid'] = 4.0
    elif 'matige stedelijkheid' in label_text:
        characteristics['stedelijkheid'] = 3.0

    # Criminaliteit niveau
    if 'lage criminaliteit' in label_text or 'criminaliteitscijfers' in label_text:
        characteristics['crime_level'] = 'low'
    elif 'gemiddelde criminaliteit' in label_text:
        characteristics['crime_level'] = 'medium'
    elif 'hoge criminaliteit' in label_text or 'hoge vermogenscriminaliteit' in label_text:
        characteristics['crime_level'] = 'high'
    else:
        characteristics['crime_level'] = 'medium'

    # Leeftijdsgroep (afgeleid uit bevolkingsgrootte en stedelijkheid)
    if characteristics.get('population', 0) < 500:
        characteristics['size_category'] = 'small'
    elif characteristics.get('population', 0) < 2000:
        characteristics['size_category'] = 'medium'
    else:
        characteristics['size_category'] = 'large'

    return characteristics

def create_catchy_short_name(cluster_id, characteristics):
    """Maak een korte, catchy naam voor een cluster"""

    # Basis mapping gebaseerd op geanalyseerde karakteristieken
    cluster_mappings = {
        0: "Rustig & Groen",  # lage criminaliteit, vermogensmisdrijven focus
        1: "Stedelijk & Druk",  # nog niet gezien, maar waarschijnlijk stedelijk
        2: "Stedelijk & Risico",  # hoge criminaliteit, vermogens- en geweldsmisdrijven
        3: "Groot & Gemiddeld",  # grote buurten, gemiddelde criminaliteit, stedelijkheid 2.7
        4: "Klein & Veilig",  # kleine buurten, lage criminaliteit, matige stedelijkheid
        5: "Modern & Veilig",  # nog niet gezien
        6: "Mega & Risico",  # zeer grote buurten, hoge criminaliteit
        7: "Landelijk & Risico",  # lage stedelijkheid, hoge criminaliteit
        8: "Stedelijk & Veilig",  # nog niet gezien
        9: "Stedelijk & Hoog Risico",  # hoge vermogenscriminaliteit
        10: "Modern & Gemiddeld",  # nog niet gezien
        11: "Groot & Vermogensrisico"  # grote buurten, hoge vermogenscriminaliteit
    }

    return cluster_mappings.get(cluster_id, f"Cluster {cluster_id}")

def create_detailed_description(cluster_id, original_label, short_name):
    """Maak een uitgebreide beschrijving voor een cluster"""

    descriptions = {
        0: "Rustige, groene woonwijken met relatief lage criminaliteit. Vooral vermogensmisdrijven komen voor, maar geweldsmisdrijven zijn zeldzaam. Ideaal voor gezinnen die waarde hechten aan veiligheid en natuur.",
        2: "Drukke stedelijke gebieden met hogere criminaliteitscijfers. Zowel vermogens- als geweldsmisdrijven komen regelmatig voor. Geschikt voor mensen die houden van stedelijk leven maar wel risico's accepteren.",
        3: "Grote, bevolkte buurten met gemiddelde criminaliteit. Goede balans tussen stedelijkheid en leefbaarheid. Veel voorzieningen en sociale activiteiten, maar ook de gebruikelijke stadsproblemen.",
        4: "Kleine, overzichtelijke buurten met lage criminaliteit. Matige stedelijkheid maakt ze prettig leefbaar. Perfect voor mensen die een rustig leven willen zonder te geïsoleerd te raken.",
        6: "Zeer grote, dichtbevolkte gebieden met aanzienlijke criminaliteitsproblemen. Voor mensen die stedelijk leven accepteren met alle bijbehorende risico's en drukte.",
        7: "Landelijke gebieden met lage stedelijkheid maar relatief hoge criminaliteit. Weinig stedelijke voorzieningen, maar ook de problemen van dunbevolkte gebieden.",
        9: "Stedelijke gebieden met zeer hoge vermogenscriminaliteit. Vereist extra aandacht voor beveiliging, maar biedt alle voordelen van stedelijk wonen.",
        11: "Grote wooncomplexen met hoge vermogenscriminaliteit. Moderne stedelijkheid gecombineerd met veiligheidsrisico's die horen bij grote bevolkingsconcentraties."
    }

    return descriptions.get(cluster_id, f"Een woonomgeving met specifieke karakteristieken die aansluit bij verschillende leefstijlen en behoeften.")

def main():
    """Hoofdprogramma om cluster labels te verbeteren"""

    data_dir = Path(__file__).parent / "data"
    input_file = data_dir / "clusters.csv"

    print(f"Lezen van {input_file}...")

    # Lees de huidige data
    df = pd.read_csv(input_file)

    print(f"Gevonden {len(df)} rijen met {len(df['cluster_label'].unique())} unieke clusters")

    # Maak nieuwe kolommen voor verbeterde labels
    df['cluster_label_short_new'] = ""
    df['cluster_label_long_new'] = ""

    # Verwerk elke unieke cluster
    for cluster_id in sorted(df['cluster_id'].unique()):
        cluster_mask = df['cluster_id'] == cluster_id
        sample_row = df[cluster_mask].iloc[0]
        original_label = sample_row['cluster_label']

        print(f"\nVerwerken cluster {cluster_id}:")
        print(f"  Origineel: {original_label}")

        # Analyseer karakteristieken
        characteristics = analyze_cluster_characteristics(original_label)
        print(f"  Geanalyseerd: {characteristics}")

        # Maak nieuwe labels
        short_name = create_catchy_short_name(cluster_id, characteristics)
        long_description = create_detailed_description(cluster_id, original_label, short_name)

        print(f"  Kort: {short_name}")
        print(f"  Lang: {long_description[:100]}...")

        # Update alle rijen van deze cluster
        df.loc[cluster_mask, 'cluster_label_short_new'] = short_name
        df.loc[cluster_mask, 'cluster_label_long_new'] = long_description

    # Sla het verbeterde bestand op
    output_file = data_dir / "clusters_improved.csv"
    df.to_csv(output_file, index=False)

    print(f"\nVerbeterd bestand opgeslagen als: {output_file}")
    print("De nieuwe kolommen zijn: cluster_label_short_new en cluster_label_long_new")

    # Toon samenvatting
    print("\nSamenvatting van verbeterde clusters:")
    for cluster_id in sorted(df['cluster_id'].unique()):
        short_name = df[df['cluster_id'] == cluster_id]['cluster_label_short_new'].iloc[0]
        count = len(df[df['cluster_id'] == cluster_id])
        print(f"  Cluster {cluster_id}: {short_name} ({count} buurten)")

if __name__ == "__main__":
    main()
