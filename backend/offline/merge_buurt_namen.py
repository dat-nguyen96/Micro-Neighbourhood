#!/usr/bin/env python3
"""
Merge buurt namen met de bestaande buurt data
"""
import pandas as pd
from pathlib import Path


def merge_buurt_namen():
    """Voeg buurt namen toe aan de bestaande data"""

    # Lees bestaande data
    buurten_file = Path("../data/buurten_features_clusters_with_crime_2024.csv")
    namen_file = Path("../data/cbs_buurt_namen_83765.csv")

    print(f"Lees buurten data: {buurten_file}")
    buurten_df = pd.read_csv(buurten_file)

    print(f"Lees buurt namen: {namen_file}")
    namen_df = pd.read_csv(namen_file)

    print(f"Buurten data shape: {buurten_df.shape}")
    print(f"Namen data shape: {namen_df.shape}")

    # Check if buurt_naam column already exists
    if "buurt_naam" in buurten_df.columns:
        print("buurt_naam kolom bestaat al, update bestaande waarden...")
        # Merge and update existing buurt_naam column
        merged_df = buurten_df.merge(
            namen_df,
            left_on="WijkenEnBuurten",
            right_on="Identifier",
            how="left"
        )
        # Update buurt_naam where it's empty or NaN
        mask = merged_df["buurt_naam"].isna() | (merged_df["buurt_naam"] == "")
        merged_df.loc[mask, "buurt_naam"] = merged_df.loc[mask, "Title"]
        merged_df["buurt_naam"] = merged_df["buurt_naam"].fillna("Onbekende buurt")

        # Remove the merged columns
        merged_df = merged_df.drop(columns=["Identifier", "Title"])
    else:
        print("buurt_naam kolom bestaat niet, voeg toe...")
        # Merge op WijkenEnBuurten = Identifier
        merged_df = buurten_df.merge(
            namen_df,
            left_on="WijkenEnBuurten",
            right_on="Identifier",
            how="left"
        )

        # Hernoem kolom en vul missende namen
        merged_df = merged_df.rename(columns={"Title": "buurt_naam"})
        merged_df["buurt_naam"] = merged_df["buurt_naam"].fillna("Onbekende buurt")

        # Verwijder Identifier kolom (duplicate)
        merged_df = merged_df.drop(columns=["Identifier"])

    # Sorteer op buurt_naam voor netheid (maar het is mogelijk niet uniek)
    try:
        merged_df = merged_df.sort_values("buurt_naam")
    except ValueError:
        print("Kon niet sorteren op buurt_naam (mogelijk niet uniek), ga verder zonder sortering")
        pass

    # Schrijf terug naar dezelfde file
    output_file = Path("../data/buurten_features_clusters_with_crime_2024.csv")
    merged_df.to_csv(output_file, index=False)

    print(f"✅ Merged data geschreven naar: {output_file}")
    print(f"Final shape: {merged_df.shape}")

    # Toon enkele voorbeelden
    print("\nVoorbeelden:")
    sample = merged_df[["WijkenEnBuurten", "buurt_naam", "Gemeentenaam_1"]].head(5)
    print(sample.to_string(index=False))

    # Check Rotterdam voorbeeld
    rotterdam = merged_df[merged_df["WijkenEnBuurten"] == "BU05990110"]
    if not rotterdam.empty:
        print(f"\nRotterdam voorbeeld: {rotterdam.iloc[0]['WijkenEnBuurten']} -> {rotterdam.iloc[0]['buurt_naam']}")


if __name__ == "__main__":
    merge_buurt_namen()
