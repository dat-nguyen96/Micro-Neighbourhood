# backend/offline/process_crime_data.py

import pandas as pd
from pathlib import Path

def prepare_crime_year_for_buurten(
    year: str,
    crime_csv: str = "data/cbs_crime_2024JJ00_buurten.csv",
    outfile: str = None,
):
    """Process crime data for a specific year and aggregate by neighborhood"""

    print(f"Processing crime data for year {year}...")

    # Load the crime data
    df = pd.read_csv(crime_csv)
    print(f"Loaded {len(df)} rows from {crime_csv}")

    # Filter for the specific year
    df_year = df[df['Perioden'].astype(str) == str(year)]
    print(f"Rows for year {year}: {len(df_year)}")

    if df_year.empty:
        print(f"No data found for year {year}")
        return

    # Filter for neighborhoods (BU...)
    df_buurten = df_year[df_year['WijkenEnBuurten'].astype(str).str.startswith('BU')]
    print(f"Neighborhood rows for year {year}: {len(df_buurten)}")

    if df_buurten.empty:
        print(f"No neighborhood data found for year {year}")
        return

    # Group by neighborhood and sum crime counts
    crime_cols = ['GeregistreerdeMisdrijven_1']  # This seems to be the main crime count column

    agg_df = df_buurten.groupby('WijkenEnBuurten')[crime_cols].sum().reset_index()

    # Rename columns for clarity
    agg_df = agg_df.rename(columns={
        'GeregistreerdeMisdrijven_1': 'total_crimes'
    })

    # Save the result
    if outfile is None:
        clean_year = year.replace("JJ00", "")
        outfile = f"data/cbs_crime_{clean_year}_buurten.csv"

    agg_df.to_csv(outfile, index=False)
    print(f"Saved aggregated crime data to {outfile}")
    print(f"Shape: {agg_df.shape}")
    print("Sample:", agg_df.head(3).to_dict('records'))

def merge_crime_with_clusters(year="2024JJ00"):
    """Merge crime data with existing cluster data"""

    # Load existing cluster data
    clusters_df = pd.read_csv("data/buurten_features_clusters.csv")
    print(f"Loaded cluster data: {clusters_df.shape}")

    # Load crime data for the specified year
    crime_file = f"data/cbs_crime_{year}_buurten.csv"
    crime_df = pd.read_csv(crime_file)
    print(f"Loaded crime data: {crime_df.shape}")

    # Merge on neighborhood code
    merged_df = clusters_df.merge(
        crime_df,
        on='WijkenEnBuurten',
        how='left'
    )

    # Save the merged data
    clean_year = year.replace("JJ00", "")
    output_file = f"data/buurten_features_clusters_with_crime_{clean_year}.csv"
    merged_df.to_csv(output_file, index=False)

    print(f"Saved merged data to {output_file}")
    print(f"Final shape: {merged_df.shape}")

    # Show some stats
    crime_coverage = merged_df['total_crimes'].notna().sum()
    print(f"Neighborhoods with crime data: {crime_coverage}/{len(merged_df)}")

if __name__ == "__main__":
    # Process crime data for 2024 (most recent year available)
    prepare_crime_year_for_buurten("2024JJ00")

    # Try to merge (this will fail if the crime file doesn't exist, but shows the structure)
    try:
        merge_crime_with_clusters()
    except FileNotFoundError as e:
        print(f"Crime data file not found: {e}")
        print("Run prepare_crime_year_for_buurten first")
