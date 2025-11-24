# backend/offline/fetch_crime_and_merge.py

import requests
import pandas as pd
from pathlib import Path
from typing import Union


def fetch_crime_raw_to_csv(
    table_id: str = "47018NED",
    outfile: Union[Path, str] = "data/cbs_crime_raw.csv",
    batch_size: int = 1000,
):
    """
    Haal alle rijen van de Politie/CBS-tabel (default: 47018NED)
    batch-gewijs op en schrijf naar een ruwe CSV.
    """
    base_url = f"https://dataderden.cbs.nl/ODataApi/OData/{table_id}/TypedDataSet"
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching CBS {table_id} to CSV with pagination...")
    print(f"Base URL: {base_url}")

    all_rows = []
    skip = 0
    total_fetched = 0

    while True:
        url = f"{base_url}?$top={batch_size}&$skip={skip}"
        print(f"Requesting batch: $skip={skip}, $top={batch_size}")

        resp = requests.get(url)
        if not resp.ok:
            print(f"[ERROR] HTTP {resp.status_code} bij ophalen batch. Stop.")
            break

        data = resp.json()
        rows = data.get("value", [])
        batch_len = len(rows)
        print(f"  -> ontvangen rijen: {batch_len}")

        if batch_len == 0:
            print("Geen rijen meer, paginatie klaar.")
            break

        all_rows.extend(rows)
        total_fetched += batch_len
        skip += batch_len  # veilig: sommige tabellen geven minder dan batch_size terug

    if not all_rows:
        print("Geen data opgehaald; CSV wordt niet geschreven.")
        return

    print(f"Totaal opgehaald: {total_fetched} rijen. Schrijf naar CSV...")

    df = pd.DataFrame(all_rows)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outfile, index=False)
    print(f"Gereed. Geschreven naar: {outfile.resolve()}")


def prepare_crime_year_for_buurten(
    year: str,
    raw_csv: Union[Path, str] = "data/cbs_crime_raw.csv",
    outfile: Union[Path, str, None] = None,
) -> Path:
    """
    Neem de ruwe 47018NED CSV, filter op jaar + buurten,
    aggregeer naar 1 rij per buurt en schrijf naar CSV.

    Let op: omdat we de kolomnamen van 47018NED hier niet hard-coden,
    worden alle numerieke kolommen (aantallen, evt. ratio's) gesommeerd.
    Wil je alleen specifieke kolommen (bijv. totaal aantal misdrijven),
    dan kun je 'numeric_cols' zelf verder beperken.
    """
    raw_csv = Path(raw_csv)
    if outfile is None:
        outfile = raw_csv.parent / f"cbs_crime_{year}_buurten.csv"
    outfile = Path(outfile)

    print(f"Lees ruwe crime-data uit {raw_csv}...")
    df = pd.read_csv(raw_csv)

    # alleen gekozen jaar
    if "Perioden" not in df.columns:
        raise RuntimeError("Kolom 'Perioden' niet gevonden in crime CSV.")
    df = df[df["Perioden"].astype(str) == str(year)]

    # alleen buurten (BU...)
    if "WijkenEnBuurten" not in df.columns:
        raise RuntimeError("Kolom 'WijkenEnBuurten' niet gevonden in crime CSV.")
    df = df[df["WijkenEnBuurten"].astype(str).str.startswith("BU")].copy()

    if df.empty:
        print(f"Waarschuwing: geen rijen voor jaar {year} na filtering.")
        df.to_csv(outfile, index=False)
        print(f"Lege CSV geschreven naar: {outfile.resolve()}")
        return outfile

    # Bepaal welke kolommen we aggregeren
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Sleutelkolommen niet sommeren
    key_cols = ["WijkenEnBuurten"]
    numeric_cols = [c for c in numeric_cols if c not in key_cols]

    print(f"Aggregeren {len(numeric_cols)} numerieke kolommen per buurt...")

    # Aggregeren per buurt
    agg_df = (
        df.groupby("WijkenEnBuurten")[numeric_cols]
        .sum(min_count=1)
        .reset_index()
    )

    agg_df.to_csv(outfile, index=False)
    print(f"Gereed. Aggregerende crime-data geschreven naar: {outfile.resolve()}")
    return outfile


def merge_clusters_with_crime(
    clusters_csv: Union[Path, str] = "backend/data/buurten_features_clusters.csv",
    crime_buurten_csv: Union[Path, str] = "data/cbs_crime_2024_buurten.csv",
    outfile: Union[Path, str, None] = None,
) -> Path:
    """
    Merge de bestaande buurt-features/cluster-CSV met de geaggregeerde
    criminaliteitsdata per buurt.

    Resultaat: zelfde aantal rijen als clusters_csv,
    met extra kolommen uit de crime-CSV (left join).
    """
    clusters_csv = Path(clusters_csv)
    crime_buurten_csv = Path(crime_buurten_csv)

    if outfile is None:
        outfile = clusters_csv.with_name(
            clusters_csv.stem + "_with_crime" + clusters_csv.suffix
        )
    outfile = Path(outfile)

    print(f"Lees clusters uit: {clusters_csv}")
    clusters = pd.read_csv(clusters_csv)

    print(f"Lees crime-data per buurt uit: {crime_buurten_csv}")
    crime = pd.read_csv(crime_buurten_csv)

    if "WijkenEnBuurten" not in clusters.columns:
        raise RuntimeError("Kolom 'WijkenEnBuurten' ontbreekt in clusters CSV.")
    if "WijkenEnBuurten" not in crime.columns:
        raise RuntimeError("Kolom 'WijkenEnBuurten' ontbreekt in crime CSV.")

    merged = clusters.merge(
        crime,
        on="WijkenEnBuurten",
        how="left",
        suffixes=("", "_crime"),
    )

    merged.to_csv(outfile, index=False)
    print(f"Gereed. Merge geschreven naar: {outfile.resolve()}")
    return outfile


if __name__ == "__main__":
    # 1) Ruwe 47018NED ophalen (eenmalig, of af en toe verversen)
    # Start with smaller batch size for testing
    fetch_crime_raw_to_csv(
        table_id="47018NED",
        outfile="data/cbs_crime_raw.csv",
        batch_size=1000,
    )

    # 2) Specifiek jaar klaarzetten – 2024 is het laatste beschikbare jaar
    year = "2024JJ00"  # CBS format voor jaar 2024
    crime_year_csv = prepare_crime_year_for_buurten(
        year=year,
        raw_csv="data/cbs_crime_raw.csv",
        outfile=f"data/cbs_crime_{year}_buurten.csv",
    )

    # 3) Merge met jouw bestaande features + clusters
    merge_clusters_with_crime(
        clusters_csv="backend/data/buurten_features_clusters.csv",
        crime_buurten_csv=crime_year_csv,
        outfile=f"backend/data/buurten_features_clusters_with_crime_{year}.csv",
    )
