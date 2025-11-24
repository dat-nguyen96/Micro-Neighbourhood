# backend/offline/fetch_crime_and_merge.py

import requests
import pandas as pd
from pathlib import Path
from typing import Union, Optional


def fetch_crime_raw_to_csv(
    table_id: str = "47018NED",
    outfile: Union[Path, str] = "data/cbs_crime_raw.csv",
    year_filter: str = "2024JJ00",  # Filter op specifiek jaar
    max_rows: Optional[int] = None,   # optioneel: harde limiet
):
    """
    Haal alle rijen van de Politie/CBS-tabel (default: 47018NED) voor een specifiek jaar
    via ODataApi met $filter, $top en $skip paginatie.

    Dit is robuuster dan zelf nextLink volgen.
    """
    base_url = f"https://dataderden.cbs.nl/ODataApi/OData/{table_id}/TypedDataSet"
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching CBS {table_id} for year {year_filter}...")
    print(f"Base URL: {base_url}")

    all_rows: list[dict] = []
    skip = 0
    batch_size = 1000  # Blijf onder de 10k limiet
    batch_count = 0

    while True:
        batch_count += 1

        # Bouw URL met filter, top en skip
        filter_param = f"Perioden eq '{year_filter}'"
        url = f"{base_url}?$filter={filter_param}&$top={batch_size}&$skip={skip}"
        print(f"Requesting batch {batch_count}: $skip={skip}, $top={batch_size}")

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

        # Optionele harde limiet op aantal rijen
        if max_rows is not None and len(all_rows) >= max_rows:
            print(f"max_rows={max_rows} bereikt; stoppen met ophalen.")
            all_rows = all_rows[:max_rows]
            break

        # Controleer of we minder rijen kregen dan gevraagd - laatste batch
        if batch_len < batch_size:
            print(f"Laatste batch: {batch_len} rijen (minder dan {batch_size}).")
            break

        skip += batch_size

    if not all_rows:
        print("Geen data opgehaald; CSV wordt niet geschreven.")
        return

    print(f"Totaal opgehaald: {len(all_rows)} rijen. Schrijf naar CSV...")

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
    Lees de ruwe CBS crime data, filter op gegeven jaar en BU-wijken,
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
    clusters_csv: Union[Path, str],
    crime_buurten_csv: Union[Path, str],
    outfile: Union[Path, str, None] = None,
) -> Path:
    """
    Merge de clusters+features CSV met de geaggregeerde crime data per buurt.
    """
    clusters_csv = Path(clusters_csv)
    crime_buurten_csv = Path(crime_buurten_csv)
    if outfile is None:
        outfile = clusters_csv.parent / f"{clusters_csv.stem}_with_crime_{crime_buurten_csv.stem.split('_')[-1]}.csv"
    outfile = Path(outfile)

    print(f"Lees clusters uit {clusters_csv}...")
    clusters = pd.read_csv(clusters_csv)

    print(f"Lees crime data uit {crime_buurten_csv}...")
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
    print("=== STARTING CRIME DATA FETCH (2024 only) ===")
    year_filter = "2024JJ00"
    fetch_crime_raw_to_csv(
        table_id="47018NED",
        outfile="../data/cbs_crime_raw.csv",
        year_filter=year_filter,
        max_rows=None,  # Haal alle beschikbare 2024 data op
    )
    print("=== CRIME DATA FETCH COMPLETE ===")

    # 2) Data per buurt aggregeren (alleen 2024 data is al gefilterd)
    year = "2024JJ00"
    crime_year_csv = prepare_crime_year_for_buurten(
        year=year,
        raw_csv="data/cbs_crime_raw.csv",
        outfile=f"../data/cbs_crime_{year}_buurten.csv",
    )

    # 3) Merge met jouw bestaande features + clusters
    merge_clusters_with_crime(
        clusters_csv="../data/buurten_features_clusters.csv",
        crime_buurten_csv=crime_year_csv,
        outfile=f"../data/buurten_features_clusters_with_crime_{year}.csv",
    )
