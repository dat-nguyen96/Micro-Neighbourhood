#!/usr/bin/env python3
"""
Offline data pipeline - 2 functies:
1. fetch_all_data() - Haalt alle data op en schrijft naar CSV
2. build_clusters() - Gebruikt de data om clusters te bouwen

Aanpassingen:
- Crime: alleen totaal geweld (incl. seksueel geweld) als feature voor clustering.
- Voeg ook rate per 1000 inwoners toe.
"""

import os
from pathlib import Path

import httpx
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load environment
env_path = Path("../.env")
if env_path.exists():
    load_dotenv(env_path)

# Paths
DATA_DIR = Path("../data")
RAW_DATA_CSV = DATA_DIR / "raw_data.csv"
CLUSTERS_CSV = DATA_DIR / "clusters.csv"

# OpenAI client (nu niet gebruikt, maar gelaten voor later)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

# Cluster label mappings - korte namen en uitgebreide beschrijvingen (8 clusters)
CLUSTER_SHORT_NAMES = {
    0: "Jong & Levendig",
    1: "Vergrijzend & Groen",
    2: "Modern & Veilig",
    3: "Stedelijk & Risico",
    4: "Landelijk & Rustig",
    5: "Sociaal & Uitdagend",
    6: "Luxueus & Afgezonderd",
    7: "Traditioneel & Gemengd"
}

CLUSTER_LONG_DESCRIPTIONS = {
    0: "Dynamische wijken met veel jonge bewoners en culturele diversiteit. Levendig straatleven met restaurants en cafés, maar ook hogere criminaliteit door druk sociaal leven.",
    1: "Rustige woonwijken met veel oudere bewoners en groene ruimtes. Veilige straten en goede zorgvoorzieningen, maar soms gevoel van vereenzaming.",
    2: "Moderne buurten met goede voorzieningen en relatief lage geweldscriminaliteit. Geschikt voor mensen die veiligheid en stedelijk comfort willen combineren.",
    3: "Drukke stedelijke centra met hoge bevolkingsdichtheid en relatief meer geweldsincidenten. Veel dynamiek, maar ook meer veiligheidsrisico’s.",
    4: "Landelijke gebieden met veel natuur en ruimte. Over het algemeen weinig geweldsmisdrijven en een rustig straatbeeld.",
    5: "Betaalbare woonwijken met sterke sociale netwerken, maar ook sociale uitdagingen. De ervaren veiligheid kan per straat verschillen.",
    6: "Luxueuze woonwijken met hoge veiligheidsnormen en weinig geweldsmisdrijven, maar soms minder sociale verbondenheid.",
    7: "Traditionele woonwijken met gemengde bevolkingssamenstelling. Gemiddelde niveaus van geweldscriminaliteit en gevarieerd straatbeeld."
}


def fetch_all_data():
    """
    Haal alle benodigde data op van CBS:
    1. CBS 85984NED buurt data (ruwe buurten naar CSV)
    2. CBS buurt namen (optioneel, nu apart CSV)
    3. CBS crime data (47018NED) - alleen geweld (incl. seksueel geweld)
    4. Merge alles samen in RAW_DATA_CSV
    """
    print("=== Start data ophalen ===")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. CBS 85984NED (buurt data) - ruwe data
    print("1. CBS 85984NED (ruwe buurt data) ophalen...")
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
    cbs_raw_csv = DATA_DIR / "cbs_buurten_raw_complete.csv"
    cbs_raw_df.to_csv(cbs_raw_csv, index=False)
    print(f"Ruwe CBS data geschreven naar {cbs_raw_csv} ({len(cbs_raw_df)} rijen)")

    # 2. CBS buurt namen (optioneel)
    print("2. CBS buurt namen ophalen...")
    namen_url = "https://datasets.cbs.nl/odata/v1/CBS/85984NED/WijkenEnBuurtenCodes"
    namen_data = []
    page = 1

    while namen_url:
        print(f"  -> Pagina {page}: {namen_url}")
        resp = requests.get(namen_url)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("value", [])

        buurten = [row for row in batch if str(row.get("Identifier", "")).startswith("BU")]
        namen_data.extend(buurten)
        print(f"    -> {len(batch)} totaal, {len(buurten)} buurten")

        next_link = data.get("@odata.nextLink")
        if not next_link:
            break

        namen_url = next_link
        page += 1

    if namen_data:
        namen_df = pd.DataFrame(namen_data)
        namen_keep_cols = [c for c in namen_df.columns if c in ("Identifier", "Title", "ParentTitle")]
        namen_df = namen_df[namen_keep_cols]
        namen_csv = DATA_DIR / "cbs_buurt_namen_85984.csv"
        namen_df.to_csv(namen_csv, index=False)
        print(f"Buurt namen geschreven naar {namen_csv} ({len(namen_df)} rijen)")

    # 3. Filter alleen buurten
    print("3. Filter alleen buurten...")
    cbs_df = cbs_raw_df[cbs_raw_df["SoortRegio_2"].str.strip() == "Buurt"].copy()
    print(f"Na filtering buurten: {len(cbs_df)} rijen")
    buurten_csv = DATA_DIR / "cbs_buurten_raw_buurten.csv"
    cbs_df.to_csv(buurten_csv, index=False)
    print(f"Gefilterde buurt data geschreven naar {buurten_csv}")

    # 4. CBS 47018NED (crime data) - alleen geweld (incl. seksueel)
    print("4. CBS 47018NED (crime) ophalen...")

    crime_url = "https://dataderden.cbs.nl/ODataApi/OData/47018NED/TypedDataSet"

    # 4a. Beschikbare perioden
    print("  -> Controleren beschikbare perioden...")
    meta_url = "https://dataderden.cbs.nl/ODataApi/OData/47018NED/Perioden"
    resp_meta = requests.get(meta_url)
    resp_meta.raise_for_status()
    periods = [r["Key"] for r in resp_meta.json()["value"]]
    print(f"  -> Beschikbare perioden (eerste 10): {periods[:10]}")

    print("  -> Zoeken naar periode met gedetailleerde crime data...")
    crime_period = None

    import time

    for period in reversed(periods):
        if not period.endswith("JJ00"):
            continue

        print(f"    Testen periode {period}...")
        test_params = {
            "$filter": f"Perioden eq '{period}' and SoortMisdrijf ne '0.0.0 '",
            "$top": 1,
        }
        test_resp = requests.get(crime_url, params=test_params)
        if test_resp.status_code == 200:
            test_data = test_resp.json()
            if test_data.get("value"):
                crime_period = period
                print(f"    ✅ Periode {period} heeft gedetailleerde data!")
                break
            else:
                print(f"    ❌ Periode {period} heeft alleen totals")
        else:
            print(f"    ❌ Periode {period} error: {test_resp.status_code}")
        time.sleep(0.2)

    if not crime_period:
        print("  -> Geen periode met gedetailleerde data gevonden, pak meest recente JJ00")
        crime_period = next((p for p in reversed(periods) if p.endswith("JJ00")), periods[-1])

    print(f"  -> Gebruik periode: {crime_period}")

    # 4b. Haal SoortMisdrijf-dimensie op en selecteer geweld
    print("  -> Ophalen SoortMisdrijf mapping (voor geweld)...")
    crime_data = []

    try:
        misdrijf_url = "https://dataderden.cbs.nl/ODataApi/OData/47018NED/SoortMisdrijf"
        misdrijf_resp = requests.get(misdrijf_url)
        misdrijf_resp.raise_for_status()
        misdrijf_data = misdrijf_resp.json()["value"]

        total_keys = []             # 0.0.0 - totaal misdrijven (optioneel)
        sexual_violence_keys = []   # 1.4.1, 1.4.2
        other_violence_keys = []    # overige 1.4.x

        for item in misdrijf_data:
            raw_key = item["Key"]       # met spatie(s) voor API
            key = raw_key.strip()

            if key == "0.0.0":
                total_keys.append(raw_key)
            elif key in ["1.4.1", "1.4.2"]:
                sexual_violence_keys.append(raw_key)
            elif key.startswith("1.4."):
                other_violence_keys.append(raw_key)

        # Alle geweldscodes (incl. seksueel)
        violence_all_keys = list(set(sexual_violence_keys + other_violence_keys))

        print(f"  -> Totaal-codes: {len(total_keys)}")
        print(f"  -> Seksueel geweld-codes: {len(sexual_violence_keys)}")
        print(f"  -> Overig geweld-codes: {len(other_violence_keys)}")
        print(f"  -> Alle geweld-codes: {len(violence_all_keys)}")

    except Exception as e:
        print(f"  -> FOUT bij ophalen SoortMisdrijf: {e}")
        # Fallback - meest relevante keys met spatie
        total_keys = ["0.0.0 "]
        sexual_violence_keys = ["1.4.1 ", "1.4.2 "]
        other_violence_keys = ["1.4.3 ", "1.4.5 ", "1.4.6 ", "1.4.7 "]
        violence_all_keys = list(set(sexual_violence_keys + other_violence_keys))

    # 4c. Haal crime-data op voor alle geweldscodes
    print("  -> Ophalen crime-data per geweldscode...")
    detailed_categories = list(set(total_keys + violence_all_keys))

    for cat in detailed_categories:
        print(f"    -> Ophalen categorie {cat.strip()}...")
        cat_data = []
        cat_skip = 0

        while True:
            params = {
                "$filter": f"Perioden eq '{crime_period}' and SoortMisdrijf eq '{cat}'",
                "$skip": cat_skip,
                "$top": batch_size,
            }
            resp = requests.get(crime_url, params=params)
            if resp.status_code != 200:
                print(f"      FOUT-status {resp.status_code}, stoppen voor deze categorie")
                break

            data = resp.json()
            batch = data.get("value", [])
            if not batch:
                break

            cat_data.extend(batch)
            if len(batch) < batch_size:
                break

            cat_skip += batch_size

        crime_data.extend(cat_data)
        print(f"      -> {len(cat_data)} rijen voor categorie {cat}")
        time.sleep(0.4)

    print(f"Totaal crime-rijen (geweld + totaal): {len(crime_data)}")

    if crime_data:
        crime_df = pd.DataFrame(crime_data)
        crime_raw_csv = DATA_DIR / f"cbs_crime_geweld_raw_{crime_period}.csv"
        crime_df.to_csv(crime_raw_csv, index=False)
        print(f"Crime (geweld) data geschreven naar {crime_raw_csv} ({len(crime_df)} rijen)")
    else:
        crime_df = pd.DataFrame()
        print("GEEN crime-data gevonden voor geweldscodes")

    # 4d. Aggreren naar totaal geweld per buurt
    print("5. Data combineren met geweldscriminaliteit...")

    if not crime_df.empty:
        # Regiocode kolom zoeken
        regio_col = None
        for col in ["WijkenEnBuurten", "RegioS", "Regiocode"]:
            if col in crime_df.columns:
                regio_col = col
                break

        if not regio_col:
            print("ERROR: Geen regiokolom gevonden in crime data!")
            merged_df = cbs_df.copy()
        else:
            # Kolom met aantallen
            crime_col = None
            for col in ["GeregistreerdeMisdrijven_1", "Misdrijven_1", "TotaalMisdrijven"]:
                if col in crime_df.columns:
                    crime_col = col
                    break

            if not crime_col:
                print("ERROR: Geen misdrijvenkolom gevonden in crime data!")
                merged_df = cbs_df.copy()
            else:
                print(f"Gebruik regio kolom: {regio_col}, crime kolom: {crime_col}")

                def categorize_crime(soort_code: str) -> str:
                    code = str(soort_code).strip()
                    if code == "0.0.0":
                        return "total_crimes"
                    elif code in ["1.4.1", "1.4.2"]:
                        return "sexual_violence"
                    elif code.startswith("1.4."):
                        return "other_violence"
                    else:
                        return "other"

                crime_df["crime_category"] = crime_df["SoortMisdrijf"].apply(categorize_crime)

                # Aggregatie per buurt
                agg = (
                    crime_df.groupby([regio_col, "crime_category"])[crime_col]
                    .sum()
                    .unstack(fill_value=0)
                    .reset_index()
                )

                # Zorg dat alle verwachte kolommen bestaan
                for col in ["total_crimes", "sexual_violence", "other_violence"]:
                    if col not in agg.columns:
                        agg[col] = 0

                agg = agg.rename(columns={regio_col: "WijkenEnBuurten"})
                agg["crime_violence_all"] = agg["sexual_violence"] + agg["other_violence"]

                # Merge met buurtdata
                merged_df = cbs_df.merge(agg, on="WijkenEnBuurten", how="left")

    else:
        merged_df = cbs_df.copy()
        merged_df["total_crimes"] = 0
        merged_df["sexual_violence"] = 0
        merged_df["other_violence"] = 0
        merged_df["crime_violence_all"] = 0

    # NaNs -> 0
    for col in ["total_crimes", "sexual_violence", "other_violence", "crime_violence_all"]:
        if col not in merged_df.columns:
            merged_df[col] = 0
        merged_df[col] = merged_df[col].fillna(0)

    # Rate per 1000 inwoners
    print("6. Bereken geweldsrate per 1000 inwoners...")
    if "AantalInwoners_5" in merged_df.columns:
        merged_df["crime_violence_rate_per_1000"] = (
            merged_df["crime_violence_all"] / merged_df["AantalInwoners_5"].replace({0: pd.NA})
        ) * 1000
        merged_df["crime_violence_rate_per_1000"] = merged_df["crime_violence_rate_per_1000"].fillna(0)
    else:
        print("WAARSCHUWING: AantalInwoners_5 niet gevonden, rate wordt 0")
        merged_df["crime_violence_rate_per_1000"] = 0

    nonzero_geweld = (merged_df["crime_violence_all"] > 0).sum()
    print(f"Aantal buurten met >0 geweldsmisdrijven: {nonzero_geweld}")

    print("Top 5 buurten op geweldsrate (per 1000):")
    cols_show = [
        "WijkenEnBuurten",
        "AantalInwoners_5",
        "crime_violence_all",
        "crime_violence_rate_per_1000",
    ]
    print(
        merged_df[cols_show]
        .sort_values("crime_violence_rate_per_1000", ascending=False)
        .head()
        .to_string(index=False)
    )

    # Schrijf naar hoofdbestand
    merged_df.to_csv(RAW_DATA_CSV, index=False)
    print(f"=== Gedetailleerde data (incl. geweld) geschreven naar {RAW_DATA_CSV} ({len(merged_df)} rijen) ===")


def build_clusters():
    """
    Gebruik de opgehaalde data om KMeans clusters te bouwen.
    Crime-feature: alleen geweldsrate per 1000 inwoners.
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

    # Voeg alleen geweldsrate toe als crime-feature
    crime_feature_used = False
    if "crime_violence_rate_per_1000" in df.columns and df["crime_violence_rate_per_1000"].notna().any():
        feature_cols.append("crime_violence_rate_per_1000")
        crime_feature_used = True
        print("Crime-feature toegevoegd: crime_violence_rate_per_1000")
    else:
        print("GEEN bruikbare crime_violence_rate_per_1000 gevonden - clusters zonder crime-feature")

    # Filter rijen met complete features
    df_features = df[feature_cols].copy()
    mask = df_features.notna().all(axis=1)
    df_clean = df[mask].reset_index(drop=True)
    df_features = df_features[mask].reset_index(drop=True)

    print(f"Na cleaning: {len(df_clean)} rijen")
    print(f"Features gebruikt voor clustering: {feature_cols}")

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(df_features)

    # KMeans
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X)

    # PCA voor visualisatie
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # Labels toewijzen
    cluster_labels = {}
    cluster_labels_long = {}

    for i in range(n_clusters):
        short_name = CLUSTER_SHORT_NAMES.get(i, f"Cluster {i}")
        long_desc = CLUSTER_LONG_DESCRIPTIONS.get(
            i,
            "Een woonomgeving met specifieke kenmerken die aansluit bij verschillende leefstijlen en behoeften.",
        )
        cluster_labels[i] = short_name
        cluster_labels_long[i] = long_desc

    print("Cluster labels toegewezen:")
    for i in range(n_clusters):
        count_i = (clusters == i).sum()
        print(f"  Cluster {i}: {cluster_labels[i]} ({count_i} buurten)")

    # Resultaatdataframe
    result_df = df_clean.copy()
    result_df["cluster_id"] = clusters
    result_df["cluster_label"] = result_df["cluster_id"].map(cluster_labels)
    result_df["cluster_label_long"] = result_df["cluster_id"].map(cluster_labels_long)
    result_df["pca_x"] = X_pca[:, 0]
    result_df["pca_y"] = X_pca[:, 1]

    # Extra check: gemiddelde geweldsrate per cluster (voor interpretatie)
    if crime_feature_used:
        print("Gemiddelde geweldsrate per cluster (per 1000 inwoners):")
        print(
            result_df.groupby("cluster_id")["crime_violence_rate_per_1000"]
            .mean()
            .round(3)
            .to_string()
        )

    # Schrijf naar CSV
    result_df.to_csv(CLUSTERS_CSV, index=False)
    print(
        f"=== Clusters geschreven naar {CLUSTERS_CSV} "
        f"({len(result_df)} rijen, {n_clusters} clusters, incl. cluster_label_long) ==="
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "fetch":
            fetch_all_data()
        elif command == "build":
            build_clusters()
        else:
            print("Gebruik: python scriptnaam.py [fetch|build]")
    else:
        print("Gebruik:")
        print("  python scriptnaam.py fetch    # Haal data op")
        print("  python scriptnaam.py build    # Bouw clusters")
