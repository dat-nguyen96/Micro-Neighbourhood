#!/usr/bin/env python3
"""
Offline data pipeline - 2 functies:
1. fetch_all_data() - Haalt alle data op en schrijft naar CSV
2. build_clusters() - Gebruikt de data om clusters te bouwen

Belangrijk:
- We gebruiken alleen TOTAAL GEWELD (incl. seksueel geweld) als feature voor clustering.
- Seksueel geweld + ander geweld = crime_violence_total
- Gedetailleerde misdrijfcodes worden wel bewaard voor tooltips/visualisatie.
"""

import os
from pathlib import Path

import httpx
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# === Environment ===
env_path = Path("../.env")
if env_path.exists():
    load_dotenv(env_path)

# === Paths ===
DATA_DIR = Path("../data")
RAW_DATA_CSV = DATA_DIR / "raw_data.csv"
CLUSTERS_CSV = DATA_DIR / "clusters.csv"

# === Cluster label mappings (8 clusters) ===
CLUSTER_SHORT_NAMES = {
    0: "Jong & Levendig",
    1: "Vergrijzend & Groen",
    2: "Modern & Veilig",
    3: "Stedelijk & Risico",
    4: "Landelijk & Rustig",
    5: "Sociaal & Uitdagend",
    6: "Luxueus & Afgezonderd",
    7: "Traditioneel & Gemengd",
}

CLUSTER_LONG_DESCRIPTIONS = {
    0: "Dynamische wijken met veel jonge bewoners en culturele diversiteit. Levendig straatleven met restaurants en cafés, maar ook iets hogere criminaliteit door een druk sociaal leven.",
    1: "Rustige woonwijken met veel oudere bewoners en groene ruimtes. Veilige straten en goede zorgvoorzieningen, maar soms een gevoel van vereenzaming.",
    2: "Moderne appartementen en wooncomplexen met goede beveiliging en stedelijke voorzieningen. Schone, goed onderhouden buurten met een relatief veilig straatbeeld.",
    3: "Drukke stedelijke centra met hoge bevolkingsdichtheid en meer veiligheidsrisico's. Veel voorzieningen en openbaar vervoer, maar ook meer sociale en veiligheidsspanning.",
    4: "Landelijke gebieden met veel natuur en ruimte. Rustige woonomgevingen met relatief weinig criminaliteit, maar soms beperkte voorzieningen en langere reistijden.",
    5: "Betaalbare woonwijken met sterke sociale cohesie maar ook meer sociaal-maatschappelijke uitdagingen. Gezellige buurten waar veiligheid niet altijd vanzelfsprekend is.",
    6: "Luxueuze woonwijken met hoge veiligheidsnormen en veel privacy. Moderne woningen, vaak met hogere woonlasten en soms minder sociale betrokkenheid in de buurt.",
    7: "Traditionele woonwijken met een mix van bewoners en voorzieningen. Sterke gemeenschapsbanden, maar een gemengd beeld qua leefbaarheid en veiligheid.",
}


# ======================================================================
# 1. DATA FETCHEN
# ======================================================================
def fetch_all_data():
    """
    Haal alle benodigde data op van CBS:
    1. CBS 85984NED buurt data (ruwe buurten naar CSV)
    2. CBS buurt namen (Identifier + Title naar CSV)
    3. CBS crime data (47018NED) - inclusief geweld (met seksueel geweld)
    4. Merge alles samen in raw_data.csv
    """
    print("=== Start data ophalen ===")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. CBS 85984NED - alle regio's
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 2. CBS buurt namen (alleen BU-codes)
    # --------------------------------------------------
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

        buurten = [
            row for row in batch if str(row.get("Identifier", "")).startswith("BU")
        ]
        namen_data.extend(buurten)
        print(f"    -> {len(batch)} totaal, {len(buurten)} buurten")

        next_link = data.get("@odata.nextLink")
        if not next_link:
            break
        namen_url = next_link
        page += 1

    namen_df = pd.DataFrame(namen_data)
    if not namen_df.empty:
        keep_cols = ["Identifier", "Title", "ParentTitle"]
        keep_cols = [c for c in keep_cols if c in namen_df.columns]
        namen_df = namen_df[keep_cols]
        namen_csv = DATA_DIR / "cbs_buurt_namen_85984.csv"
        namen_df.to_csv(namen_csv, index=False)
        print(f"Buurt namen geschreven naar {namen_csv} ({len(namen_df)} rijen)")

    # --------------------------------------------------
    # 3. Filter alleen buurten in de hoofdtabel
    # --------------------------------------------------
    print("3. Filter alleen buurten...")
    cbs_df = cbs_raw_df[cbs_raw_df["SoortRegio_2"].str.strip() == "Buurt"].copy()
    print(f"Na filtering buurten: {len(cbs_df)} rijen")
    buurten_csv = DATA_DIR / "cbs_buurten_raw_buurten.csv"
    cbs_df.to_csv(buurten_csv, index=False)
    print(f"Gefilterde buurt data geschreven naar {buurten_csv}")

    # --------------------------------------------------
    # 4. Crime data 47018NED
    # --------------------------------------------------
    print("4. CBS 47018NED (crime) ophalen...")

    crime_url = "https://dataderden.cbs.nl/ODataApi/OData/47018NED/TypedDataSet"

    # 4a. Bepaal periode met gedetailleerde SoortMisdrijf-data
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
            "$filter": "Perioden eq '{}' and SoortMisdrijf ne '0.0.0 '".format(period),
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
        print("  -> Geen periode met gedetailleerde data, neem meest recente JJ00")
        crime_period = next(
            (p for p in reversed(periods) if p.endswith("JJ00")), periods[-1]
        )

    print(f"  -> Gebruik periode: {crime_period}")

    # 4b. Haal mapping SoortMisdrijf op en bepaal categorieën
    crime_data = []
    print("  -> Ophalen mapping SoortMisdrijf...")

    try:
        misdrijf_url = "https://dataderden.cbs.nl/ODataApi/OData/47018NED/SoortMisdrijf"
        misdrijf_resp = requests.get(misdrijf_url)
        misdrijf_resp.raise_for_status()
        misdrijf_data = misdrijf_resp.json()["value"]

        total_keys = []            # 0.0.0 (totaal misdrijven)
        sexual_violence_keys = []  # 1.4.1, 1.4.2 (zeden/seksueel geweld)
        other_violence_keys = []   # overige 1.4.x
        property_keys = []         # 1.1.x, 1.2.x
        vandalism_keys = []        # 2.2.1, 3.6.4, 3.7.x (vernieling / openbare orde)

        for item in misdrijf_data:
            raw_key = item["Key"]  # bevat trailing spatie
            key = raw_key.strip()

            if key == "0.0.0":
                total_keys.append(raw_key)
            elif key in ["1.4.1", "1.4.2"]:
                sexual_violence_keys.append(raw_key)
            elif key.startswith("1.4."):
                other_violence_keys.append(raw_key)
            elif key.startswith("1.1.") or key.startswith("1.2."):
                property_keys.append(raw_key)
            elif key in ["2.2.1", "3.6.4"] or key.startswith("3.7."):
                vandalism_keys.append(raw_key)

        detailed_categories = (
            total_keys
            + sexual_violence_keys
            + other_violence_keys
            + property_keys
            + vandalism_keys
        )
        detailed_categories = list(set(detailed_categories))

        print(
            "  -> {} categorieën geselecteerd "
            "(totaal={}, seksueel geweld={}, geweld={}, vermogen={}, vernieling/openb.orde={})".format(
                len(detailed_categories),
                len(total_keys),
                len(sexual_violence_keys),
                len(other_violence_keys),
                len(property_keys),
                len(vandalism_keys),
            )
        )

    except Exception as e:
        print(f"  -> Kon misdrijf mapping niet ophalen: {e}")
        detailed_categories = [
            "0.0.0 ",
            "1.4.1 ",
            "1.4.2 ",
            "1.4.3 ",
            "1.4.5 ",
            "1.4.6 ",
            "1.4.7 ",
            "1.1.1 ",
            "1.1.2 ",
            "1.2.1 ",
            "1.2.2 ",
            "2.2.1 ",
            "3.6.4 ",
            "3.7.1 ",
        ]

    # 4c. Haal crime data per SoortMisdrijf categorie op
    print(f"  -> Ophalen data voor {len(detailed_categories)} categorieën...")

    for cat in detailed_categories:
        print(f"  -> Ophalen categorie {cat.strip()}...")
        cat_data = []
        cat_skip = 0

        while True:
            params = {
                "$filter": "Perioden eq '{}' and SoortMisdrijf eq '{}'".format(
                    crime_period, cat
                ),
                "$skip": cat_skip,
                "$top": batch_size,
            }
            resp = requests.get(crime_url, params=params)
            if resp.status_code != 200:
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
        print(f"    -> {len(cat_data)} rijen voor categorie {cat}")
        time.sleep(0.3)

    print(f"Totaal gedetailleerde crime data: {len(crime_data)} rijen")

    if crime_data:
        crime_df = pd.DataFrame(crime_data)
        crime_raw_csv = DATA_DIR / f"cbs_crime_raw_{crime_period}.csv"
        crime_df.to_csv(crime_raw_csv, index=False)
        print(f"Crime data geschreven naar {crime_raw_csv} ({len(crime_df)} rijen)")
    else:
        crime_df = pd.DataFrame()
        print("⚠️ Geen crime data opgehaald")

    # --------------------------------------------------
    # 5. Crime verwerken: geweld (incl. seksueel) + details
    # --------------------------------------------------
    print("5. Crime data verwerken en aggregeren...")

    if not crime_df.empty:
        print(f"Crime columns: {crime_df.columns.tolist()}")

        # Bepaal regiocode kolom
        regio_col = None
        for col in ["WijkenEnBuurten", "RegioS", "Regiocode"]:
            if col in crime_df.columns:
                regio_col = col
                break

        if not regio_col:
            print("ERROR: Geen regiokolom gevonden in crime data!")
            return

        # Bepaal misdrijvencolom
        crime_col = None
        for col in [
            "GeregistreerdeMisdrijven_1",
            "MisdrijvenTotaal_1",
            "MisdrijvenTotaal",
        ]:
            if col in crime_df.columns:
                crime_col = col
                break

        if not crime_col:
            print("ERROR: Geen misdrijvencolom gevonden in crime data!")
            return

        print(f"Gebruik regio kolom: {regio_col}, crime kolom: {crime_col}")

        def categorize_crime(soort_code: str) -> str:
            """Categoriseer per gedetailleerde code."""
            code = str(soort_code).strip()
            if code == "0.0.0":
                return "total_crimes"
            elif code in ["1.4.1", "1.4.2"]:
                return "sexual_violence"
            elif code.startswith("1.4"):
                return "other_violence"
            elif code.startswith("1.1") or code.startswith("1.2"):
                return "property"
            elif code in ["2.2.1"] or code in ["3.6.4"] or code.startswith("3.7"):
                return "vandalism"
            else:
                return "other"

        crime_df["crime_category"] = crime_df["SoortMisdrijf"].apply(categorize_crime)

        # Pivot met alle individuele SoortMisdrijf-waarden (voor tooltips)
        detailed_crime_pivot = (
            crime_df.pivot_table(
                index=regio_col,
                columns="SoortMisdrijf",
                values=crime_col,
                fill_value=0,
            )
            .reset_index()
        )

        rename_dict = {regio_col: "WijkenEnBuurten"}
        for col in detailed_crime_pivot.columns:
            if col != regio_col and str(col).strip():
                clean_code = str(col).strip().replace(".", "_")
                rename_dict[col] = f"crime_{clean_code}"

        detailed_crime_pivot = detailed_crime_pivot.rename(columns=rename_dict)

        # Aggregatie per buurt en categorie
        crime_agg = (
            crime_df.groupby([regio_col, "crime_category"])[crime_col]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )

        crime_agg = crime_agg.rename(columns={regio_col: "WijkenEnBuurten"})

        # Zorg dat ontbrekende categorieën bestaan
        for needed in [
            "total_crimes",
            "sexual_violence",
            "other_violence",
            "property",
            "vandalism",
        ]:
            if needed not in crime_agg.columns:
                crime_agg[needed] = 0

        # TOTAAL GEWELD = seksueel geweld + ander geweld
        crime_agg["crime_violence_total"] = (
            crime_agg["sexual_violence"] + crime_agg["other_violence"]
        )

        # Totaal misdrijven fallback (als 0.0.0 ontbreekt)
        if crime_agg["total_crimes"].sum() == 0:
            crime_agg["total_crimes"] = (
                crime_agg["crime_violence_total"]
                + crime_agg["property"]
                + crime_agg["vandalism"]
            )

        # Kolomnamen meer consistent voor clustering
        crime_agg = crime_agg.rename(
            columns={
                "total_crimes": "total_crimes",
                "sexual_violence": "crime_sexual_violence",
                "other_violence": "crime_other_violence",
                "property": "crime_property",
                "vandalism": "crime_vandalism",
            }
        )

        # Merge met alle gedetailleerde crime-codes
        crime_agg = crime_agg.merge(detailed_crime_pivot, on="WijkenEnBuurten", how="left")

        print("Samenvatting crime aggregatie:")
        print(f"  - Totaal misdrijven: {crime_agg['total_crimes'].sum()}")
        print(
            f"  - Totaal geweld (incl. seksueel): {crime_agg['crime_violence_total'].sum()}"
        )
        print(
            f"  - Seksueel geweld: {crime_agg.get('crime_sexual_violence', pd.Series([0])).sum()}"
        )

        # Merge met buurtdata
        merged_df = cbs_df.merge(crime_agg, on="WijkenEnBuurten", how="left")
    else:
        print("Geen crime data gevonden, voeg lege kolommen toe")
        merged_df = cbs_df.copy()
        merged_df["total_crimes"] = 0
        merged_df["crime_violence_total"] = 0
        merged_df["crime_sexual_violence"] = 0
        merged_df["crime_other_violence"] = 0
        merged_df["crime_property"] = 0
        merged_df["crime_vandalism"] = 0

    # Na merge: NaN -> 0 voor crime kolommen
    crime_columns = [
        "total_crimes",
        "crime_violence_total",
        "crime_sexual_violence",
        "crime_other_violence",
        "crime_property",
        "crime_vandalism",
    ]
    for col in crime_columns:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(0)

    nonzero_crimes = (merged_df["total_crimes"] > 0).sum()
    print(f"Aantal buurten met niet-0 crimes: {nonzero_crimes}")

    # Schrijf gecombineerde data naar raw_data.csv
    merged_df.to_csv(RAW_DATA_CSV, index=False)
    print(f"=== Gedetailleerde data geschreven naar {RAW_DATA_CSV} ({len(merged_df)} rijen) ===")


# ======================================================================
# 2. CLUSTERS BOUWEN
# ======================================================================
def build_clusters():
    """
    Gebruik de opgehaalde data om KMeans clusters te bouwen.
    We gebruiken:
    - Demografische / stedelijkheid features
    - Alleen TOTAAL GEWELD als crime feature: crime_violence_total
    """
    print("=== Start cluster bouw ===")

    df = pd.read_csv(RAW_DATA_CSV)
    print(f"Data geladen: {len(df)} rijen")

    # Basis features (CBS 85984NED)
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

    # ENKEL geweld als crime feature
    crime_feature = None
    if "crime_violence_total" in df.columns and df["crime_violence_total"].notna().any():
        feature_cols.append("crime_violence_total")
        crime_feature = "crime_violence_total"
        print("Violence feature toegevoegd aan clustering: crime_violence_total")
    else:
        print("⚠️ Geen crime_violence_total gevonden, clustering zonder crime features")

    # Filter alleen rijen waar alle features aanwezig zijn
    df_features = df[feature_cols].copy()
    mask = df_features.notna().all(axis=1)
    df_clean = df[mask].reset_index(drop=True)
    df_features = df_features[mask].reset_index(drop=True)

    print(f"Na cleaning: {len(df_clean)} rijen over")
    print(f"Features gebruikt voor clustering: {feature_cols}")

    # Standardiseren
    scaler = StandardScaler()
    X = scaler.fit_transform(df_features)

    # KMeans
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X)

    # PCA voor 2D visualisatie
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # Labels (kort + lang) koppelen
    cluster_labels = {}
    cluster_labels_long = {}

    for i in range(n_clusters):
        short_name = CLUSTER_SHORT_NAMES.get(i, f"Cluster {i}")
        long_desc = CLUSTER_LONG_DESCRIPTIONS.get(
            i,
            "Een woonomgeving met specifieke karakteristieken die aansluit bij verschillende leefstijlen en behoeften.",
        )
        cluster_labels[i] = short_name
        cluster_labels_long[i] = long_desc

    print("Cluster labels toegewezen:")
    for i in range(n_clusters):
        cluster_count = (clusters == i).sum()
        print(f"  Cluster {i}: {cluster_labels[i]} ({cluster_count} buurten)")

    # Resultaat dataframe
    result_df = df_clean.copy()
    result_df["cluster_id"] = clusters
    result_df["cluster_label"] = result_df["cluster_id"].map(cluster_labels)
    result_df["cluster_label_long"] = result_df["cluster_id"].map(cluster_labels_long)
    result_df["pca_x"] = X_pca[:, 0]
    result_df["pca_y"] = X_pca[:, 1]

    # Schrijf clusters naar CSV (inclusief cluster_label_long!)
    result_df.to_csv(CLUSTERS_CSV, index=False)
    print(
        f"=== Clusters geschreven naar {CLUSTERS_CSV} "
        f"({len(result_df)} rijen, {n_clusters} clusters) ==="
    )


# ======================================================================
# CLI
# ======================================================================
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
