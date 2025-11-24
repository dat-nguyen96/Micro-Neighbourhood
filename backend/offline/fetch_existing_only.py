#!/usr/bin/env python3
"""
Offline data pipeline - 2 functies:
1. fetch_all_data() - Haalt alle data op en schrijft naar CSV
2. build_clusters() - Gebruikt de data om clusters te bouwen
"""

import pandas as pd
import httpx
import requests
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment
env_path = Path("../.env")
if env_path.exists():
    load_dotenv(env_path)

# Paths
DATA_DIR = Path("../data")
RAW_DATA_CSV = DATA_DIR / "raw_data.csv"
CLUSTERS_CSV = DATA_DIR / "clusters.csv"

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

def fetch_all_data():
    """
    Haal alle benodigde data op van CBS:
    1. CBS 85984NED buurt data (ruwe buurten naar CSV)
    2. CBS buurt namen (Identifier + Title naar CSV)
    3. CBS crime data (47018NED)
    4. Merge alles samen
    """
    print("=== Start data ophalen ===")

    # Zorg dat data directory bestaat
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. CBS 85984NED (buurt data) - schrijf eerst ruwe data naar CSV
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

    # Schrijf ruwe CBS data naar CSV
    cbs_raw_df = pd.DataFrame(all_cbs_data)
    cbs_raw_csv = DATA_DIR / "cbs_buurten_raw_complete.csv"
    cbs_raw_df.to_csv(cbs_raw_csv, index=False)
    print(f"Ruwe CBS data geschreven naar {cbs_raw_csv} ({len(cbs_raw_df)} rijen)")

    # 2. CBS buurt namen ophalen (Identifier + Title)
    print("2. CBS buurt namen ophalen...")
    namen_url = "https://datasets.cbs.nl/odata/v1/CBS/85984NED/WijkenEnBuurtenCodes"
    namen_data = []
    skip = 0

    while namen_url:
        print(f"  -> Pagina {skip//1000 + 1}: {namen_url}")
        resp = requests.get(namen_url)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("value", [])

        # Filter alleen buurten (BU codes)
        buurten = [row for row in batch if str(row.get("Identifier", "")).startswith("BU")]
        namen_data.extend(buurten)
        print(f"    -> {len(batch)} totaal, {len(buurten)} buurten")

        if len(batch) < 1000:  # Laatste pagina
            break

        namen_url = data.get("@odata.nextLink")
        skip += 1000

    # Schrijf buurt namen naar CSV
    namen_df = pd.DataFrame(namen_data)
    if not namen_df.empty:
        namen_keep_cols = [c for c in namen_df.columns if c in ("Identifier", "Title", "ParentTitle")]
        namen_df = namen_df[namen_keep_cols]
        namen_csv = DATA_DIR / "cbs_buurt_namen_85984.csv"
        namen_df.to_csv(namen_csv, index=False)
        print(f"Buurt namen geschreven naar {namen_csv} ({len(namen_df)} rijen)")

    # 3. Filter alleen buurten uit ruwe CBS data
    print("3. Filter alleen buurten...")
    cbs_df = cbs_raw_df[cbs_raw_df["SoortRegio_2"].str.strip() == "Buurt"].copy()
    print(f"Na filtering buurten: {len(cbs_df)} rijen")

    # Schrijf gefilterde buurt data naar CSV
    buurten_csv = DATA_DIR / "cbs_buurten_raw_buurten.csv"
    cbs_df.to_csv(buurten_csv, index=False)
    print(f"Gefilterde buurt data geschreven naar {buurten_csv}")

    # 4. CBS 47018NED (crime data)
    print("4. CBS 47018NED (crime) ophalen...")

    crime_url = "https://dataderden.cbs.nl/ODataApi/OData/47018NED/TypedDataSet"

    # Check beschikbare perioden eerst
    print("  -> Controleren beschikbare perioden...")
    meta_url = "https://dataderden.cbs.nl/ODataApi/OData/47018NED/Perioden"
    resp_meta = requests.get(meta_url)
    resp_meta.raise_for_status()
    periods = [r["Key"] for r in resp_meta.json()["value"]]
    print(f"  -> Beschikbare perioden: {periods[:10]}...")

    # Zoek periode met gedetailleerde crime data (niet alleen totals)
    print("  -> Zoeken naar periode met gedetailleerde crime data...")
    crime_period = None

    # Test elke periode van nieuw naar oud
    for period in reversed(periods):
        if not period.endswith("JJ00"):
            continue

        print(f"    Testen periode {period}...")
        test_params = {"$filter": f"Perioden eq '{period}' and SoortMisdrijf ne '0.0.0 '", "$top": 1}
        test_resp = requests.get(crime_url, params=test_params)

        if test_resp.status_code == 200:
            test_data = test_resp.json()
            if test_data.get("value"):  # Heeft gedetailleerde data
                crime_period = period
                print(f"    ✅ Periode {period} heeft gedetailleerde data!")
                break
            else:
                print(f"    ❌ Periode {period} heeft alleen totals")
        else:
            print(f"    ❌ Periode {period} error: {test_resp.status_code}")

        # Kleine pauze tussen requests
        import time
        time.sleep(0.2)

    if not crime_period:
        print("  -> Geen periode gevonden met gedetailleerde data, gebruik meest recente voor totals")
        crime_period = next((p for p in reversed(periods) if p.endswith("JJ00")), periods[-1])

    print(f"  -> Gebruik periode: {crime_period}")

    crime_data = []
    skip = 0

    # CBS ODataApi limiet: we moeten slim filteren om alle categorieën te krijgen
    # In plaats van alles ophalen, filteren we op specifieke categorieën
    print("  -> Ophalen gedetailleerde crime categorieën...")

    # Haal eerst de mapping van misdrijf soorten op
    try:
        misdrijf_url = "https://dataderden.cbs.nl/ODataApi/OData/47018NED/SoortMisdrijf"
        misdrijf_resp = requests.get(misdrijf_url)
        misdrijf_resp.raise_for_status()
        misdrijf_data = misdrijf_resp.json()["value"]

        # Selecteer categorieën volgens gedetailleerde indeling
        # MisdrijvenTotaal: 0.0.0 (totaal)
        # SeksueelGeweld: 1.4.1, 1.4.2 (zedenmisdrijven)
        # Geweldsmisdrijven: overige 1.4.x (bedreiging, mishandeling, moord, straatroof, overvallen, openlijk geweld)
        # Vermogensmisdrijven: 1.1.x, 1.2.x (diefstal, inbraak)
        # Vernieling/OpenbareOrde: 2.2.1, 3.6.4, 3.7.x (vernieling, discriminatie, etc.)

        total_keys = []        # 0.0.0 - totaal misdrijven
        sexual_violence_keys = []  # 1.4.1, 1.4.2 - seksueel geweld/zeden
        violence_keys = []    # Overige 1.4.x - geweldmisdrijven
        property_keys = []    # 1.1.x, 1.2.x - vermogensmisdrijven
        vandalism_keys = []   # 2.2.1, 3.6.4, 3.7.x - vernieling, openbare orde

        for item in misdrijf_data:
            raw_key = item["Key"]  # Bewaar spaties voor API filter
            key = raw_key.strip()  # Voor prefix checks

            if key == "0.0.0":
                total_keys.append(raw_key)
            elif key in ['1.4.1', '1.4.2']:
                sexual_violence_keys.append(raw_key)
            elif key.startswith('1.4.'):
                violence_keys.append(raw_key)
            elif key.startswith(('1.1.', '1.2.')):
                property_keys.append(raw_key)
            elif key in ['2.2.1', '3.6.4'] or key.startswith('3.7.'):
                vandalism_keys.append(raw_key)

        # Combineer alle categorieën
        detailed_categories = total_keys + sexual_violence_keys + violence_keys + property_keys + vandalism_keys
        detailed_categories = list(set(detailed_categories))  # Unieke categorieën

        print(f"  -> {len(detailed_categories)} categorieën geselecteerd")
        print(f"  -> Totaal: {len(total_keys)}, Seksueel geweld: {len(sexual_violence_keys)}, Geweld: {len(violence_keys)}, Vermogens: {len(property_keys)}, Vernieling: {len(vandalism_keys)}")
        print(f"  -> Voorbeelden: {detailed_categories[:5]}")

    except Exception as e:
        print(f"  -> Kon misdrijf mapping niet ophalen: {e}")
        # Fallback naar handmatig geselecteerde belangrijke categorieën (met spaties voor API)
        detailed_categories = [
            "0.0.0 ",  # Totaal misdrijven
            "1.4.1 ", "1.4.2 ",  # Seksueel geweld
            "1.4.3 ", "1.4.5 ", "1.4.6 ", "1.4.7 ",  # Geweldmisdrijven
            "1.1.1 ", "1.1.2 ", "1.2.1 ", "1.2.2 ",  # Vermogensmisdrijven
            "2.2.1 ", "3.6.4 ", "3.7.1 "  # Vernieling/openbare orde
        ]

    # Haal data op per categorie om API limiet te omzeilen
    print(f"  -> Ophalen data voor {len(detailed_categories)} categorieën...")
    for cat in detailed_categories:
        print(f"  -> Ophalen categorie {cat.strip()}...")
        cat_data = []
        cat_skip = 0

        while True:  # Haal alle data voor deze categorie
            params = {
                "$filter": f"Perioden eq '{crime_period}' and SoortMisdrijf eq '{cat}'",
                "$skip": cat_skip,
                "$top": batch_size
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

        # Pauze om API niet te overbelasten
        import time
        time.sleep(0.5)

    print(f"Totaal gedetailleerde crime data: {len(crime_data)} rijen")

    # Gebruik alle opgehaalde data
    if crime_data:
        crime_df = pd.DataFrame(crime_data)
        print(f"Crime categorieën opgehaald: {crime_df['SoortMisdrijf'].unique()}")
    else:
        crime_df = pd.DataFrame()

    # 5. Crime data naar CSV
    if crime_data:
        crime_df = pd.DataFrame(crime_data)
        crime_raw_csv = DATA_DIR / f"cbs_crime_raw_{crime_period}.csv"
        crime_df.to_csv(crime_raw_csv, index=False)
        print(f"Crime data geschreven naar {crime_raw_csv} ({len(crime_df)} rijen)")

    # 6. Merge buurt data met gedetailleerde crime data
    print("6. Data combineren met gedetailleerde criminaliteit...")

    if crime_data:
        print(f"Crime columns: {crime_df.columns.tolist()}")

        # Check welke kolom de regiocode bevat
        regio_col = None
        for col in ["WijkenEnBuurten", "RegioS", "Regiocode"]:
            if col in crime_df.columns:
                regio_col = col
                break

        if not regio_col:
            print("ERROR: Geen regiokolom gevonden in crime data!")
            return

        # Check welke kolom de misdrijvencount bevat
        crime_col = None
        for col in ["GeregistreerdeMisdrijven_1", "TotaalMisdrijven", "MisdrijvenTotaal"]:
            if col in crime_df.columns:
                crime_col = col
                break

        if not crime_col:
            print("ERROR: Geen misdrijfenkolom gevonden in crime data!")
            return

        print(f"Gebruik regio kolom: {regio_col}, crime kolom: {crime_col}")

        # Categoriseer misdrijven volgens CBS mapping
        def categorize_crime(soort_code):
            """Categoriseer misdrijf volgens gedetailleerde indeling"""
            code = str(soort_code).strip()
            if code == "0.0.0":
                return 'total_crimes'
            elif code in ['1.4.1', '1.4.2']:
                return 'sexual_violence'
            elif code.startswith('1.4'):  # Overige 1.4.x (geweldmisdrijven)
                return 'violence'
            elif code.startswith('1.1') or code.startswith('1.2'):  # Vermogensmisdrijven
                return 'property'
            elif code in ['2.2.1'] or code in ['3.6.4'] or code.startswith('3.7'):  # Vernieling, openbare orde
                return 'vandalism'
            else:
                return 'other'

        # Voeg categorie toe aan crime data
        crime_df['crime_category'] = crime_df['SoortMisdrijf'].apply(categorize_crime)

        # Aggregeer per buurt en categorie
        crime_agg = crime_df.groupby([regio_col, 'crime_category'])[crime_col].sum().unstack(fill_value=0).reset_index()
        crime_agg = crime_agg.rename(columns={
            regio_col: "WijkenEnBuurten",
            'total_crimes': 'total_crimes',  # Totaal misdrijven
            'sexual_violence': 'crime_sexual_violence',  # Seksueel geweld
            'violence': 'crime_violence',  # Geweldmisdrijven
            'property': 'crime_property',  # Vermogensmisdrijven
            'vandalism': 'crime_vandalism',  # Vernieling/openbare orde
            'other': 'crime_other'
        })

        # Voeg totaal toe als fallback (som van alle categorieën)
        crime_cols = [col for col in crime_agg.columns if col.startswith('crime_') and col != 'total_crimes']
        if 'total_crimes' not in crime_agg.columns:
            crime_agg['total_crimes'] = crime_agg[crime_cols].sum(axis=1)

        print(f"Gedetailleerde crime aggregatie:")
        print(f"  - Totaal misdrijven: {crime_agg['total_crimes'].sum()}")
        print(f"  - Seksueel geweld: {crime_agg.get('crime_sexual_violence', pd.Series([0])).sum()}")
        print(f"  - Geweldsmisdrijven: {crime_agg.get('crime_violence', pd.Series([0])).sum()}")
        print(f"  - Vermogensmisdrijven: {crime_agg.get('crime_property', pd.Series([0])).sum()}")
        print(f"  - Vernieling/openbare orde: {crime_agg.get('crime_vandalism', pd.Series([0])).sum()}")

        # Merge met buurt data
        merged_df = cbs_df.merge(crime_agg, on="WijkenEnBuurten", how="left")
    else:
        print("Geen crime data gevonden, voeg lege kolommen toe")
        merged_df = cbs_df.copy()
        merged_df["total_crimes"] = 0
        merged_df["crime_sexual_violence"] = 0
        merged_df["crime_violence"] = 0
        merged_df["crime_property"] = 0
        merged_df["crime_vandalism"] = 0
        merged_df["crime_other"] = 0

    # Fill NaN met 0
    crime_columns = ["total_crimes", "crime_sexual_violence", "crime_violence", "crime_property", "crime_vandalism", "crime_other"]
    for col in crime_columns:
        merged_df[col] = merged_df[col].fillna(0)

    # Controleer eindresultaat
    nonzero_crimes = (merged_df["total_crimes"] > 0).sum()
    print(f"Aantal rijen met niet-0 crimes: {nonzero_crimes}")
    if nonzero_crimes > 0:
        print("Top 5 hoogste total crime counts:")
        top5 = merged_df[["WijkenEnBuurten", "total_crimes", "crime_sexual_violence", "crime_violence", "crime_property", "crime_vandalism"]].sort_values("total_crimes", ascending=False).head()
        print(top5.to_string(index=False))

    # Schrijf naar CSV
    merged_df.to_csv(RAW_DATA_CSV, index=False)
    print(f"=== Gedetailleerde data geschreven naar {RAW_DATA_CSV} ({len(merged_df)} rijen) ===")

def build_clusters():
    """
    Gebruik de opgehaalde data om KMeans clusters te bouwen
    """
    print("=== Start cluster bouw ===")

    # Lees data
    df = pd.read_csv(RAW_DATA_CSV)
    print(f"Data geladen: {len(df)} rijen")

    # Feature kolommen inclusief gedetailleerde criminaliteit
    feature_cols = [
        "AantalInwoners_5", "Bevolkingsdichtheid_34", "GemiddeldeHuishoudensgrootte_33",
        "HuishoudensMetKinderen_32", "HuishoudensTotaal_29", "MateVanStedelijkheid_122",
        "k_0Tot15Jaar_8", "k_15Tot25Jaar_9", "k_25Tot45Jaar_10", "k_45Tot65Jaar_11", "k_65JaarOfOuder_12"
    ]

    # Voeg gedetailleerde crime data toe
    crime_features = []
    if "crime_sexual_violence" in df.columns and df["crime_sexual_violence"].notna().any():
        feature_cols.append("crime_sexual_violence")
        crime_features.append("crime_sexual_violence")
    if "crime_violence" in df.columns and df["crime_violence"].notna().any():
        feature_cols.append("crime_violence")
        crime_features.append("crime_violence")
    if "crime_property" in df.columns and df["crime_property"].notna().any():
        feature_cols.append("crime_property")
        crime_features.append("crime_property")
    if "crime_vandalism" in df.columns and df["crime_vandalism"].notna().any():
        feature_cols.append("crime_vandalism")
        crime_features.append("crime_vandalism")

    if crime_features:
        print(f"Gedetailleerde crime features toegevoegd: {crime_features}")
    else:
        # Fallback naar total_crimes als gedetailleerde data niet beschikbaar is
        if "total_crimes" in df.columns and df["total_crimes"].notna().any():
            feature_cols.append("total_crimes")
            print("Total crime data toegevoegd aan features")
        else:
            print("Geen crime data beschikbaar - clusters zonder crime features")

    # Filter rijen met alle features
    df_features = df[feature_cols].copy()
    mask = df_features.notna().all(axis=1)
    df_clean = df[mask].reset_index(drop=True)
    df_features = df_features[mask].reset_index(drop=True)

    print(f"Na cleaning: {len(df_clean)} rijen")
    print(f"Features gebruikt: {feature_cols}")

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(df_features)

    # KMeans
    n_clusters = 12
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X)

    # PCA voor visualisatie
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # Labels genereren
    cluster_labels = {}
    for i in range(n_clusters):
        cluster_data = df_features[clusters == i]
        if client:
            # Genereer korte beschrijving met LLM
            summary = f"Cluster {i}: {len(cluster_data)} buurten, "
            summary += f"gemiddelde inwoners: {cluster_data['AantalInwoners_5'].mean():.0f}, "
            summary += f"stedelijkheid: {cluster_data['MateVanStedelijkheid_122'].mean():.1f}"

            # Voeg gedetailleerde criminaliteit info toe aan prompt
            crime_info = []
            if "crime_sexual_violence" in cluster_data.columns and cluster_data['crime_sexual_violence'].mean() > 0:
                crime_info.append(f"seksueel geweld: {cluster_data['crime_sexual_violence'].mean():.1f}")
            if "crime_violence" in cluster_data.columns and cluster_data['crime_violence'].mean() > 0:
                crime_info.append(f"geweldsmisdrijven: {cluster_data['crime_violence'].mean():.1f}")
            if "crime_property" in cluster_data.columns and cluster_data['crime_property'].mean() > 0:
                crime_info.append(f"vermogensmisdrijven: {cluster_data['crime_property'].mean():.1f}")
            if "crime_vandalism" in cluster_data.columns and cluster_data['crime_vandalism'].mean() > 0:
                crime_info.append(f"vernieling/openbare orde: {cluster_data['crime_vandalism'].mean():.1f}")

            if crime_info:
                summary += f", criminaliteit ({', '.join(crime_info)})"
            elif "total_crimes" in cluster_data.columns and cluster_data['total_crimes'].mean() > 0:
                summary += f", criminaliteit: {cluster_data['total_crimes'].mean():.1f}"

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Geef een korte Nederlandse beschrijving (max 20 woorden) van dit buurt cluster."},
                        {"role": "user", "content": summary}
                    ],
                    max_tokens=50
                )
                label = response.choices[0].message.content.strip()
            except:
                label = f"Cluster {i}"
        else:
            label = f"Cluster {i}"

        cluster_labels[i] = label

    # Resultaat dataframe
    result_df = df_clean.copy()
    result_df["cluster_id"] = clusters
    result_df["cluster_label"] = result_df["cluster_id"].map(cluster_labels)
    result_df["pca_x"] = X_pca[:, 0]
    result_df["pca_y"] = X_pca[:, 1]

    # Schrijf naar CSV
    result_df.to_csv(CLUSTERS_CSV, index=False)
    print(f"=== Clusters geschreven naar {CLUSTERS_CSV} ({len(result_df)} rijen, {n_clusters} clusters) ===")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "fetch":
            fetch_all_data()
        elif command == "build":
            build_clusters()
        else:
            print("Gebruik: python fetch_existing_only.py [fetch|build]")
    else:
        print("Gebruik:")
        print("  python fetch_existing_only.py fetch    # Haal data op")
        print("  python fetch_existing_only.py build    # Bouw clusters")
