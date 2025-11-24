#!/usr/bin/env python3
"""
Offline data pipeline - 2 functies:
1. fetch_all_data() - Haalt alle data op en schrijft naar CSV
2. build_clusters() - Bouwt KMeans clusters op basis van bevolkingsdichtheid en geweld
   en laat AI achteraf de clusters betekenisvolle labels geven.

Belangrijk:
- We gebruiken alleen TOTAAL GEWELD (incl. seksueel geweld) als crime feature.
- Seksueel geweld + ander geweld = crime_violence_total
- Voor clustering gebruiken we:
    - Bevolkingsdichtheid_34
    - crime_violence_total per 1.000 inwoners
- Gedetailleerde misdrijfcodes worden wel bewaard voor tooltips/visualisatie.
"""

import os
import json
from pathlib import Path

import httpx
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import openai

# OpenAI
import openai
from openai import OpenAI

# === Environment ===
env_path = Path("../.env")
if env_path.exists():
    load_dotenv(env_path)

openai.api_key = os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# === Paths ===
DATA_DIR = Path("../data")
RAW_DATA_CSV = DATA_DIR / "raw_data.csv"
CLUSTERS_CSV = DATA_DIR / "clusters.csv"

# ======================================================================
# AI HELPER: Cluster labels genereren
# ======================================================================

def generate_ai_cluster_labels(df_clean, clusters, n_clusters):
    """
    Genereer korte en lange labels per cluster met behulp van OpenAI.
    We gaan uit van 2 features:
    - Bevolkingsdichtheid_34
    - violence_per_1000

    Retourneert:
        cluster_labels: dict {cluster_id: korte_naam}
        cluster_labels_long: dict {cluster_id: lange_beschrijving}
    """
    if openai.api_key is None:
        print("⚠️ Geen OPENAI_API_KEY gevonden, gebruik fallback cluster labels.")
        raise RuntimeError("No OPENAI_API_KEY")

    has_stedelijkheid = "MateVanStedelijkheid_122" in df_clean.columns

    # Check voor leeftijdskolommen
    age_columns = ["k_0Tot15Jaar_8", "k_15Tot25Jaar_9", "k_25Tot45Jaar_10", "k_45Tot65Jaar_11", "k_65JaarOfOuder_12"]
    available_age_cols = [col for col in age_columns if col in df_clean.columns]

    # Check voor nieuwe demografische features
    demo_columns = [
        "aandeel_jongeren_0_25",
        "aandeel_65_plus",
        "GemiddeldeHuishoudensgrootte_33",
        "HuishoudensMetKinderen_32"
    ]
    available_demo_cols = [col for col in demo_columns if col in df_clean.columns]

    # Maak een compacte samenvatting per cluster
    rows = []
    for cid in range(n_clusters):
        mask = clusters == cid
        sub = df_clean.loc[mask]

        if sub.empty:
            avg_density = avg_violence = avg_sted = 0.0
            age_avgs = {col: 0.0 for col in available_age_cols}
            count = 0
        else:
            avg_density = sub["Bevolkingsdichtheid_34"].mean()
            avg_violence = sub["violence_per_1000"].mean()
            avg_sted = sub["MateVanStedelijkheid_122"].mean() if has_stedelijkheid else None
            count = int(mask.sum())

            # Bereken gemiddelde leeftijd percentages per cluster
            age_avgs = {}
            for age_col in available_age_cols:
                age_avgs[age_col] = round(float(sub[age_col].mean()), 1)

            # Bereken gemiddelde demografische indicators per cluster
            demo_avgs = {}
            for demo_col in available_demo_cols:
                demo_avgs[demo_col] = round(float(sub[demo_col].mean()), 2)

        row_data = {
            "cluster_id": cid,
            "avg_density": round(float(avg_density), 1),
            "avg_violence_per_1000": round(float(avg_violence), 3),
            "avg_stedelijkheid": round(float(avg_sted), 2) if has_stedelijkheid and avg_sted is not None else None,
            "count": count,
        }

        # Voeg leeftijd data toe
        for age_col in available_age_cols:
            row_data[f"avg_{age_col}"] = age_avgs[age_col]

        # Voeg demografische data toe
        for demo_col in available_demo_cols:
            row_data[f"avg_{demo_col}"] = demo_avgs[demo_col]

        rows.append(row_data)

    # Numerieke samenvatting in CSV-vorm
    header_cols = ["cluster_id", "avg_density", "avg_violence_per_1000", "count"]
    if has_stedelijkheid:
        header_cols.insert(3, "avg_stedelijkheid")

    # Voeg leeftijdskolommen toe aan header
    for age_col in available_age_cols:
        header_cols.append(f"avg_{age_col}")

    # Voeg demografische kolommen toe aan header
    for demo_col in available_demo_cols:
        header_cols.append(f"avg_{demo_col}")

    lines = [",".join(header_cols)]
    for r in rows:
        cols = [
            str(r["cluster_id"]),
            str(r["avg_density"]),
            str(r["avg_violence_per_1000"]),
        ]
        if has_stedelijkheid:
            cols.append(str(r["avg_stedelijkheid"]))
        cols.append(str(r["count"]))

        # Voeg leeftijd data toe
        for age_col in available_age_cols:
            cols.append(str(r[f"avg_{age_col}"]))

        # Voeg demografische data toe
        for demo_col in available_demo_cols:
            cols.append(str(r[f"avg_{demo_col}"]))

        lines.append(",".join(cols))

    csv_summary = "\n".join(lines)

    # Uitleg stedelijkheid-schaal voor GPT (CBS: 1 = zeer sterk stedelijk, 5 = niet stedelijk)
    stedelijkheid_info = (
        "De variabele 'MateVanStedelijkheid_122' is een schaal van 1 t/m 5:\n"
        "1 = zeer sterk stedelijk, 2 = sterk stedelijk, 3 = matig stedelijk,\n"
        "4 = weinig stedelijk, 5 = niet stedelijk (landelijk).\n"
    )

    # Uitleg leeftijdskolommen voor GPT
    leeftijd_info = (
        "Leeftijdskolommen zijn percentages van de bevolking:\n"
        "k_0Tot15Jaar_8 = percentage kinderen (0-15 jaar)\n"
        "k_15Tot25Jaar_9 = percentage jongeren (15-25 jaar)\n"
        "k_25Tot45Jaar_10 = percentage jongvolwassenen (25-45 jaar)\n"
        "k_45Tot65Jaar_11 = percentage middenleeftijd (45-65 jaar)\n"
        "k_65JaarOfOuder_12 = percentage ouderen (65+ jaar)\n"
        "\n"
        "Demografische indicators:\n"
        "aandeel_jongeren_0_25 = percentage jongeren (0-25 jaar)\n"
        "aandeel_65_plus = percentage senioren (65+ jaar)\n"
        "GemiddeldeHuishoudensgrootte_33 = gemiddelde aantal personen per huishouden\n"
        "HuishoudensMetKinderen_32 = percentage huishoudens met kinderen\n"
        "\n"
        "Deze indicators helpen bij het identificeren van:\n"
        "- Gezinsbuurten (hoog kindpercentage, grote huishoudens)\n"
        "- Studentenbuurten (hoog jongerenpercentage, kleine huishoudens)\n"
        "- Actieve beroepsbevolking (hoog 25-45 jaar)\n"
        "- Seniorenbuurten (hoog 65+ jaar)\n"
        "- Singles appartementen (kleine huishoudens, weinig kinderen)\n"
        "- Gezinnenwijken (veel huishoudens met kinderen)\n"
    )

    system_msg = (
        "Je bent een Nederlandse data-analist gespecialiseerd in woonbuurten. "
        "Je krijgt clusters van Nederlandse woonbuurten met bevolkingsdichtheid, geweldsniveau, stedelijkheid, "
        "leeftijdssamenstelling en demografische indicators (huishoudgrootte, gezinsstructuur). "
        "Je taak is om korte, herkenbare Nederlandse namen te geven die aansluiten bij bekende woonconcepten. "
        "Gebruik alle beschikbare data om specifieke doelgroepen te identificeren: "
        "gezinsbuurten (veel kinderen, grote huishoudens), studentenwijken (veel jongeren, kleine huishoudens), "
        "seniorencomplexen (veel 65+, weinig kinderen), Vinex-wijken (gezinnen, buiten stad), etc. "
        "Gebruik termen als: druk centrum, rustige Vinex-wijk, stedelijke appartementen, landelijke dorpen, uitgaansgebied, villawijk, hoogbouw, gezinswijk, "
        "studentenbuurt, etc. "
        "Elke korte naam moet uniek zijn per cluster."
    )

    user_msg = f"""

Hier is de uitleg van de stedelijkheidsschaal:

{stedelijkheid_info}

Hier is de uitleg van de leeftijdskolommen:

{leeftijd_info}

Hier is de numerieke samenvatting van de clusters:

{csv_summary}

Geef voor iedere cluster_id precies één regel met Nederlandse woonconcepten:

cluster_id | korte_label (max 4 woorden, uniek) | lange_beschrijving

Voorbeelden van gewenste korte labels (gebruik demografie voor perfecte typeringen):
- Druk centrum (hoge density, jongeren)
- Rustige Vinex-wijk (gezinnen, buiten stad)
- Stedelijke appartementen (kleine huishoudens, singles)
- Landelijke dorpen (senioren, rustige omgeving)
- Voorstedelijke eengezinswoningen (gezinnen, matige density)
- Uitgaansgebied (jongeren, stedelijk)
- Villawijk (hoge inkomens, rust)
- Hoogbouw complex (stedelijk, kleine huishoudens)
- Studentenwijk (veel 15-25 jaar, kleine huishoudens)
- Seniorenpark (veel 65+, weinig kinderen)
- Gezinsbuurt (veel kinderen, grote huishoudens)
- Actieve wijk (beroepsgroep 25-45 jaar)

Gebruik GEEN extra tekst, geen uitleg, geen markdown, alleen de regels in dit formaat:

0 | Druk centrum | ...

1 | Studentenwijk | ...

enzovoort.

"""

    try:
        # Gebruik moderne OpenAI API (zoals in main.py)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=800,
        )

        content = response.choices[0].message.content.strip()
        # Parse de regels: "id | short | long"
        cluster_labels = {}
        cluster_labels_long = {}

        for line in content.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|", 2)]
            if len(parts) != 3:
                continue
            cid_str, short_label, long_label = parts
            try:
                cid = int(cid_str)
            except ValueError:
                continue
            cluster_labels[cid] = short_label
            cluster_labels_long[cid] = long_label

        # Check of we voor alle clusters iets hebben
        missing = [cid for cid in range(n_clusters) if cid not in cluster_labels]
        if missing:
            raise ValueError(f"Ontbrekende labels voor clusters: {missing}")

        # Extra check: korte labels moeten uniek zijn
        if len(set(cluster_labels.values())) != len(cluster_labels):
            raise ValueError("Korte labels zijn niet uniek, AI-output ongeldig.")

        return cluster_labels, cluster_labels_long

    except Exception as e:
        print(f"⚠️ Fout bij genereren AI labels: {e}")
        raise

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
        crime_agg = crime_agg.merge(
            detailed_crime_pivot, on="WijkenEnBuurten", how="left"
        )

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
    print(
        f"=== Gedetailleerde data geschreven naar {RAW_DATA_CSV} ({len(merged_df)} rijen) ==="
    )




# ======================================================================
# 3. CLUSTERS BOUWEN
# ======================================================================
def build_clusters():
    """
    Gebruik de opgehaalde data om KMeans clusters te bouwen.

    Clustering-features:
    - Bevolkingsdichtheid_34
    - Geweld (incl. seksueel geweld) per 1.000 inwoners (crime_violence_total / AantalInwoners_5 * 1000)

    Labels:
    - Met AI gegenereerde korte en lange cluster-namen (als OPENAI_API_KEY aanwezig is)
    - Anders: simpele fallback labels
    """
    print("=== Start cluster bouw ===")

    df = pd.read_csv(RAW_DATA_CSV)
    print(f"Data geladen: {len(df)} rijen")

    # Check benodigde kolommen
    required_cols = ["Bevolkingsdichtheid_34", "crime_violence_total", "AantalInwoners_5"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Ontbrekende kolommen in RAW_DATA_CSV: {missing}")

    # Geweld per 1.000 inwoners
    df["violence_per_1000"] = 0.0
    mask_pop = df["AantalInwoners_5"] > 0
    df.loc[mask_pop, "violence_per_1000"] = (
        df.loc[mask_pop, "crime_violence_total"] / df.loc[mask_pop, "AantalInwoners_5"] * 1000.0
    )

    # Geweld per 1.000 inwoners
    df["violence_per_1000"] = 0.0
    mask_pop = df["AantalInwoners_5"] > 0
    df.loc[mask_pop, "violence_per_1000"] = (
        df.loc[mask_pop, "crime_violence_total"] / df.loc[mask_pop, "AantalInwoners_5"] * 1000.0
    )

    # Extra leeftijd features voor betere clustering
    df["aandeel_jongeren_0_25"] = df.get("k_0Tot15Jaar_8", 0) + df.get("k_15Tot25Jaar_9", 0)
    df["aandeel_65_plus"] = df.get("k_65JaarOfOuder_12", 0)

    # Features voor clustering - uitgebreid met demografische indicators
    feature_cols = [
        "Bevolkingsdichtheid_34",      # Bevolkingsdichtheid
        "violence_per_1000",          # Geweld per 1000 inwoners
    ]

    # Voeg optionele demografische features toe als ze bestaan
    optional_features = [
        "aandeel_jongeren_0_25",      # Jongeren aandeel (0-25 jaar)
        "aandeel_65_plus",            # Senioren aandeel (65+ jaar)
        "GemiddeldeHuishoudensgrootte_33",  # Gemiddelde huishoudgrootte
        "HuishoudensMetKinderen_32",  # % huishoudens met kinderen
    ]

    for feature in optional_features:
        if feature in df.columns:
            feature_cols.append(feature)
        else:
            print(f"⚠️ Feature '{feature}' niet gevonden in dataset, wordt overgeslagen")
    df_features = df[feature_cols].copy()

    # Filter rijen met complete data
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

    # PCA voor 2D visualisatie (alleen voor plotting, niet voor labels)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # === AI-clusterlabels genereren ===
    try:
        ai_short, ai_long = generate_ai_cluster_labels(df_clean, clusters, n_clusters)
        print("AI cluster labels succesvol gegenereerd.")
    except Exception as e:
        print(f"AI label generatie mislukt: {e}")
        print("Gebruik fallback cluster labels.")
        ai_short = {i: f"Cluster {i}" for i in range(n_clusters)}
        ai_long = {
            i: "Een woonomgeving met eigen kenmerken op het gebied van bevolkingsdichtheid, stedelijkheid en geweldscriminaliteit."
            for i in range(n_clusters)
        }

    print("Cluster labels toegewezen:")
    for i in range(n_clusters):
        cluster_count = int((clusters == i).sum())
        print(f"  Cluster {i}: {ai_short[i]} ({cluster_count} buurten)")

    # Resultaat dataframe - alle originele kolommen bewaren
    result_df_full = df.loc[mask].reset_index(drop=True)
    result_df_full["cluster_id"] = clusters
    result_df_full["cluster_label"] = result_df_full["cluster_id"].map(ai_short)
    result_df_full["cluster_label_long"] = result_df_full["cluster_id"].map(ai_long)
    result_df_full["pca_x"] = X_pca[:, 0]
    result_df_full["pca_y"] = X_pca[:, 1]

    # Schrijf clusters naar CSV
    result_df_full.to_csv(CLUSTERS_CSV, index=False)
    print(
        f"=== Clusters geschreven naar {CLUSTERS_CSV} "
        f"({len(result_df_full)} rijen, {n_clusters} clusters) ==="
    )

    # Kleine preview
    print("Voorbeeld clusters:")
    print(
        result_df_full[["cluster_id", "cluster_label", "cluster_label_long"]]
        .drop_duplicates()
        .sort_values("cluster_id")
        .to_string(index=False)
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
