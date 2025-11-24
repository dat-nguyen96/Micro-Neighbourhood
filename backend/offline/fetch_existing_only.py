#!/usr/bin/env python3
"""
Offline data pipeline - 2 functies:
1. fetch_all_data() - Haalt alle data op en schrijft naar CSV
2. build_clusters() - Bouwt KMeans clusters op basis van relevante features
   en laat AI achteraf de clusters betekenisvolle labels geven, gekozen uit
   een vaste set van 8 labels (incl. Vinex-wijk).

Clustering-features:
- Bevolkingsdichtheid_34
- Geweld per 1.000 inwoners (crime_violence_total / AantalInwoners_5 * 1000)
- MateVanStedelijkheid_122 (1 = zeer stedelijk, 5 = landelijk)
- Leeftijdsopbouw (aandeel jongeren 0–25, aandeel 65+)
- GemiddeldeHuishoudensgrootte_33
- HuishoudensMetKinderen_32
- PercentageMeergezinswoning_45 (hoogbouw/appartementen)
- PercentageEengezinswoning_40 (rijtjes/vrijstaand)
- Studenten_per_1000 (MBO + HBO + WO per 1000 inwoners)
- GemiddeldInkomenPerInwoner_78

AI-labels:
GPT kiest per cluster één label uit deze lijst (exacte naam verplicht):
- "Druk stadscentrum"
- "Stedelijke appartementen"
- "Rustige Vinex-wijk"
- "Landelijke dorpskern"
- "Studentenbuurt"
- "Seniorenwijk"
- "Gezinsrijke buitenwijk"
- "Gemengde woon-werkwijk"
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
from openai import OpenAI

# === Environment ===
env_path = Path("../.env")
if env_path.exists():
    load_dotenv(env_path)

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
    GPT mag ALLEEN kiezen uit ALLOWED_LABELS en moet ieder label precies één keer gebruiken.

    Retourneert:
        cluster_labels: dict {cluster_id: korte_naam}
        cluster_labels_long: dict {cluster_id: lange_beschrijving}
    """
    if client is None or OPENAI_API_KEY is None:
        print("⚠️ Geen OPENAI_API_KEY gevonden, gebruik fallback cluster labels.")
        raise RuntimeError("No OPENAI_API_KEY")

    # --- vaste lijst van 8 labels -----------------------------

    # Alleen uit deze labels mag GPT kiezen, en elk label moet precies 1x gebruikt worden.
    allowed_labels = {
        "Druk stadscentrum": (
            "Zeer hoge bevolkingsdichtheid, sterk stedelijk, veel voorzieningen, weinig gezinnen met kinderen."
        ),
        "Stedelijke appartementen": (
            "Hoge dichtheid, sterk stedelijk, veel meergezinswoningen/hoogbouw, veel alleenstaanden, weinig kinderen."
        ),
        "Studentenbuurt": (
            "Hoge concentratie studenten en jongeren (15-25 jaar), kleine huishoudens, vaak nabij onderwijsinstellingen."
        ),
        "Gezinsrijke Vinex-wijk": (
            "Nieuwe/relatief jonge woonwijken met veel eengezinswoningen, veel huishoudens met kinderen, matig tot weinig stedelijk."
        ),
        "Voorstedelijke gezinswijk": (
            "Eengezinswoningen, veel gezinnen, matige bevolkingsdichtheid, rand van stad of grotere kern."
        ),
        "Landelijke dorpskern": (
            "Lage bevolkingsdichtheid, weinig stedelijk, gemengde leeftijden, dorps karakter, relatief veel eigen woningen."
        ),
        "Seniorenwijk": (
            "Hoog aandeel 65+, lage aandelen jongeren, vaak rustige omgeving met beperkte kinderdichtheid."
        ),
        "Gemengde woon-werkwijk": (
            "Gemiddelde dichtheid, mix van wonen en werken (relatief veel bedrijven), gemengde demografie."
        ),
    }

    # --- welke features vatten we per cluster samen? ----------
    feature_cols_for_summary = [
        "Bevolkingsdichtheid_34",
        "violence_per_1000",
        "MateVanStedelijkheid_122",
        "aandeel_jongeren_0_25",
        "aandeel_65_plus",
        "GemiddeldeHuishoudensgrootte_33",
        "HuishoudensMetKinderen_32",
        "PercentageMeergezinswoning_45",
        "PercentageEengezinswoning_40",
        "studenten_per_1000",
        "recent_build_share",
        "bedrijven_per_1000",
    ]
    # Hou alleen kolommen die echt bestaan
    feature_cols_for_summary = [
        c for c in feature_cols_for_summary if c in df_clean.columns
    ]

    # Maak een compacte samenvatting per cluster
    rows = []
    for cid in range(n_clusters):
        mask = clusters == cid
        sub = df_clean.loc[mask]

        row_data = {"cluster_id": cid, "count": int(mask.sum())}
        if sub.empty:
            for col in feature_cols_for_summary:
                row_data[col] = 0.0
        else:
            for col in feature_cols_for_summary:
                row_data[col] = float(sub[col].mean())

        rows.append(row_data)

    # Bouw CSV-achtige tekst
    header = ["cluster_id", "count"] + feature_cols_for_summary
    lines = [",".join(header)]
    for r in rows:
        values = [str(r["cluster_id"]), str(r["count"])]
        for feat in feature_cols_for_summary:
            values.append(str(r[feat]))
        lines.append(",".join(values))

    csv_summary = "\n".join(lines)

    # Uitleg voor GPT over de features
    uitleg_features = """
Feature-uitleg (kolomnamen beginnen in de CSV met 'avg_'):

- Bevolkingsdichtheid_34: inwoners per km² (hoog = druk, laag = rustig/landelijk).
- violence_per_1000: aantal geweldsincidenten per 1.000 inwoners (hoog = onveiliger).
- MateVanStedelijkheid_122: schaal 1–5:
    1 = zeer sterk stedelijk
    2 = sterk stedelijk
    3 = matig stedelijk
    4 = weinig stedelijk
    5 = niet stedelijk (landelijk).
- aandeel_jongeren_0_25: aandeel bevolking 0-25 jaar (hoog = veel jongeren/kinderen).
- aandeel_65_plus: aandeel bevolking 65+ (hoog = seniorenwijk).
- GemiddeldeHuishoudensgrootte_33: gemiddeld aantal personen per huishouden.
- HuishoudensMetKinderen_32: aandeel huishoudens met kinderen.
- PercentageMeergezinswoning_45: aandeel appartementen/hoogbouw.
- PercentageEengezinswoning_40: aandeel eengezinswoningen.
- studenten_per_1000: aantal studenten (MBO/HBO/WO) per 1000 inwoners.
- recent_build_share: aandeel woningen gebouwd in de afgelopen 10 jaar (hoger = nieuwer, vaak Vinex).
- bedrijven_per_1000: aantal bedrijfsvestigingen per 1000 inwoners (hoog = meer werkfuncties).
"""

    label_list_text = "\n".join(
        [f"- {name}: {desc}" for name, desc in allowed_labels.items()]
    )

    system_msg = (
        "Je bent een Nederlandse data-analist gespecialiseerd in woonbuurten. "
        "Je krijgt clusters van Nederlandse buurten met samenvattingen van demografie, woningvoorraad, "
        "stedelijkheid en geweldsincidenten. "
        "Je moet voor elk cluster één label kiezen uit een VASTE lijst van 8 labels. "
        "Je mag GEEN nieuwe labels verzinnen. Elke labelnaam mag maar één keer gebruikt worden."
    )

    user_msg = f"""

Hier is een uitleg van de features:

{uitleg_features}

Hier is de lijst van toegestane labels (je mag alleen uit deze namen kiezen, elk label precies 1 keer):

{label_list_text}

Hier is de numerieke samenvatting van de clusters (per cluster de gemiddelden):

{csv_summary}

OPDRACHT:

- Koppel elk cluster_id aan EXACT één van de bovenstaande label-namen.
- Gebruik elk label UITSLUITEND één keer (dus 8 clusters = 8 verschillende labels).
- Kies de label die het beste past bij de gemiddelde kenmerken van dat cluster.

Geef voor iedere cluster_id precies één regel in dit formaat:

cluster_id | KORTE_LABEL | LANGE_BESCHRIJVING

Waar KORTE_LABEL exact één van de 8 namen is uit de lijst:
{", ".join(allowed_labels.keys())}

Geen extra tekst, geen uitleg, geen markdown. Alleen de regels.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=900,
        )

        content = response.choices[0].message.content.strip()
        cluster_labels = {}
        cluster_labels_long = {}
        used_labels = set()

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

        # Validatie: alle clusters gelabeld
        missing_clusters = [cid for cid in range(n_clusters) if cid not in cluster_labels]
        if missing_clusters:
            raise ValueError(f"Ontbrekende labels voor clusters: {missing_clusters}")

        # Validatie: labels komen uit allowed_labels
        for lbl in cluster_labels.values():
            if lbl not in allowed_labels:
                raise ValueError(f"Onbekend label gebruikt door AI: {lbl}")

        # Validatie: labels zijn uniek
        if len(set(cluster_labels.values())) != len(cluster_labels):
            raise ValueError("Korte labels zijn niet uniek, AI-output ongeldig.")

        return cluster_labels, cluster_labels_long

    except Exception as e:
        print(f"⚠️ Fout bij genereren AI labels: {e}")
        raise


# ======================================================================
# 1. DATA FETCHEN (zoals bij jou, alleen licht opgeschoond)
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
    # 4. Crime data 47018NED (zoals in jouw versie)
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

        total_keys = []
        sexual_violence_keys = []
        other_violence_keys = []
        property_keys = []
        vandalism_keys = []

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

        detailed_categories = list(
            set(
                total_keys
                + sexual_violence_keys
                + other_violence_keys
                + property_keys
                + vandalism_keys
            )
        )

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
                "$filter": f"Perioden eq '{crime_period}' and SoortMisdrijf eq '{cat}'",
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
    # 5. Crime verwerken + merge met buurten
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

        crime_agg = (
            crime_df.groupby([regio_col, "crime_category"])[crime_col]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )
        crime_agg = crime_agg.rename(columns={regio_col: "WijkenEnBuurten"})

        for needed in [
            "total_crimes",
            "sexual_violence",
            "other_violence",
            "property",
            "vandalism",
        ]:
            if needed not in crime_agg.columns:
                crime_agg[needed] = 0

        crime_agg["crime_violence_total"] = (
            crime_agg["sexual_violence"] + crime_agg["other_violence"]
        )

        if crime_agg["total_crimes"].sum() == 0:
            crime_agg["total_crimes"] = (
                crime_agg["crime_violence_total"]
                + crime_agg["property"]
                + crime_agg["vandalism"]
            )

        crime_agg = crime_agg.rename(
            columns={
                "total_crimes": "total_crimes",
                "sexual_violence": "crime_sexual_violence",
                "other_violence": "crime_other_violence",
                "property": "crime_property",
                "vandalism": "crime_vandalism",
            }
        )

        crime_agg = crime_agg.merge(
            detailed_crime_pivot, on="WijkenEnBuurten", how="left"
        )

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

    Clustering-features (indien aanwezig):
    - Bevolkingsdichtheid_34
    - Geweld (incl. seksueel) per 1.000 inwoners (violence_per_1000)
    - MateVanStedelijkheid_122 (1 = zeer stedelijk, 5 = landelijk)
    - aandeel_jongeren_0_25
    - aandeel_65_plus
    - GemiddeldeHuishoudensgrootte_33
    - HuishoudensMetKinderen_32
    - PercentageMeergezinswoning_45 (appartementen)
    - PercentageEengezinswoning_40
    - students_per_1000
    - recent_build_share (nieuwbouw-aandeel -> Vinex-achtig)
    - bedrijven_per_1000 (woon-werkwijk-gevoel)

    Labels:
    - Met AI gegenereerde korte en lange cluster-namen uit een vaste set van 8 labels.
    """
    print("=== Start cluster bouw ===")

    df = pd.read_csv(RAW_DATA_CSV)
    print(f"Data geladen: {len(df)} rijen")

    # Basis check
    if "crime_violence_total" not in df.columns or "AantalInwoners_5" not in df.columns:
        raise ValueError("Benodigde kolommen voor geweld per 1000 ontbreken.")

    # Geweld per 1.000 inwoners
    df["violence_per_1000"] = 0.0
    mask_pop = df["AantalInwoners_5"] > 0
    df.loc[mask_pop, "violence_per_1000"] = (
        df.loc[mask_pop, "crime_violence_total"] / df.loc[mask_pop, "AantalInwoners_5"] * 1000.0
    )

    # Leeftijd features
    df["aandeel_jongeren_0_25"] = df.get("k_0Tot15Jaar_8", 0).fillna(0) + df.get("k_15Tot25Jaar_9", 0).fillna(0)
    df["aandeel_65_plus"] = df.get("k_65JaarOfOuder_12", 0).fillna(0)

    # --- Studenten per 1000 -------------------------------------------

    if {"StudentenMboExclExtranei_64", "StudentenHbo_65", "StudentenWo_66"}.issubset(df.columns):
        df["students_per_1000"] = 0.0
        total_students = (
            df["StudentenMboExclExtranei_64"].fillna(0) +
            df["StudentenHbo_65"].fillna(0) +
            df["StudentenWo_66"].fillna(0)
        )
        df.loc[mask_pop, "students_per_1000"] = total_students[mask_pop] / df.loc[mask_pop, "AantalInwoners_5"] * 1000.0
    else:
        df["students_per_1000"] = 0.0

    # --- Nieuwbouw-aandeel (Vinex-achtig) -----------------------------

    if {"BouwjaarAfgelopenTienJaar_52", "Woningvoorraad_35"}.issubset(df.columns):
        df["recent_build_share"] = 0.0
        mask_woning = df["Woningvoorraad_35"] > 0
        df.loc[mask_woning, "recent_build_share"] = (
            df.loc[mask_woning, "BouwjaarAfgelopenTienJaar_52"].fillna(0) /
            df.loc[mask_woning, "Woningvoorraad_35"]
        )
    else:
        df["recent_build_share"] = 0.0

    # --- Bedrijven per 1000 inwoners (woon-werkmix) -------------------

    if "BedrijfsvestigingenTotaal_97" in df.columns:
        df["bedrijven_per_1000"] = 0.0
        df.loc[mask_pop, "bedrijven_per_1000"] = (
            df["BedrijfsvestigingenTotaal_97"].fillna(0)[mask_pop] /
            df.loc[mask_pop, "AantalInwoners_5"] * 1000.0
        )
    else:
        df["bedrijven_per_1000"] = 0.0

    # --- Featurelijst voor clustering --------------------------------

    candidate_features = [
        "Bevolkingsdichtheid_34",         # drukte
        "violence_per_1000",              # veiligheid
        "MateVanStedelijkheid_122",       # stedelijk vs landelijk
        "aandeel_jongeren_0_25",          # jongeren/kinderen
        "aandeel_65_plus",                # ouderen
        "GemiddeldeHuishoudensgrootte_33",
        "HuishoudensMetKinderen_32",
        "PercentageMeergezinswoning_45",
        "PercentageEengezinswoning_40",
        "students_per_1000",
        "recent_build_share",
        "bedrijven_per_1000",
    ]

    feature_cols = [c for c in candidate_features if c in df.columns]

    if len(feature_cols) < 2:
        raise ValueError(f"Te weinig features voor clustering. Gevonden: {feature_cols}")



    df_features = df[feature_cols].copy()



    # Filter rijen met complete data

    mask_complete = df_features.notna().all(axis=1)

    df_clean = df[mask_complete].reset_index(drop=True)

    df_features = df_features[mask_complete].reset_index(drop=True)



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

    # === AI-clusterlabels genereren op basis van df_clean (inclusief extra kolommen) ===

    try:

        ai_short, ai_long = generate_ai_cluster_labels(df_clean, clusters, n_clusters)

        print("AI cluster labels succesvol gegenereerd.")

    except Exception as e:

        print(f"AI label generatie mislukt: {e}")

        print("Gebruik fallback cluster labels.")

        ai_short = {i: f"Cluster {i}" for i in range(n_clusters)}

        ai_long = {

            i: "Een woonomgeving met eigen kenmerken op het gebied van bevolkingsdichtheid, stedelijkheid, demografie en geweldscriminaliteit."

            for i in range(n_clusters)

        }

    print("Cluster labels toegewezen:")
    for i in range(n_clusters):
        cluster_count = int((clusters == i).sum())
        print(f"  Cluster {i}: {ai_short[i]} ({cluster_count} buurten)")

    # Resultaat dataframe - alle originele kolommen bewaren

    result_df_full = df.loc[mask_complete].reset_index(drop=True)

    result_df_full["cluster_id"] = clusters

    result_df_full["cluster_label"] = result_df_full["cluster_id"].map(ai_short)

    result_df_full["cluster_label_long"] = result_df_full["cluster_id"].map(ai_long)

    result_df_full["pca_x"] = X_pca[:, 0]

    result_df_full["pca_y"] = X_pca[:, 1]

    result_df_full.to_csv(CLUSTERS_CSV, index=False)
    print(
        f"=== Clusters geschreven naar {CLUSTERS_CSV} "
        f"({len(result_df_full)} rijen, {n_clusters} clusters) ==="
    )

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
