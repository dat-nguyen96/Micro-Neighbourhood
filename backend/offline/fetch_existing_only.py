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

# === Vaste label-catalogus (8 labels) ===
ALLOWED_LABELS = [
    "Druk stadscentrum",
    "Stedelijke appartementen",
    "Rustige Vinex-wijk",
    "Landelijke dorpskern",
    "Studentenbuurt",
    "Seniorenwijk",
    "Gezinsrijke buitenwijk",
    "Gemengde woon-werkwijk",
]

LABEL_DESCRIPTIONS = {
    "Druk stadscentrum": (
        "Een zeer stedelijke buurt met hoge bevolkingsdichtheid, veel voorzieningen en vaak "
        "meer geweldsincidenten door drukte, uitgaan en toerisme."
    ),
    "Stedelijke appartementen": (
        "Een stedelijke appartementenwijk met veel meergezinswoningen, relatief kleine huishoudens "
        "en een gemengde bevolking van starters, singles en tweeverdieners."
    ),
    "Rustige Vinex-wijk": (
        "Een moderne, relatief nieuwe buitenwijk met veel eengezinswoningen, gezinnen met kinderen, "
        "gemiddelde bevolkingsdichtheid en een laag geweldsniveau."
    ),
    "Landelijke dorpskern": (
        "Een weinig tot niet stedelijke buurt met lage bevolkingsdichtheid, veel ruimte, "
        "lokale voorzieningen en vaak een gemengde leeftijdsopbouw."
    ),
    "Studentenbuurt": (
        "Een sterk stedelijke buurt met veel jongeren (15–25 jaar), relatief kleine huishoudens, "
        "veel studenten en een levendig nacht- en uitgaansleven."
    ),
    "Seniorenwijk": (
        "Een rustige buurt met relatief veel 65-plussers, weinig huishoudens met kinderen en een "
        "lager aandeel jongeren."
    ),
    "Gezinsrijke buitenwijk": (
        "Een gematigd tot weinig stedelijke wijk met veel gezinnen, grotere huishoudens en veel "
        "eengezinswoningen."
    ),
    "Gemengde woon-werkwijk": (
        "Een gemengde buurt met zowel wonen als werken, gemiddelde bevolkingsdichtheid en een "
        "mix van huishoudtypes en inkomens."
    ),
}

# ======================================================================
# AI HELPER: Cluster labels genereren
# ======================================================================

def generate_ai_cluster_labels(df_clean: pd.DataFrame, clusters, n_clusters: int):
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

    # Welke features willen we samenvatten voor GPT?
    summary_features = [
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
        "GemiddeldInkomenPerInwoner_78",
    ]
    summary_features = [f for f in summary_features if f in df_clean.columns]

    # Maak een compacte numerieke samenvatting per cluster
    rows = []
    for cid in range(n_clusters):
        mask = clusters == cid
        sub = df_clean.loc[mask]

        row = {
            "cluster_id": cid,
            "count": int(mask.sum()),
        }

        if sub.empty:
            for feat in summary_features:
                row[feat] = 0.0
        else:
            for feat in summary_features:
                row[feat] = round(float(sub[feat].mean()), 3)

        rows.append(row)

    # Bouw CSV-achtige tekst
    header = ["cluster_id", "count"] + summary_features
    lines = [",".join(header)]
    for r in rows:
        values = [str(r["cluster_id"]), str(r["count"])]
        for feat in summary_features:
            values.append(str(r[feat]))
        lines.append(",".join(values))

    csv_summary = "\n".join(lines)

    # Uitleg features in natuurlijke taal
    feature_expl = """
Feature-uitleg:
- Bevolkingsdichtheid_34: aantal inwoners per km2 (hoe hoger, hoe drukker).
- violence_per_1000: aantal geweldsincidenten per 1.000 inwoners.
- MateVanStedelijkheid_122: schaal 1–5 (1 = zeer sterk stedelijk, 5 = niet stedelijk/landelijk).
- aandeel_jongeren_0_25: percentage inwoners tot en met 24 jaar.
- aandeel_65_plus: percentage inwoners van 65 jaar en ouder.
- GemiddeldeHuishoudensgrootte_33: gemiddeld aantal personen per huishouden.
- HuishoudensMetKinderen_32: percentage huishoudens met kinderen.
- PercentageMeergezinswoning_45: aandeel appartementen/hoogbouw.
- PercentageEengezinswoning_40: aandeel eengezinswoningen (rijtjes, vrijstaand, 2-onder-1-kap).
- studenten_per_1000: aantal MBO/HBO/WO-studenten per 1.000 inwoners.
- GemiddeldInkomenPerInwoner_78: gemiddeld inkomen per inwoner.
"""

    label_list_text = "\n".join(
        f"- {name}: {LABEL_DESCRIPTIONS[name]}"
        for name in ALLOWED_LABELS
    )

    system_msg = (
        "Je bent een Nederlandse data-analist gespecialiseerd in woonbuurten. "
        "Je krijgt clusters van Nederlandse buurten met een numerieke samenvatting van drukte, geweld, "
        "stedelijkheid, leeftijdsopbouw, woningtype, studenten en inkomen. "
        "Je taak is om voor elk cluster één label te kiezen uit een VASTE lijst van 8 woonconcepten."
    )

    user_msg = f"""
Hier is de uitleg van de features:
{feature_expl}

Hieronder staat de vaste lijst van labels waaruit je MOET kiezen.
Je mag GEEN eigen labels verzinnen. Gebruik ieder label PRECIES één keer:

{label_list_text}

Hieronder zie je de numerieke samenvatting per cluster:

{csv_summary}

Opdracht:
- Er zijn 8 clusters (cluster_id 0 t/m 7) en 8 labels.
- Koppel elk cluster aan precies één label uit de lijst.
- Gebruik elk label exact één keer.
- Kies de label die het beste past bij de gemiddelde kenmerken van dat cluster.

Output-formaat (één cluster per regel, GEEN extra tekst):

cluster_id | korte_label | lange_beschrijving

Waar:
- korte_label is EXACT één van de labels uit de lijst.
- lange_beschrijving is een Nederlandse omschrijving waarom dat label past bij het cluster.
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

            if short_label not in ALLOWED_LABELS:
                raise ValueError(f"Onbekend label '{short_label}' in AI-output.")
            if short_label in used_labels:
                raise ValueError(f"Label '{short_label}' wordt meerdere keren gebruikt.")
            used_labels.add(short_label)

            cluster_labels[cid] = short_label
            cluster_labels_long[cid] = long_label

        # Check of we voor alle clusters iets hebben
        missing_ids = [cid for cid in range(n_clusters) if cid not in cluster_labels]
        if missing_ids:
            raise ValueError(f"Ontbrekende labels voor clusters: {missing_ids}")

        # Check of alle labels zijn gebruikt
        if used_labels != set(ALLOWED_LABELS):
            raise ValueError("Niet alle toegestane labels zijn gebruikt.")

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

    Features:
    - Bevolkingsdichtheid_34
    - violence_per_1000
    - MateVanStedelijkheid_122
    - aandeel_jongeren_0_25
    - aandeel_65_plus
    - GemiddeldeHuishoudensgrootte_33
    - HuishoudensMetKinderen_32
    - PercentageMeergezinswoning_45
    - PercentageEengezinswoning_40
    - studenten_per_1000
    - GemiddeldInkomenPerInwoner_78

    AI geeft achteraf labels uit ALLOWED_LABELS.
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

    # Studenten per 1000
    for col in ["StudentenMboExclExtranei_64", "StudentenHbo_65", "StudentenWo_66"]:
        if col not in df.columns:
            df[col] = 0
    df["studenten_totaal"] = (
        df["StudentenMboExclExtranei_64"].fillna(0)
        + df["StudentenHbo_65"].fillna(0)
        + df["StudentenWo_66"].fillna(0)
    )
    df["studenten_per_1000"] = 0.0
    df.loc[mask_pop, "studenten_per_1000"] = (
        df.loc[mask_pop, "studenten_totaal"] / df.loc[mask_pop, "AantalInwoners_5"] * 1000.0
    )

    # Features voor clustering
    feature_cols = [
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
        "GemiddeldInkomenPerInwoner_78",
    ]

    # Filter op bestaande kolommen
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise ValueError("Geen geldige feature kolommen gevonden voor clustering.")

    df_features = df[feature_cols].copy()

    # Missing values vullen met 0 (veilig voor percentages en dichtheden)
    df_features = df_features.fillna(0)

    # Check of we nog data hebben
    if df_features.empty or len(df_features) == 0:
        raise ValueError("Geen data over na feature selectie.")

    print(f"Data klaar voor clustering: {len(df_features)} rijen")
    print(f"Features gebruikt voor clustering: {feature_cols}")

    # Gebruik alle data (geen extra filtering)
    df_clean = df.copy()

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

    # AI labels
    try:
        ai_short, ai_long = generate_ai_cluster_labels(df, clusters, n_clusters)
        print("AI cluster labels succesvol gegenereerd.")
    except Exception as e:
        print(f"AI label generatie mislukt: {e}")
        print("Gebruik fallback cluster labels (volgorde van ALLOWED_LABELS).")
        ai_short = {i: ALLOWED_LABELS[i] for i in range(n_clusters)}
        ai_long = {i: LABEL_DESCRIPTIONS[ALLOWED_LABELS[i]] for i in range(n_clusters)}

    print("Cluster labels toegewezen:")
    for i in range(n_clusters):
        cluster_count = int((clusters == i).sum())
        print(f"  Cluster {i}: {ai_short[i]} ({cluster_count} buurten)")

    # Resultaat dataframe - alle originele data bewaren
    result_df_full = df.copy()
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
