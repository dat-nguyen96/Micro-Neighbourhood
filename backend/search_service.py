# backend/search_service.py - ML Search and Clustering Services
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

try:
    # Try relative imports (when run as package)
    from .data_service import _ensure_cbs_data_loaded, FEATURE_COLUMNS
except ImportError:
    # Fallback to absolute imports (when run as script)
    from data_service import _ensure_cbs_data_loaded, FEATURE_COLUMNS


def analyse_neighbourhood_data(data: Dict[str, Any]) -> Tuple[str, Dict[str, Optional[float]]]:
    """
    Analyseer buurt data en geef een korte Nederlandse samenvatting + gestandaardiseerde metrics.

    Args:
        data: Dictionary met buurt eigenschappen

    Returns:
        Tuple van (samenvatting_tekst, gestandaardiseerde_metrics_dict)
    """
    # Extract relevante metrics
    metrics = {}

    # Bevolkingsdichtheid - normaliseer naar standaard schaal
    density = data.get("Bevolkingsdichtheid_34")
    if density is not None:
        # Nederlandse stedelijkheid: < 500 = landelijk, 500-1000 = matig, 1000-2500 = stedelijk, >2500 = dicht
        if density < 500:
            density_cat = "landelijk"
        elif density < 1000:
            density_cat = "matig stedelijk"
        elif density < 2500:
            density_cat = "stedelijk"
        else:
            density_cat = "zeer stedelijk"
        metrics["bevolkingsdichtheid"] = density
        metrics["density_category"] = density_cat

    # Leeftijdssamenstelling
    age_groups = {
        "kinderen_0_15": data.get("k_0Tot15Jaar_8"),
        "jongeren_15_25": data.get("k_15Tot25Jaar_9"),
        "jongvolwassenen_25_45": data.get("k_25Tot45Jaar_10"),
        "middenleeftijd_45_65": data.get("k_45Tot65Jaar_11"),
        "ouderen_65_plus": data.get("k_65JaarOfOuder_12")
    }

    # Huishouden informatie
    household_size = data.get("GemiddeldeHuishoudensgrootte_33")
    households_with_kids = data.get("HuishoudensMetKinderen_32")

    if household_size is not None:
        if household_size < 2.0:
            household_cat = "kleine huishoudens (singles/studenten)"
        elif household_size < 2.5:
            household_cat = "gemiddelde huishoudens"
        else:
            household_cat = "grote gezinnen"
        metrics["huishoudgrootte"] = household_size
        metrics["household_category"] = household_cat

    # Stedelijkheid
    stedelijkheid = data.get("MateVanStedelijkheid_122")
    if stedelijkheid is not None:
        stedelijkheid_map = {
            1: "zeer sterk stedelijk",
            2: "sterk stedelijk",
            3: "matig stedelijk",
            4: "weinig stedelijk",
            5: "niet stedelijk (landelijk)"
        }
        stedelijkheid_cat = stedelijkheid_map.get(int(stedelijkheid), f"stedelijkheid niveau {stedelijkheid}")
        metrics["stedelijkheid_score"] = stedelijkheid
        metrics["stedelijkheid_category"] = stedelijkheid_cat

    # Criminaliteit (indien beschikbaar)
    violence_total = data.get("crime_violence_total")
    population = data.get("AantalInwoners_5")

    if violence_total is not None and population is not None and population > 0:
        violence_per_1000 = (violence_total / population) * 1000

        # Nederlandse veiligheidscontext
        if violence_per_1000 < 2:
            safety_cat = "zeer veilig"
        elif violence_per_1000 < 5:
            safety_cat = "veilig"
        elif violence_per_1000 < 10:
            safety_cat = "gemiddeld veilig"
        else:
            safety_cat = "minder veilig"

        metrics["geweld_per_1000"] = violence_per_1000
        metrics["safety_category"] = safety_cat

    # Bouw samenvatting
    parts = []

    if "density_category" in metrics:
        parts.append(f"Een {metrics['density_category']} gebied")

    if "household_category" in metrics:
        parts.append(f"met {metrics['household_category']}")

    if "stedelijkheid_category" in metrics:
        parts.append(f"in een {metrics['stedelijkheid_category']} omgeving")

    if "safety_category" in metrics:
        parts.append(f"dat als {metrics['safety_category']} wordt ervaren")

    summary = "Dit is " + " ".join(parts) + "."

    # Geef ook de ruwe metrics terug voor verdere processing
    raw_metrics = {
        "bevolkingsdichtheid": metrics.get("bevolkingsdichtheid"),
        "gemiddelde_huishoudgrootte": metrics.get("huishoudgrootte"),
        "percentage_kinderen": age_groups.get("kinderen_0_15"),
        "percentage_jongeren": age_groups.get("jongeren_15_25"),
        "percentage_ouderen": age_groups.get("ouderen_65_plus"),
        "stedelijkheid_score": metrics.get("stedelijkheid_score"),
        "geweld_per_1000": metrics.get("geweld_per_1000")
    }

    return summary, raw_metrics


def find_similar_neighbourhoods(buurt_code: str, n_similar: int = 5) -> List[Dict[str, Any]]:
    """
    Vind vergelijkbare buurten op basis van ML features met KNN.

    Args:
        buurt_code: CBS buurt code
        n_similar: Aantal vergelijkbare buurten om terug te geven

    Returns:
        Lijst van dictionaries met vergelijkbare buurten
    """
    print(f"🔍 Finding similar neighbourhoods for {buurt_code}")
    df = _ensure_cbs_data_loaded()
    print(f"📊 Data loaded: {len(df)} rows")

    # Controleer of buurt bestaat
    try:
        target_idx = df[df["WijkenEnBuurten"] == buurt_code].index[0]
        print(f"✅ Buurt gevonden op index {target_idx}")
    except IndexError:
        raise ValueError(f"Buurt code '{buurt_code}' niet gevonden")

    # Gebruik alle numerieke features voor KNN (inclusief nieuwe demografische features)
    numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and not col.startswith('pca_')]

    # Filter out non-feature columns
    exclude_cols = ['ID', 'cluster_id']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    print(f"🔢 Found {len(numeric_cols)} numeric features: {numeric_cols[:5]}...")

    if len(numeric_cols) < 3:
        raise RuntimeError(f"Niet genoeg numerieke features voor KNN: {len(numeric_cols)} gevonden")

    print(f"🔍 KNN gebruikt {len(numeric_cols)} features")

    # Prepare data voor KNN
    X = df[numeric_cols].fillna(0)  # Fill NaN met 0 voor eenvoud

    # Standaardiseer
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KNN model
    knn = NearestNeighbors(n_neighbors=n_similar + 1, metric='euclidean')  # +1 omdat target buurt zelf ook matcht
    knn.fit(X_scaled)

    # Vind neighbors
    target_vector = X_scaled[target_idx].reshape(1, -1)
    distances, indices = knn.kneighbors(target_vector)

    # Exclude de target buurt zelf (eerste neighbor)
    similar_indices = indices[0][1:]  # Skip eerste (is target zelf)
    similar_distances = distances[0][1:]

    # Bouw resultaten
    results = []
    for idx, distance in zip(similar_indices, similar_distances):
        row = df.iloc[idx]

        # Haal gemeente en buurt naam uit de data
        buurt_code_similar = str(row.get("WijkenEnBuurten", f"BU{idx}"))
        gemeente = str(row.get("Gemeentenaam_1", "")).strip() or "Onbekend"
        naam = None  # CBS data heeft geen buurt namen, alleen codes

        # Cluster info - gebruik dezelfde logica als data_service
        cluster_id = int(row.get("cluster_id", 0))

        # Probeer cluster_label_short eerst, dan cluster_label
        cluster_label = row.get("cluster_label_short")
        if pd.isna(cluster_label) or cluster_label is None:
            cluster_label = row.get("cluster_label")

        if pd.isna(cluster_label) or cluster_label is None:
            cluster_label = f"Cluster {cluster_id}"

        cluster_description = row.get("cluster_label_long", "")
        if pd.isna(cluster_description) or cluster_description is None:
            cluster_description = ""

        cluster_label = str(cluster_label)
        cluster_description = str(cluster_description)

        results.append({
            "buurt_code": buurt_code_similar,
            "gemeente": gemeente,
            "naam": naam,  # None is OK, frontend handelt dit af
            "distance": float(distance),
            "cluster_label": cluster_label,
            "cluster_description": cluster_description
        })

    return results
