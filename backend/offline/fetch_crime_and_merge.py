import requests
import pandas as pd
from pathlib import Path
from typing import Union, Optional


def fetch_crime_raw_to_csv(
    table_id: str = "47018NED",
    outfile: Union[Path, str] = "data/cbs_crime_raw.csv",
    year_filter: str = "2024JJ00",
    max_rows: Optional[int] = None,
):
    """
    Haal alle rijen van de Politie/CBS-tabel (default: 47018NED) voor een specifiek jaar
    via ODataApi, waarbij we steeds @odata.nextLink volgen i.p.v. handmatig $skip.

    Zo voorkom je infinite loops als de server $skip negeert of met skiptokens werkt.
    """
    base_url = f"https://dataderden.cbs.nl/ODataApi/OData/{table_id}/TypedDataSet"
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching CBS {table_id} for year {year_filter}...")
    print(f"Base URL: {base_url}")

    all_rows: list[dict] = []
    batch_size = 1000

    # eerste call: met filter + top in querystring
    params = {
        "$filter": f"Perioden eq '{year_filter}'",
        "$top": str(batch_size),
    }
    next_url = base_url
    batch_count = 0

    while next_url:
        batch_count += 1
        print(f"Requesting batch {batch_count}: {next_url}")

        if next_url == base_url:
            # eerste request: gebruik params
            resp = requests.get(next_url, params=params)
        else:
            # volgende requests: nextLink bevat al alles (incl. skiptoken)
            resp = requests.get(next_url)

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

        # Optionele harde limiet
        if max_rows is not None and len(all_rows) >= max_rows:
            print(f"max_rows={max_rows} bereikt; stoppen met ophalen.")
            all_rows = all_rows[:max_rows]
            break

        # Volgende URL op basis van @odata.nextLink; als die er niet is, zijn we klaar
        next_url = data.get("@odata.nextLink")
        if not next_url:
            print("Geen @odata.nextLink meer; laatste batch opgehaald.")
            break

    if not all_rows:
        print("Geen data opgehaald; CSV wordt niet geschreven.")
        return

    print(f"Totaal opgehaald: {len(all_rows)} rijen. Schrijf naar CSV...")

    df = pd.DataFrame(all_rows)
    outfile.to_csv(outfile, index=False)
    print(f"Gereed. Geschreven naar: {outfile.resolve()}")
