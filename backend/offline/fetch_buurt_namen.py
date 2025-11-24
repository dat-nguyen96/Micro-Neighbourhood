#!/usr/bin/env python3
"""
Haal alle buurtcodes + namen op van CBS 83765NED
"""
import requests
import pandas as pd
from pathlib import Path
from typing import Union, Optional


def fetch_buurt_namen_to_csv(
    table_id: str = "83765NED",
    outfile: Union[str, Path] = "../data/cbs_buurt_namen_83765.csv",
    max_rows: Optional[int] = None,  # Test: beperk aantal rijen
):
    """
    Haal alle codes + namen van WijkenEnBuurten op voor 83765NED via OData v4
    en schrijf naar CSV met kolommen:
      - Identifier  (bijv. BU04053306)
      - Title       (buurtnaam)
      - ParentTitle / andere metadata (optioneel)
    """
    base = f"https://datasets.cbs.nl/odata/v1/CBS/{table_id}"
    url = base + "/WijkenEnBuurtenCodes"

    rows = []
    page = 0

    while url:
        page += 1
        print(f"Ophalen pagina {page}: {url}")

        resp = requests.get(url)
        resp.raise_for_status()

        data = resp.json()
        batch_rows = data.get("value", [])

        # Filter alleen buurten (BU codes)
        buurten = [row for row in batch_rows if str(row.get("Identifier", "")).startswith("BU")]
        rows.extend(buurten)
        print(f"  -> {len(batch_rows)} totaal, {len(buurten)} buurten opgehaald (totaal buurten: {len(rows)})")

        # Check max_rows limit
        if max_rows is not None and len(rows) >= max_rows:
            print(f"max_rows={max_rows} bereikt, stoppen met ophalen")
            rows = rows[:max_rows]
            break

        url = data.get("@odata.nextLink")  # v4 paginatie

    df = pd.DataFrame(rows)
    print(f"Totaal {len(df)} rijen opgehaald")

    # Beperk tot relevante kolommen
    keep_cols = [c for c in df.columns if c in ("Identifier", "Title", "ParentTitle")]
    df = df[keep_cols]

    out = Path(outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("Buurt-namen CSV geschreven naar:", out.resolve())
    print(f"Voorbeeld rijen:")
    print(df.head())


if __name__ == "__main__":
    # Haal alle buurten op
    fetch_buurt_namen_to_csv(max_rows=None)
