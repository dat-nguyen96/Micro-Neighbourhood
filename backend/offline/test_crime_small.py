# backend/offline/test_crime_small.py

import requests
import pandas as pd
from pathlib import Path

def fetch_crime_small_test():
    """Fetch just a small amount of crime data for testing"""

    base_url = "https://dataderden.cbs.nl/ODataApi/OData/47018NED/TypedDataSet"

    print("Fetching small sample of CBS crime data...")

    # Just fetch first 1000 rows
    url = f"{base_url}?$top=1000"
    print(f"URL: {url}")

    resp = requests.get(url)
    if not resp.ok:
        print(f"Error: HTTP {resp.status_code}")
        return

    data = resp.json()
    rows = data.get("value", [])
    print(f"Received {len(rows)} rows")

    if rows:
        df = pd.DataFrame(rows)
        outfile = Path("data/cbs_crime_test.csv")
        outfile.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outfile, index=False)
        print(f"Saved to {outfile}")

        # Show sample of columns
        print("Columns:", list(df.columns))
        print("Sample row:", df.iloc[0].to_dict())

if __name__ == "__main__":
    fetch_crime_small_test()
