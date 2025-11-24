# backend/offline/test_crime_api.py

import requests

def test_crime_api():
    """Test if we can access the CBS crime data API"""

    url = "https://dataderden.cbs.nl/ODataApi/OData/47018NED/TypedDataSet?$top=5"

    print(f"Testing CBS crime API: {url}")

    try:
        resp = requests.get(url)
        print(f"Status: {resp.status_code}")

        if resp.ok:
            data = resp.json()
            rows = data.get("value", [])
            print(f"Received {len(rows)} rows")

            if rows:
                print("Sample row keys:", list(rows[0].keys()))
                print("Sample row:", rows[0])
        else:
            print(f"Error: {resp.text}")

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_crime_api()
