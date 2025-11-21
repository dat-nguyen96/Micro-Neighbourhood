# Quick test of CBS fetching
import httpx
import pandas as pd

def test_fetch():
    base_url = "https://opendata.cbs.nl/ODataApi/OData/83765NED/TypedDataSet"

    # Test with small batch first
    params = {
        "$skip": 0,
        "$top": 10,  # Just 10 records for testing
        "$filter": "SoortRegio_2 eq 'Buurt'"  # Only buurten
    }

    print("Testing CBS fetch...")
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(base_url, params=params)
        if resp.status_code == 200:
            data = resp.json()["value"]
            print(f"✅ Success! Got {len(data)} records")
            if data:
                df = pd.DataFrame(data[:3])  # Show first 3
                print("Sample columns:", list(df.columns)[:10])
                print("Sample data:")
                print(df.head(3))
        else:
            print(f"❌ Error: {resp.status_code} - {resp.text}")

if __name__ == "__main__":
    test_fetch()
