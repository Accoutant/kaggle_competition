import pandas as pd
from pathlib import Path
import json
from geopy.geocoders import Nominatim

pd.set_option('display.max_columns', 500)
files = list(Path("data/").glob("*.csv"))
for file in files:
    print(f"creates: '{file.stem}'")
    globals()[file.stem] = pd.read_csv(file)
f = open('../data/county_id_to_name_map.json')
county_codes = json.load(f)
parsed_counties = {v.lower().rstrip("maa"): k for k, v in county_codes.items()}
DATA_DIR = "../data/"

# Read CSVs and parse relevant date columns
train = pd.read_csv(DATA_DIR + "train.csv")
client = pd.read_csv(DATA_DIR + "client.csv")
historical_weather = pd.read_csv(DATA_DIR + "historical_weather.csv")
forecast_weather = pd.read_csv(DATA_DIR + "forecast_weather.csv")
electricity = pd.read_csv(DATA_DIR + "electricity_prices.csv")
gas = pd.read_csv(DATA_DIR + "gas_prices.csv")
name_mapping = {
    "valga": "valg",
    "põlva": "põlv",
    "jõgeva": "jõgev",
    "rapla": "rapl",
    "järva": "järv"
}

for i, coords in  forecast_weather[["latitude", "longitude"]].drop_duplicates().iterrows():

    lat, lon = coords["latitude"], coords["longitude"]

    geoLoc = Nominatim(user_agent="GetLoc")

    # passing the coordinates
    locname = geoLoc.reverse(f"{lat}, {lon}")   # lat, lon
    if locname is None: continue

    location = locname.raw["address"]
    if location["country"] == "Eesti":
        county = location['county'].split()[0].lower()
        county = name_mapping.get(county, county)
        print(f"county: '{county}', county code:", parsed_counties[county], (lat, lon))