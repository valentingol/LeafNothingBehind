"""Weather data extraction."""
import argparse
from datetime import datetime
from typing import Tuple, Union

import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from meteostat import Daily, Stations
from tqdm import tqdm


def retrieve_weather(
    geolocator: Nominatim,
    city: str,
    start_date: Union[str, datetime.date],  # type: ignore
    end_date: Union[str, datetime.date] = None,  # type: ignore
    time_format: str = "%Y-%m-%d",
) -> pd.Series:
    """Get weather data for a city and a date range."""
    if geolocator is None:
        geolocator = Nominatim(user_agent="leaf-nothing-behind")

    if end_date is None:
        end_date = start_date
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, time_format or "%Y-%m-%d")
    if isinstance(start_date, str):
        end_date = datetime.strptime(end_date, time_format or "%Y-%m-%d")

    # Use the geolocator to get the location coordinates of the city
    location = geolocator.geocode(city)

    # Get daily data from start to end date
    kilometers = 1000
    stations = Stations()
    stations = stations.nearby(location.latitude, location.longitude, 100 * kilometers)
    print(f"Stations found: {stations.count()}")

    stations = stations.fetch()
    all_data = Daily(stations, start_date, end_date)
    all_data = all_data.fetch()
    for multidx in list(all_data.index):
        data = all_data.loc[multidx[0]]
        if len(data) > 0:
            break

    # print(f"Data fetched from {start_date} to {end_date}")
    # print(data.shape)

    # Add city and date columns
    # data["city"] = city
    # data["date"] = data.index

    return data


def decompose(file_name: str) -> Tuple[str, str, str]:
    """Parse city, start date and end date from input file name."""
    file_name = file_name[0]
    splitted = file_name.split("_")
    city = splitted[0] + " " + splitted[1]
    start_date = splitted[2]
    end_date = splitted[3].rsplit("-", 4)[0]
    return city, start_date, end_date


def extract(csv_path: str) -> None:
    """Extract weather data from csv file and save to csv."""
    dataf = pd.read_csv(csv_path)
    dataf["city"], dataf["start_date"], dataf["end_date"] = zip(
        *dataf.apply(decompose, axis=1),
    )
    # Remove all columns except city, start_date, end_date
    dataf = dataf[["city", "start_date", "end_date"]]
    # Remove duplicates
    dataf = dataf.drop_duplicates()

    meteo_df = pd.DataFrame()

    # Create a geolocator object
    geolocator = Nominatim(user_agent="leaf-nothing-behind")
    for _, row in tqdm(dataf.iterrows(), total=len(dataf)):
        city = row["city"]
        start_date = row["start_date"]
        end_date = row["end_date"]
        # Retrieve weather data
        data = retrieve_weather(geolocator, city, start_date, end_date)
        # Append data to row
        if data.shape[0] == 0:
            meteo_df = meteo_df.append(
                {
                    "city": city,
                    "date": start_date,
                    "temp_avg": 0,
                    "temp_min": 0,
                    "temp_max": 0,
                    "precipitation": 0,
                    "snow": 0,
                    "wind": 0,
                    "wind_dir": 0,
                    "wind_speed": 0,
                    "wind_gust": 0,
                    "pressure": 0,
                    "sun": 0,
                },
                ignore_index=True,
            )
            print("Skipping empty data...")
            continue

        for idx, row in data.iterrows():
            # Replace NaN values with 0
            row = row.fillna(0)
            meteo_df = meteo_df.append(
                {
                    "city": city,
                    "date": idx,
                    "temp_avg": row["tavg"],
                    "temp_min": row["tmin"],
                    "temp_max": row["tmax"],
                    "precipitation": row["prcp"],
                    "snow": row["snow"],
                    "wind": row["wspd"],
                    "wind_dir": row["wdir"],
                    "wind_speed": row["wspd"],
                    "wind_gust": row["wpgt"],
                    "pressure": row["pres"],
                    "sun": row["tsun"],
                },
                ignore_index=True,
            )
    # Save to csv
    dataf.to_csv("..\\data\\meteo.csv", index=False)
    meteo_df.to_csv("..\\data\\meteo_full.csv", index=False)


if __name__ == "__main__":
    # Arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="sequential")
    args = parser.parse_args()
    # Load and process the image series dataframe
    series = pd.read_csv("../data/image_series.csv")
    weather = pd.read_csv("../data/meteo_full.csv")
    # Get weather data
    location = decompose(series.iloc[45])[0]
    weather = weather[weather["city"] == location]
    # Reset index
    weather = weather.reset_index(drop=True)
    # Limit weather data to the 5 last entries
    weather = weather.iloc[-5:]
    print(weather)
    # Convert weather to numpy array and remove city and date columns
    weather = weather.drop(columns=["city", "date"])
    weather = weather.to_numpy()
    print(weather.shape)
    # Convert shape from (5, 11) to (55,)
    weather = weather.reshape(-1)
    weather = np.expand_dims(weather, axis=(1, 2))
    print(weather.shape)
