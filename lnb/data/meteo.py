"""Weather data extraction."""
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
import requests
from geopy.geocoders import Nominatim
from meteostat import Daily, Hourly, Stations
from tqdm import tqdm


def request_patch(slf: Any, *args: Tuple, **kwargs: Dict) -> requests.Response:
    """Patch requests to set a timeout of 5 seconds."""
    timeout = kwargs.pop("timeout", 5)
    return slf.request_orig(*args, **kwargs, timeout=timeout)


setattr(requests.sessions.Session, "request_orig", requests.sessions.Session.request)
requests.sessions.Session.request = request_patch  # type: ignore


def retrieve_weather(
    geolocator: Nominatim,
    city: str,
    start_date: Union[str, datetime.date],  # type: ignore
    end_date: Union[str, datetime.date] = None,  # type: ignore
    time_format: str = "%Y-%m-%d",
    *,
    hourly: bool = True,
) -> pd.Series:
    """Get weather data for a city and a date range."""
    if geolocator is None:
        geolocator = Nominatim(user_agent="leaf-nothing-behind")
    print(start_date, end_date)
    if end_date is None:
        end_date = start_date
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, time_format or "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, time_format or "%Y-%m-%d")

    # Use the geolocator to get the location coordinates of the city
    location = geolocator.geocode(city)
    if location is None:
        print(f"Location not found for {city}")
        location = geolocator.geocode(city.split(" ")[0])

    # Get daily data from start to end date
    km_unit = 1000
    stations = Stations()
    stations = stations.nearby(location.latitude, location.longitude, 100 * km_unit)
    print(f"Stations found: {stations.count()}")

    stations = stations.fetch(limit=min(stations.count(), 15))
    if hourly:
        all_data = Hourly(stations, start_date, end_date)
    else:
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


def daily_row(city: str, date: str, row: Optional[pd.Series]) -> dict:
    """Prepare a row of daily data for insertion in the database."""
    if row is None:
        # Return mean values
        return {
            "city": city,
            "date": date,
            "temp_avg": 15,
            "temp_min": 15,
            "temp_max": 15,
            "precipitation": 0,
            "snow": 0,
            "wind_speed": 0,
            "wind_gust": 0,
            "pressure": 1013,
        }
    return {
        "city": city,
        "date": date,
        "temp_avg": row["tavg"] if "tavg" in row else 15,
        "temp_min": row["tmin"] if "tmin" in row else 15,
        "temp_max": row["tmax"] if "tmax" in row else 15,
        "precipitation": row["prcp"] if "prcp" in row else 0,
        "snow": row["snow"] if "snow" in row else 0,
        "wind_speed": row["wspd"] if "wspd" in row else 0,
        "wind_gust": row["wpgt"] if "wpgt" in row else 0,
        "pressure": row["pres"] if "pres" in row else 1013,
    }


def hourly_row(city: str, date: str, row: Optional[pd.Series]) -> dict:
    """Prepare a row of hourly data for insertion in the database."""
    if row is None:
        # Return mean values
        return {
            "city": city,
            "date": date,
            "temp": 15,
            "dew_point": 15,
            "rel_humidity": 0.2,
            "precipitation": 0,
            "snow": 0,
            "wind_speed": 0,
            "wind_gust": 0,
            "pressure": 1013,
        }
    return {
        "city": city,
        "date": date,
        "temp": row["temp"] if "temp" in row else 15,
        "dew_point": row["dwpt"] if "dwpt" in row else 15,
        "rel_humidity": row["rhum"] if "rhum" in row else 0.2,
        "precipitation": row["prcp"] if "prcp" in row else 0,
        "snow": row["snow"] if "snow" in row else 0,
        "wind_speed": row["wspd"] if "wspd" in row else 0,
        "wind_gust": row["wpgt"] if "wpgt" in row else 0,
        "pressure": row["pres"] if "pres" in row else 1013,
    }


def decompose(file_name: Union[str, pd.Series]) -> Tuple[str, str, str]:
    """Parse city, start date and end date from input file name."""
    file_name = file_name[0]
    splitted = file_name.split("_")
    city = splitted[0] + " " + splitted[1]
    start_date = splitted[2]
    end_date = splitted[3].rsplit("-", 4)[0]
    return city, start_date, end_date


def extract(csv_path: str, *, hourly: bool = True) -> None:
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
    for idx, row in tqdm(dataf.iterrows(), total=len(dataf)):
        city = row["city"]
        start_date = row["start_date"]
        end_date = row["end_date"]
        # Retrieve weather data
        data = retrieve_weather(geolocator, city, start_date, end_date, hourly=hourly)
        # Append data to row
        if data.shape[0] == 0:
            meteo_df = meteo_df.append(  # type: ignore
                daily_row(city, idx, None)
                if not hourly
                else hourly_row(city, idx, row),
                ignore_index=True,
            )
            print("Skipping empty data...")
            continue

        for idx, row in data.iterrows():
            # Remove columns with NaN values
            row = row.dropna()

            meteo_df = meteo_df.append(  # type: ignore
                daily_row(city, idx, row)
                if not hourly
                else hourly_row(city, idx, row),
                ignore_index=True,
            )
    # Save to csv
    dataf.to_csv("../data/meteo.csv", index=False)
    meteo_df.to_csv("../data/meteo_test_hourly.csv", index=False)


if __name__ == "__main__":
    extract("../data/test_location.csv", hourly=True)
