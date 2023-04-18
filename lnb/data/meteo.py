"""Weather data extraction."""
import os
import requests
from datetime import datetime
from typing import Tuple, Union

import pandas as pd
from geopy.geocoders import Nominatim
from meteostat import Daily, Stations, Hourly
from tqdm import tqdm


def request_patch(slf, *args, **kwargs):
    print("\nTime out set to 5 seconds")
    timeout = kwargs.pop("timeout", 5)
    return slf.request_orig(*args, **kwargs, timeout=timeout)


setattr(requests.sessions.Session, "request_orig", requests.sessions.Session.request)
requests.sessions.Session.request = request_patch


def retrieve_weather(
    geolocator: Nominatim,
    city: str,
    start_date: Union[str, datetime.date],  # type: ignore
    end_date: Union[str, datetime.date] = None,  # type: ignore
    time_format: str = "%Y-%m-%d",
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


def daily_row(city: str, date: str, row: pd.Series) -> dict:
    return {
        "city": city,
        "date": str(date),
        "temp_avg": row["tavg"] if "tavg" in row else 15,
        "temp_min": row["tmin"] if "tmin" in row else 15,
        "temp_max": row["tmax"] if "tmax" in row else 15,
        "precipitation": row["prcp"] if "prcp" in row else 0,
        "snow": row["snow"] if "snow" in row else 0,
        "wind_speed": row["wspd"] if "wspd" in row else 0,
        "wind_gust": row["wpgt"] if "wpgt" in row else 0,
        "pressure": row["pres"] if "pres" in row else 1013,
    }


def hourly_row(city: str, date: str, row: pd.Series) -> dict:
    return {
        "city": city,
        "date": str(date),
        "temp": row["temp"] if "temp" in row else 15,
        "dew_point": row["dwpt"] if "dwpt" in row else 15,
        "rel_humidity": row["rhum"] if "rhum" in row else 0.2,
        "precipitation": row["prcp"] if "prcp" in row else 0,
        "snow": row["snow"] if "snow" in row else 0,
        "wind_speed": row["wspd"] if "wspd" in row else 0,
        "wind_gust": row["wpgt"] if "wpgt" in row else 0,
        "pressure": row["pres"] if "pres" in row else 1013,
    }


def decompose(file_name: str) -> Tuple[str, str, str]:
    """Parse city, start date and end date from input file name."""
    file_name = file_name[0]
    splitted = file_name.split("_")
    city = splitted[0] + " " + splitted[1]
    start_date = splitted[2]
    end_date = splitted[3].rsplit("-", 4)[0]
    return city, start_date, end_date


def extract(
    csv: Union[str, pd.Series, pd.DataFrame],
    output_folder: str = "../data/",
    hourly: bool = False,
) -> None:
    """Extract weather data from csv file and save to csv."""
    if type(csv) == str:
        dataf = pd.read_csv(csv)
    else:
        dataf = csv
    dataf["city"], dataf["start_date"], dataf["end_date"] = zip(
        *dataf.apply(decompose, axis=1),
    )
    # Remove all columns except city, start_date, end_date
    dataf = dataf[["city", "start_date", "end_date"]]
    # Remove duplicates
    dataf = dataf.drop_duplicates()

    if os.path.exists(os.path.join(output_folder, "meteo_dates.csv")):
        dates = pd.read_csv(os.path.join(output_folder, "meteo_dates.csv"))

        # Remove dataf rows matching city, start_date and end_date in dates dataframe
        print(dates)
        print(dataf.shape)
        dataf = dataf[
            ~dataf.apply(
                lambda row: (
                    (dates["city"] == row["city"])
                    & (dates["start_date"] == row["start_date"])
                    & (dates["end_date"] == row["end_date"])
                ).any(),
                axis=1,
            )
        ]

        dates = dates.append(dataf)
        dates.to_csv(os.path.join(output_folder, "meteo_dates.csv"), index=False)
    else:
        dataf.to_csv(os.path.join(output_folder, "meteo_dates.csv"), index=False)


    print(dataf.shape)

    if os.path.exists(os.path.join(output_folder, "meteo_features.csv")):
        meteo_df = pd.read_csv(os.path.join(output_folder, "meteo_features.csv"))
    else:
        meteo_df = pd.DataFrame()

    # Create a geolocator object
    geolocator = Nominatim(user_agent="leaf-nothing-behind")
    for _, row in tqdm(dataf.iterrows(), total=len(dataf)):
        city = row["city"]
        start_date = row["start_date"]
        end_date = row["end_date"]
        # Retrieve weather data
        data = retrieve_weather(geolocator, city, start_date, end_date, hourly=hourly)
        # Append data to row
        if data.shape[0] == 0:
            meteo_df = meteo_df.append(
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

            meteo_df = meteo_df.append(
                daily_row(city, idx, row) if not hourly else hourly_row(city, idx, row),
                ignore_index=True,
            )

    # Remove duplicates
    meteo_df = meteo_df.drop_duplicates()

    # Save to csv
    meteo_df.to_csv(os.path.join(output_folder, "meteo_features.csv"), index=False)

    return meteo_df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe."""
    # Normalize temp_avg,temp_min,temp_max from -50°C to 50°C to -1 to 1
    df["temp_avg"] = df["temp_avg"] / 50
    df["temp_min"] = df["temp_min"] / 50
    df["temp_max"] = df["temp_max"] / 50
    # Normalize precipitation from 0 to 10mm to 0 to 1
    df["precipitation"] = df["precipitation"] / 10
    # Normalize snow from 0 to 50mm to 0 to 1
    df["snow"] = df["snow"] / 50
    # Normalize wind_speed from 0 to 100km/h to 0 to 1
    df["wind_speed"] = df["wind_speed"] / 100
    # Normalize wind_gust from 0 to 100km/h to 0 to 1
    df["wind_gust"] = df["wind_gust"] / 100
    # Normalize pressure from 900 to 1100hPa to 0 to 1
    df["pressure"] = (df["pressure"] - 900) / 200

    return df


if __name__ == "__main__":
    extract("../data/test_location.csv", output_folder="../data/", hourly=False)

    # Load and process the image series dataframe
    # series = pd.read_csv("../data/image_series.csv")
    # weather = pd.read_csv("../data/meteo_full.csv")
    # # Get weather data
    # location = decompose(series.iloc[45])[0]
    # weather = weather[weather["city"] == location]
    # # Reset index
    # weather = weather.reset_index(drop=True)
    # # Limit weather data to the 5 last entries
    # weather = weather.iloc[-5:]
    # print(weather)
    # # Convert weather to numpy array and remove city and date columns
    # weather = weather.drop(columns=["city", "date"])
    # weather = weather.to_numpy()
    # print(weather.shape)
    # # Convert shape from (5, 11) to (55,)
    # weather = weather.reshape(-1)
    # weather = np.expand_dims(weather, axis=(1, 2))
    # print(weather.shape)
