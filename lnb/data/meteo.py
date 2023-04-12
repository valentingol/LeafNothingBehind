from typing import Union, Callable
from datetime import datetime
import multiprocessing
import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm

from meteostat import Daily, Stations
from geopy.geocoders import Nominatim


def retrieve_weather(
    geolocator: Nominatim,
    city: str,
    start_date: Union[str, datetime.date],
    end_date: Union[str, datetime.date] = None,
    time_format: str = "%Y-%m-%d",
) -> dict:
    if geolocator is None:
        geolocator = Nominatim(user_agent="leaf-nothing-behind")

    if end_date is None:
        end_date = start_date
    if type(start_date) == str:
        start_date = datetime.strptime(start_date, time_format or "%Y-%m-%d")
    if type(end_date) == str:
        end_date = datetime.strptime(end_date, time_format or "%Y-%m-%d")

    # Use the geolocator to get the location coordinates of the city
    location = geolocator.geocode(city)

    # Print the latitude and longitude of the city
    # print(f"Location: ({city}) {location.address}")
    # print(f"Latitude: {location.latitude}")
    # print(f"Longitude: {location.longitude}")

    # Get daily data from start to end date
    km = 1000
    stations = Stations()
    stations = stations.nearby(location.latitude, location.longitude, 100*km)
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


def parallelize(series: pd.Series, func: Callable):
    # Define the number of worker processes to use
    num_processes = multiprocessing.cpu_count()

    print(f"Using {num_processes} processes...")

    # Split the series dataframe into chunks based on the number of worker processes
    series_chunks = np.array_split(series, num_processes)

    # Create a pool of worker processes and map the retrieve_partial function to the series chunks
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(retrieve_partial, series_chunks)

    # Concatenate the results into a single dataframe
    data = pd.concat(results, axis=0)

    return data


def decompose(x):
    x = x[0]
    splitted = x.split("_")
    city = splitted[0] + " " + splitted[1]
    start_date = splitted[2]
    end_date = splitted[3].rsplit("-", 4)[0]
    return city, start_date, end_date


def extract(csv_path: str):
    df = pd.read_csv(csv_path)
    df["city"], df["start_date"], df["end_date"] = zip(*df.apply(decompose, axis=1))
    # Remove all columns except city, start_date, end_date
    df = df[["city", "start_date", "end_date"]]
    # Remove duplicates
    df = df.drop_duplicates()

    meteo_df = pd.DataFrame()

    # Create a geolocator object
    geolocator = Nominatim(user_agent="leaf-nothing-behind")
    for index, row in tqdm(df.iterrows(), total=len(df)):
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
                ignore_index=True
            )
            print("Skipping empty data...")
            continue

        for idx, rw in data.iterrows():
            # Replace NaN values with 0
            rw = rw.fillna(0)
            meteo_df = meteo_df.append(
                {
                    "city": city,
                    "date": idx,
                    "temp_avg": rw["tavg"],
                    "temp_min": rw["tmin"],
                    "temp_max": rw["tmax"],
                    "precipitation": rw["prcp"],
                    "snow": rw["snow"],
                    "wind": rw["wspd"],
                    "wind_dir": rw["wdir"],
                    "wind_speed": rw["wspd"],
                    "wind_gust": rw["wpgt"],
                    "pressure": rw["pres"],
                    "sun": rw["tsun"],
                },
                ignore_index=True,
            )

    # Save to csv
    df.to_csv("..\data\meteo.csv", index=False)
    meteo_df.to_csv("..\data\meteo_full.csv", index=False)

if __name__ == "__main__":
    # Arg parser
    args = argparse.ArgumentParser()
    args.add_argument("--mode", type=str, default="sequential")
    args = args.parse_args()

    # Load and process the image series dataframe

    series = pd.read_csv("..\data\image_series.csv")
    weather = pd.read_csv("..\data\meteo_full.csv")


    # Get weather data
    location = decompose(series.iloc[45])[0]
    weather = weather[weather['city'] == location]

    # Reset index
    weather = weather.reset_index(drop=True)
    # Limit weather data to the 5 last entries
    weather = weather.iloc[-5:]

    print(weather)

    # Convert weather to numpy array and remove city and date columns
    weather = weather.drop(columns=['city', 'date'])
    weather = weather.to_numpy()

    print(weather.shape)

    # Convert shape from (5, 11) to (55,)
    weather = weather.reshape(-1)
    weather = np.expand_dims(weather, axis=(1, 2))

    print(weather.shape)

    # extract("..\data\image_series.csv")
    # series["city"], series["start_date"], series["end_date"] = zip(
    #     *series.apply(decompose, axis=1)
    # )

    # # Create a geolocator object
    # geolocator = Nominatim(user_agent="leaf-nothing-behind")

    # # Create a partial function to pass geolocator and time_format parameters
    # retrieve_partial = partial(
    #     retrieve_weather, geolocator=geolocator, time_format="%Y-%m-%d"
    # )

    # # Retrieve weather data
    # if args.mode == "parallel":
    #     data = parallelize(series[:100], retrieve_partial)
    # elif args.mode == "sequential":
    #     data = pd.DataFrame()
    #     for index, row in tqdm(series.iterrows()):
    #         if index > 2:
    #             break
    #         city = row["city"]
    #         start_date = row["start_date"]
    #         end_date = row["end_date"]
    #         data = data.append(retrieve_weather(geolocator, city, start_date, end_date))

    # # Save the data to a CSV file
    # data.to_csv("..\data\weather.csv")
