import os
import requests
import json
import yaml
from typing import Union, Callable
from datetime import datetime
import multiprocessing
from functools import partial
import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from meteostat import Point, Daily
from geopy.geocoders import Nominatim
from torch.utils.data import DataLoader

from lnb.data.dataset import LNBDataset


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

    # Create Point
    location_point = Point(location.latitude, location.longitude)

    # Get daily data from start to end date
    data = Daily(location_point, start_date, end_date)
    data = data.fetch()

    # print(f"Data fetched from {start_date} to {end_date}")
    # print(data.shape)

    # Add city and date columns
    data["city"] = city
    data["date"] = data.index

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


if __name__ == "__main__":
    # Arg parser
    args = argparse.ArgumentParser()
    args.add_argument("--mode", type=str, default="sequential")
    args = args.parse_args()

    # Load and process the image series dataframe

    series = pd.read_csv("..\data\image_series.csv")
    series["city"], series["start_date"], series["end_date"] = zip(
        *series.apply(decompose, axis=1)
    )

    # Create a geolocator object
    geolocator = Nominatim(user_agent="leaf-nothing-behind")

    # Create a partial function to pass geolocator and time_format parameters
    retrieve_partial = partial(
        retrieve_weather, geolocator=geolocator, time_format="%Y-%m-%d"
    )

    # Retrieve weather data
    if args.mode == "parallel":
        data = parallelize(series[:100], retrieve_partial)
    elif args.mode == "sequential":
        data = pd.DataFrame()
        for index, row in tqdm(series.iterrows()):
            if index > 2:
                break
            city = row["city"]
            start_date = row["start_date"]
            end_date = row["end_date"]
            data = data.append(retrieve_weather(geolocator, city, start_date, end_date))

    # Save the data to a CSV file
    data.to_csv("..\data\weather.csv")
