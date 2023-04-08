import requests
import json
from typing import Union

from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily

from geopy.geocoders import Nominatim
import pandas as pd


def retrieve_weather(
    geolocator: Nominatim,
    location: str,
    start_date: Union[str, datetime.date],
    end_date: Union[str, datetime.date] = None,
    time_format: str = "%Y-%m-%d",
) -> dict:
    if geolocator is None:
        geolocator = Nominatim(user_agent="leaf-nothing-behind")

    if end_date is None:
        end_date = start_date
    if type(start_date) == str:
        # start_date = start_date.strftime(time_format)
        start_date = datetime.strptime(start_date, time_format or "%Y-%m-%d")
    if type(end_date) == str:
        # end_date = end_date.strftime(time_format)
        end_date = datetime.strptime(end_date, time_format or "%Y-%m-%d")
    print(type(start_date), end_date)
    # Use the geolocator to get the location coordinates of the city
    location = geolocator.geocode(location)

    print(dir(location))

    # Print the latitude and longitude of the city
    print(f"Latitude: {location.latitude}")
    print(f"Longitude: {location.longitude}")

    # Create Point for Vancouver, BC
    location_point = Point(location.latitude, location.longitude)

    # Get daily data for 2018
    data = Daily(location_point, start_date, end_date)
    data = data.fetch()

    print(data)

    # Plot line chart including average, minimum and maximum temperature
    data.plot(y=["tavg", "tmin", "tmax"])
    plt.show()


if __name__ == "__main__":
    geolocator = Nominatim(user_agent="leaf-nothing-behind")

    retrieve_weather(geolocator, "Paris", "2020-01-01", "2020-01-05")

    series = pd.read_csv(
        "E:\Antoine\Compétitions\Leaf Nothing Behind\\assignment-2023\data\image_series.csv"
    )

    for index, row in series.iterrows():
        if index > 5:
            break
        location = row[0].split('_')[0]
        print(location)
        # retrieve_weather(geolocator, "Paris", "2020-01-01", "2020-01-05")
