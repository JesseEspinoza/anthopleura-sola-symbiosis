# load some library
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import subprocess
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
import time
import pandas as pd
from matplotlib.dates import DateFormatter
import statistics
from scipy.stats import normaltest
from scipy.stats import shapiro
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
import matplotlib.dates as mdates
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import sem
from calendar import Calendar, monthrange

c = Calendar()
import math
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from IPython.core import macro
from itertools import combinations
import itertools
import seaborn as sns
from scipy.stats import linregress
import xarray as xr
import netCDF4

# from google.colab.data_table import DataTable
# DataTable.max_columns = 40
import scikit_posthocs as sp
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import sys

# import msfunctions


def normalcy(dataset, desired_variable):
    x = pull_data(dataset, desired_variable)
    xmask = x[np.logical_not(np.isnan(x))]
    shapiro_test = stats.shapiro(xmask)
    print(shapiro_test)


def kruskal_drops(your_data, yvar, zone):

    pre_drop_date_start = "2022-07-01T12:00:00"
    pre_drop_date_end = "2022-11-25T12:00:00"
    pre_drop = your_data[
        (your_data["date_time"] > pre_drop_date_start)
        & (your_data["date_time"] <= pre_drop_date_end)
    ]

    post_drop_date_start = "2022-11-26T12:00:00"
    post_drop_date_end = "2023-04-06T12:00:00"
    post_drop = your_data[
        (your_data["date_time"] > post_drop_date_start)
        & (your_data["date_time"] <= post_drop_date_end)
    ]

    print(
        f"Testing pre {stats.shapiro(pre_drop[yvar])} vs post {stats.shapiro(post_drop[yvar])} drop normalcy"
    )

    if zone == "all":
        print(
            f"Testing all intertidal zones before vs after drop:",
            stats.kruskal(pre_drop[yvar], post_drop[yvar]),
        )
    else:
        pre_drop = pre_drop[pre_drop["intertidal_zone"] == zone]
        post_drop = post_drop[post_drop["intertidal_zone"] == zone]
        print(
            f"Testing {zone} intertidal zone before vs after drop:",
            stats.kruskal(pre_drop[yvar], post_drop[yvar]),
        )


def group_data(your_data, group_num):
    group = your_data[your_data.collection_group == group_num]
    if group.empty:
        print(f"No data found for group {group_num}")
    return group


def pull_data(your_data, desired_variable):
    """
    desried variable need to be in ''
    """
    variable = your_data[desired_variable]
    if variable.empty:
        print(f"No data found for variable {desired_variable}")
    return variable


def intertidal_graph(your_data, yvar):
    intertidal_zones = ["low", "medium", "high"]
    data = []

    for zone in intertidal_zones:
        zone_data = your_data[your_data.intertidal_zone == zone]
        zone_data = pull_data(zone_data, yvar)
        zone_data = zone_data[~np.isnan(zone_data)]

        zone_mean = np.mean(zone_data)
        zone_std = np.std(zone_data)
        zone_sem = sem(zone_data)

        data.append((zone_mean, zone_sem, zone_std))

    labels = ["Low", "Middle", "High"]
    x_pos = np.arange(len(labels))
    CTEs = [mean for mean, _, _ in data]
    SEMs = [sem for _, sem, _ in data]
    error = [std for _, _, std in data]

    fig, ax = plt.subplots()
    ax.set_facecolor("white")
    ax.bar(x_pos, CTEs, width=0.7, zorder=2, color="goldenrod")
    plt.errorbar(x_pos, CTEs, yerr=SEMs, fmt="o", color="black")
    ax.set_xlabel("Tidal Zone", fontsize=20, color="black")
    ax.xaxis.set_label_coords(0.5, -0.15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=17, color="black")
    ax.tick_params(axis="y", colors="black")

    if yvar == "num_cells_per_ug_protein":
        ax.set_ylabel("Cells/ug Animal Protein", fontsize=20, color="black")
        ax.set_title("Middle Tidal Zone has \n Highest Algal Density", fontsize=20)
    elif yvar == "ng_chlorophyll_per_ug_protein":
        ax.set_ylabel(
            "ng Chlorophyll \n  per Animal Protein", fontsize=17, color="black"
        )
        ax.set_title(
            "Middle Tidal Zone has \n Highest Chlorophyll α Production", fontsize=20
        )
    elif yvar == "ng_chlorophyll_per_hundred_cells":
        ax.set_ylabel("ng Chlorophyll \n  per 100 Cells", fontsize=17, color="black")
        # ax.set_title('Tidal Zone on Chlorophyll per Cell', fontsize = 20)

    ax.grid(axis="y", color="black", linestyle="--", linewidth=0.5)

    combinations = list(itertools.combinations(intertidal_zones, 3))

    for combination in combinations:
        zone1, zone2, zone3 = combination

        zone1_data = your_data[your_data.intertidal_zone == zone1]
        zone1_data = pull_data(zone1_data, yvar)
        zone1_data = zone1_data[~np.isnan(zone1_data)]

        zone2_data = your_data[your_data.intertidal_zone == zone2]
        zone2_data = pull_data(zone2_data, yvar)
        zone2_data = zone2_data[~np.isnan(zone2_data)]

        zone3_data = your_data[your_data.intertidal_zone == zone3]
        zone3_data = pull_data(zone3_data, yvar)
        zone3_data = zone3_data[~np.isnan(zone3_data)]

        print(f"Kruskal testing {zone1}, {zone2}, and {zone3} zones:")
        print(f"Kruskal result: {stats.kruskal(zone1_data, zone2_data, zone3_data)}")
        print()
        print(f"Kruskal testing {zone1} and {zone2} zones:")
        print(f"Kruskal result: {stats.kruskal(zone1_data, zone2_data)}")
        print()
        print(f"Kruskal testing {zone1} and {zone3} zones:")
        print(f"Kruskal result: {stats.kruskal(zone1_data, zone3_data)}")
        print()
        print(f"Kruskal testing {zone2}, and {zone3} zones:")
        print(f"Kruskal result: {stats.kruskal(zone2_data, zone3_data)}")

    # return data


def merged_plot(your_data1, xvar, yvar1, your_data2, yvar2):
    start_date = "2022-08-01T00:00:00"
    end_date = "2023-03-31T00:00:00"
    f, (ax) = plt.subplots(figsize=(12, 3.8))

    if yvar1 == "num_cells_per_ug_protein":
        label1 = "Algal Density"
    elif yvar1 == "ng_chlorophyll_per_ug_protein":
        label1 = "Chlorophyll Concentration"
    elif yvar1 == "ng_chlorophyll_per_hundred_cells":
        label1 = "ng Chlorophyll per 100 Cells"
    else:
        label1 = "Algal Density"

    if yvar2 == "temp(c)":
        label2 = "Temperature"
    elif yvar2 == "salinity(psu)" or "salinity(ppt)":
        label2 = "Salinity"
    else:
        label2 = "skrt"

    ax.scatter(
        your_data1[xvar],
        your_data1[yvar1],
        color="royalblue",
        s=10,
        zorder=3,
        label=label1,
    )
    # ax.set(xlabel = "Date", ylabel='Num Cells')
    # ax.set_title('Algal Density and Star Oddi Temp', fontsize =25)
    ax.tick_params(
        axis="x", labelsize=11, rotation=15, labelbottom=True, direction="out", pad=10
    )
    ax.tick_params(axis="y", labelsize=12)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
    # ax.xaxis.set_major_formatter(DateFormatter("%m-%d-%y"))
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d-%Y"))
    ax.set_xlim(start_date, end_date)
    # ax.grid(False)

    if yvar1 == "num_cells_per_ug_protein":
        ax.set_ylabel("Cells/ug Animal Protein", fontsize=15)
        # ax.set_title('Merged algal and Fort Point salinity data overlayed', fontsize=20)

    if yvar1 == "ng_chlorophyll_per_ug_protein":
        ax.set_ylabel("ng Chlorophyll per Animal Protein", fontsize=15)
        ax.set_title("Chlorophyll α with Fort Point Salinity Overlayed", fontsize=20)

    if yvar1 == "ng_chlorophyll_per_hundred_cells":
        ax.set_ylabel("ng Chlorophyll per 100 Cells", fontsize=15)

    ax2 = ax.twinx()  # to plot a second y axis
    ax2.scatter(your_data2[xvar], your_data2[yvar2], color="orange", label=label2, s=15)
    # ax2.set(ylabel='Rainfall (mm)')
    ax2.tick_params(axis="y", labelsize=12)
    ax2.yaxis.label.set_size(20)
    ax2.set_ylim(5, 35)
    ax2.grid(False)

    if yvar2 == "temp(c)":
        plt.ylabel("Temp (c)", fontsize=15, rotation=270, va="bottom")

    if yvar2 == "salinity(psu)" or "salinity(ppt)":
        plt.ylabel("Salinity", fontsize=15, rotation=270, va="bottom")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="best", fontsize=9)


def kruskal_drops(your_data, yvar, zone):

    pre_drop_date_start = "2022-07-01T12:00:00"
    pre_drop_date_end = "2022-11-10T12:00:00"
    pre_drop = your_data[
        (your_data["date_time"] > pre_drop_date_start)
        & (your_data["date_time"] <= pre_drop_date_end)
    ]

    post_drop_date_start = "2022-11-11T12:00:00"
    post_drop_date_end = "2023-04-06T12:00:00"
    post_drop = your_data[
        (your_data["date_time"] > post_drop_date_start)
        & (your_data["date_time"] <= post_drop_date_end)
    ]

    print(
        f"Testing pre {stats.shapiro(pre_drop[yvar])} vs post {stats.shapiro(post_drop[yvar])} drop normalcy"
    )

    if zone == "all":
        print(
            f"Testing all intertidal zones before vs after drop:",
            stats.kruskal(pre_drop[yvar], post_drop[yvar]),
        )
        # display(post_drop)
    else:
        pre_drop = pre_drop[pre_drop["intertidal_zone"] == zone]
        post_drop = post_drop[post_drop["intertidal_zone"] == zone]
        print(
            f"Testing {zone} intertidal zone before vs after drop:",
            stats.kruskal(pre_drop[yvar], post_drop[yvar]),
        )
        # display(post_drop)


def get_y_label(yvar):
    if yvar == "num_cells_per_ug_protein":
        return "Cells/ug Animal Protein"
    elif yvar == "ng_chlorophyll_per_ug_protein":
        return "ng Chlorophyll per Animal Protein"
    elif yvar == "ng_chlorophyll_per_hundred_cells":
        return "ng Chlorophyll per 100 Cells"


def get_title(yvar):
    if yvar == "num_cells_per_ug_protein":
        return "Tidal Zone on Algal Density"
    elif yvar == "ng_chlorophyll_per_ug_protein":
        return "Tidal Zone on Chlorophyll α Production"
    elif yvar == "ng_chlorophyll_per_hundred_cells":
        return "Tidal Zone on Chlorophyll per Cell"


def process_abiotic_data(abiotic_data_path, start_date, end_date):
    """
    Processes multiple abiotic datasets, filters by date range, extracts relevant columns, and calculates 7-day and daily averages.

    Parameters:
    abiotic_data_path (str): The directory path where the datasets are stored.
    start_date (str): The start date in ISO format (e.g., '2022-08-01T00:00:00').
    end_date (str): The end date in ISO format (e.g., '2023-03-31T00:00:00').

    Returns:
    dict: A dictionary containing processed data from each dataset.
    dict: A dictionary containing 7-day averaged data for each dataset.
    dict: A dictionary containing daily averaged data for each dataset.
    """
    results = {}
    seven_day_averages = {}
    daily_averages = {}

    # Helper function to load, filter, and clean data
    def load_and_filter(file_name, date_col, additional_filters=None):
        data = pd.read_csv(abiotic_data_path + file_name)

        # Remove any 'Unnamed: _' columns
        data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

        # Clean column names by replacing spaces, parentheses, etc.
        data.columns = [
            col.replace("(", "_").replace(")", "").replace(" ", "_")
            for col in data.columns
        ]

        data["date_time"] = pd.to_datetime(data[date_col])
        data = data[(data["date_time"] > start_date) & (data["date_time"] <= end_date)]
        if additional_filters:
            for condition in additional_filters:
                data = data[condition(data)]

        # Remove any columns with 'date', 'time', or 'datetime'
        data = data[
            [col for col in data.columns if col not in ["date", "time", "datetime"]]
        ]

        # Reorder columns to place 'date_time' first
        cols = ["date_time"] + [col for col in data.columns if col != "date_time"]
        data = data[cols]

        # Reset index
        data.reset_index(drop=True, inplace=True)

        return data

    def calculate_seven_day_average(data, base_name):
        # Use the `start_date` from the outer scope here
        averaged_data = data.copy()
        averaged_data.set_index("date_time", inplace=True)

        # Ensure the index is sorted by date_time
        averaged_data.sort_index(inplace=True)

        # Calculate 7-day rolling averages for all numeric columns
        rolling_avg = averaged_data.rolling("7D").mean()

        # Resample to keep one row every 7 days (align to start_date)
        seven_day_avg = rolling_avg.resample("7D", origin=start_date).mean()

        # Reset index to make date_time a column again
        seven_day_avg.reset_index(inplace=True)

        # Rename columns to indicate 7-day average
        seven_day_avg.columns = ["date_time"] + [
            f"{col}_seven_day_average"
            for col in seven_day_avg.columns
            if col != "date_time"
        ]

        return seven_day_avg

    def calculate_daily_average(data):
        # Set 'date_time' as index for daily resampling
        averaged_data = data.copy()
        averaged_data.set_index("date_time", inplace=True)

        # Ensure the index is sorted by date_time
        averaged_data.sort_index(inplace=True)

        # Calculate daily averages for all numeric columns
        daily_avg = averaged_data.resample("D").mean()

        # Reset index to make date_time a column again
        daily_avg.reset_index(inplace=True)

        # Rename columns to indicate daily average
        daily_avg.columns = ["date_time"] + [
            f"{col}_daily_average" for col in daily_avg.columns if col != "date_time"
        ]

        return daily_avg

    # Process datasets
    datasets = {
        "hobo": ("cleaned_rockaway_hobo_logger.csv", "datetime", None),
        "star_oddi": (
            "cleaned_rockaway_star_oddi_salinity_data.csv",
            "datetime",
            [lambda df: df["salinity_ppt"] > 5],
        ),
        "fort_point_daily": ("fort_point_salinity_daily.csv", "datetime", None),
        "fort_point_hourly": ("fort_point_salinity_hourly.csv", "datetime", None),
        "precipitation": ("precipitation.csv", "datetime", None),
    }

    for name, (file_name, date_col, filters) in datasets.items():
        data = load_and_filter(file_name, date_col, filters)
        results[name] = data

        # Calculate and store 7-day averages
        averaged_data_seven_day = calculate_seven_day_average(data, name)
        seven_day_averages[f"{name}_seven_day_average"] = averaged_data_seven_day

        # Calculate and store daily averages
        averaged_data_daily = calculate_daily_average(data)
        daily_averages[f"{name}_daily_average"] = averaged_data_daily

    # Print the names of resulting DataFrames
    print("Available datasets:")
    for name in results.keys():
        print(f"- {name}")

    print("\nAvailable 7-day average datasets:")
    for name in seven_day_averages.keys():
        print(f"- {name}")

    print("\nAvailable daily average datasets:")
    for name in daily_averages.keys():
        print(f"- {name}")

    # Now, return all three dictionaries
    return results, seven_day_averages, daily_averages


site_dict = {
    "ap": {
        "name": "Aramai Point",
        "lon": -122.500725,
        "lat": 37.607919,
        "Location": "Pacifica, San Mateo County, CA",
        "Site Description": "Aramai Point was originally named Rockaway Beach but has since been renamed to honor the Native American tribe that lived in the area for thousands of years. This public beach has hiking trails leading to prominent headlands that overshadow a rocky intertidal habitat below. The intertidal zone is very accessible, beginning a few hundred yards from the parking lot. Anemones, sea stars, tunicates, barnacles, and many other inverts are found throughout this site. Aramai Point is heavily trafficed by surfers, fishermen, and families, especially on the weekends.",
        "Research Conducted": "Identifying the impact intertidal positioning has on the symbiotic relationship between Anthopleura sola and their algal photosynthesizers.",
        "Student Researchers": "Jesse Espinoza",
    }
}
stns = list(site_dict.keys())

lats = []
lons = []
for stn in stns:
    lats.append(
        site_dict[stn]["lat"]
    )  # Filling up the latitude list by looping through our list of stations
    lons.append(site_dict[stn]["lon"])  # Same but for longitude


def site_sample_area(site, data, start_time, end_time):
    """
    This function will take your sites coordinates and make the sample area while
    scooting a bit away from land. Then it will take those coordinates and apply
    them to a pointer (in this case sst), which has the variable selected and
    our time range in too.

    To do other pointers, this code can be expanded, or duplicated but for other
    variables

    site = dictionary site key, an abbreviation in quotation marks. Ex: 'ap'
    """
    site_lon = site_dict[site]["lon"]
    site_lon_min = site_lon - 2 * 0.04
    site_lon_max = site_lon - 0.04

    site_lat = site_dict[site]["lat"]
    site_lat_min = site_lat - 2 * 0.04
    site_lat_max = site_lat - 0.04

    print(
        "lon:", site_lon_min, site_lon_max
    )  # uncomment if you want to see the coordinates per your site
    print("lat:", site_lat_min, site_lat_max)

    your_sst = data["analysed_sst"].sel(
        latitude=slice(site_lat_min, site_lat_max),
        longitude=slice(site_lon_min, site_lon_max),
        time=slice(start_time, end_time),
    )

    return your_sst
