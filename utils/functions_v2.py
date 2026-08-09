"""
Various statistical and data processing functions for analyzing intertidal data.
"""

# load some library
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
import pandas as pd
from matplotlib.dates import DateFormatter

from calendar import Calendar

import matplotlib.dates as mdates
from scipy import stats
from scipy.stats import sem

c = Calendar()
import itertools

import xarray as xr

from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Gamma
from statsmodels.genmod.families.links import Log
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests


def normalcy(dataset: xr.Dataset, desired_variable: str):
    """
    Test for normalcy of data using Shapiro-Wilk test.

    Parameters:
    -----------
    dataset (xr.Dataset):
        The dataset containing the variable to be tested.
    desired_variable (str):
        The name of the variable to be tested for normalcy.

    Returns:
    --------
    None:
        Prints the result of the Shapiro-Wilk test.
    """
    x = pull_data(dataset, desired_variable)
    xmask = x[np.logical_not(np.isnan(x))]
    shapiro_test = stats.shapiro(xmask)
    print(shapiro_test)


def kruskal_drops(your_data: pd.DataFrame, yvar: str, zone: str):
    """
    Perform Kruskal-Wallis test to compare pre- and post-drop data.

    Parameters:
    -----------
    your_data (pd.DataFrame):
        The dataset containing the variable to be tested.
    yvar (str):
        The name of the variable to be tested.
    zone (str):
        The intertidal zone to filter by ('low', 'medium', 'high', or 'all').

    Returns:
    --------
    None:
        Prints the result of the Kruskal-Wallis test.
    """
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


def group_data(your_data: pd.DataFrame, group_num: int):
    """
    Group data by collection group number.

    Parameters:
    -----------
    your_data (pd.DataFrame):
        The dataset containing the collection group information.
    group_num (int):
        The collection group number to filter by.
    """
    group = your_data[your_data.collection_group == group_num]
    if group.empty:
        print(f"No data found for group {group_num}")
    return group


def pull_data(your_data: pd.DataFrame, desired_variable: str):
    """
    Pull and isolate a specific variable from the dataset.

    Parameters:
    -----------
    your_data (pd.DataFrame):
        The dataset containing the variable to be pulled.
    desired_variable (str):
        The name of the variable to be pulled.
    """
    variable = your_data[desired_variable]
    if variable.empty:
        print(f"No data found for variable {desired_variable}")
    return variable


def intertidal_graph(your_data: pd.DataFrame, yvar: str):
    """
    Generate a bar graph comparing means of a specified variable across intertidal zones.
    Parameters:
    -----------
    your_data (pd.DataFrame):
        The dataset containing the intertidal zone and variable data.
    yvar (str):
        The name of the variable to be analyzed and plotted.
    Returns:
    --------
    None:
        Displays a bar graph and prints Kruskal-Wallis test results.
    """
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


def kruskal_drops(your_data: pd.DataFrame, yvar: str, zone: str):
    """
    Perform Kruskal-Wallis test to compare pre- and post-drop data.

    Parameters:
    -----------
    your_data (pd.DataFrame):
        The dataset containing the variable to be tested.
    yvar (str):
        The name of the variable to be tested.
    zone (str):
        The intertidal zone to filter by ('low', 'medium', 'high', or 'all').

    Returns:
    --------
    None:
        Prints the result of the Kruskal-Wallis test.
    """

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


########################################################################
# Repeated-measures alternatives to the Kruskal-Wallis tests above.
#
# The tentacle samples in this study are NOT independent observations:
# ~35-38 tagged anemones were sampled repeatedly (bimonthly) across the
# season, so the samples from a given individual are correlated with
# each other over time. Kruskal-Wallis (and the Dunn's post-hoc test)
# assume independent observations, so applying them directly to the
# pooled, repeated-sample dataset overstates the effective sample size
# and can produce overly-optimistic p-values.
#
# The functions below use Generalized Estimating Equations (GEE) with
# a Gamma family (log link), which is appropriate for the right-skewed,
# strictly-positive algal density / chlorophyll data, and clusters
# observations by individual anemone (specimen_id) to account for the
# repeated-measures structure. This directly replaces kruskal_drops()
# (pre- vs. post-drop comparison) and the posthoc_dunn() zone
# comparison used in the notebook.
########################################################################


def prep_cluster_ids(df: pd.DataFrame, id_col: str = "specimen_id") -> pd.DataFrame:
    """
    Build a clustering column for GEE from the specimen ID column.

    Rows with a known specimen_id are clustered by that individual, so
    repeated samples from the same anemone are modeled as correlated.
    Rows with a missing specimen_id (no individual ID was recorded)
    cannot be linked to any other sample, so each is assigned its own
    unique singleton cluster; this treats them as independent, which is
    the most conservative assumption possible for an unlinkable sample.

    Parameters:
    -----------
    df (pd.DataFrame):
        The dataset containing the individual ID column.
    id_col (str):
        The name of the column identifying the sampled individual.

    Returns:
    --------
    pd.DataFrame:
        A copy of df with an added 'cluster_id' string column for use
        as the `groups` argument in GEE.
    """
    df = df.copy()
    cluster = df[id_col].astype("object")
    missing = cluster.isnull()
    n_missing = int(missing.sum())
    if n_missing:
        solo_ids = [f"solo_{i}" for i in range(n_missing)]
        cluster.loc[missing] = solo_ids
    df["cluster_id"] = cluster.astype(str)
    return df


def gee_period_test(
    your_data: pd.DataFrame,
    yvar: str,
    pre_drop_date_end: str = "2022-11-25T12:00:00",
    zone: str = "all",
    id_col: str = "specimen_id",
    verbose: bool = True,
):
    """
    Test for a pre- vs. post-drop difference using a GEE model with a
    Gamma family (log link), clustering repeated samples by individual
    anemone. Replaces kruskal_drops() for the pre/post comparison.

    Parameters:
    -----------
    your_data (pd.DataFrame):
        The dataset containing 'date_time', yvar, 'intertidal_zone',
        and the individual ID column.
    yvar (str):
        The name of the dependent variable to be tested
        (e.g. 'num_cells_per_ug_protein').
    pre_drop_date_end (str):
        Cutoff date-time; samples on/before this are 'pre', samples
        after are 'post'.
    zone (str):
        'all' to pool all intertidal zones, or 'low'/'middle'/'high'
        to subset to one zone before fitting.
    id_col (str):
        Column identifying the sampled individual.
    verbose (bool):
        If True, print the model summary.

    Returns:
    --------
    GEEResultsWrapper:
        The fitted statsmodels GEE results object.
    """
    data = your_data.dropna(subset=[yvar]).copy()
    if zone != "all":
        data = data[data["intertidal_zone"] == zone]
    data = prep_cluster_ids(data, id_col=id_col)

    data["period"] = np.where(
        pd.to_datetime(data["date_time"]) <= pre_drop_date_end, "pre", "post"
    )

    model = GEE.from_formula(
        f'{yvar} ~ C(period, Treatment(reference="pre"))',
        groups="cluster_id",
        data=data,
        family=Gamma(link=Log()),
        cov_struct=Exchangeable(),
    )
    res = model.fit()

    if verbose:
        zone_label = "all intertidal zones" if zone == "all" else f"the {zone} zone"
        print(
            f"GEE test (Gamma, clustered by {id_col}) for {zone_label}, pre vs. post drop:"
        )
        print(res.summary())

    return res


def gee_zone_test(
    your_data: pd.DataFrame,
    yvar: str,
    zones: tuple = ("low", "middle", "high"),
    id_col: str = "specimen_id",
    verbose: bool = True,
):
    """
    Test for an overall difference in yvar across intertidal zones
    using a GEE model with a Gamma family (log link), clustering
    repeated samples by individual anemone.

    Parameters:
    -----------
    your_data (pd.DataFrame):
        The dataset containing yvar, 'intertidal_zone', and the
        individual ID column.
    yvar (str):
        The name of the dependent variable to be tested.
    zones (tuple):
        The zone levels to include, in the order used for the model's
        reference category (first element is the reference).
    id_col (str):
        Column identifying the sampled individual.
    verbose (bool):
        If True, print the model summary.

    Returns:
    --------
    GEEResultsWrapper:
        The fitted statsmodels GEE results object.
    """
    data = your_data.dropna(subset=[yvar, "intertidal_zone"]).copy()
    data = prep_cluster_ids(data, id_col=id_col)
    data["intertidal_zone"] = pd.Categorical(
        data["intertidal_zone"], categories=list(zones)
    )

    model = GEE.from_formula(
        f'{yvar} ~ C(intertidal_zone, Treatment(reference="{zones[0]}"))',
        groups="cluster_id",
        data=data,
        family=Gamma(link=Log()),
        cov_struct=Exchangeable(),
    )
    res = model.fit()

    if verbose:
        print(
            f"GEE test (Gamma, clustered by {id_col}) across intertidal zones {zones}:"
        )
        print(res.summary())

    return res


def gee_pairwise_zone_contrasts(
    your_data: pd.DataFrame,
    yvar: str,
    zones: tuple = ("low", "middle", "high"),
    id_col: str = "specimen_id",
    p_adjust: str = "holm",
) -> pd.DataFrame:
    """
    Pairwise comparisons between all intertidal-zone levels using a
    Gamma-family GEE model clustered by individual anemone, with the
    reference category releveled to obtain every pairwise contrast.
    p-values are adjusted for multiple comparisons (default: Holm).
    This replaces the posthoc_dunn() zone comparison used in the
    notebook, while accounting for repeated sampling of individuals.

    Parameters:
    -----------
    your_data (pd.DataFrame):
        The dataset containing yvar, 'intertidal_zone', and the
        individual ID column.
    yvar (str):
        The name of the dependent variable to be tested.
    zones (tuple):
        The zone levels to compare.
    id_col (str):
        Column identifying the sampled individual.
    p_adjust (str):
        Multiple comparison correction method passed to
        statsmodels.stats.multitest.multipletests (default 'holm').

    Returns:
    --------
    pd.DataFrame:
        One row per pairwise zone comparison, with the GEE log-scale
        coefficient, raw p-value, and adjusted p-value.
    """
    data = your_data.dropna(subset=[yvar, "intertidal_zone"]).copy()
    data = prep_cluster_ids(data, id_col=id_col)
    zones = list(zones)

    seen = set()
    rows = []
    for ref in zones[:-1]:
        cats = [ref] + [z for z in zones if z != ref]
        data["intertidal_zone"] = pd.Categorical(
            data["intertidal_zone"], categories=cats
        )

        model = GEE.from_formula(
            f'{yvar} ~ C(intertidal_zone, Treatment(reference="{ref}"))',
            groups="cluster_id",
            data=data,
            family=Gamma(link=Log()),
            cov_struct=Exchangeable(),
        )
        res = model.fit()

        for name, coef, p in zip(
            res.params.index, res.params.values, res.pvalues.values
        ):
            if name == "Intercept":
                continue
            other = name.split("[T.")[-1].rstrip("]")
            pair = tuple(sorted([ref, other]))
            if pair not in seen:
                seen.add(pair)
                rows.append(
                    {"zone_1": pair[0], "zone_2": pair[1], "log_coef": coef, "p_raw": p}
                )

    result = pd.DataFrame(rows)
    _, adj_p, _, _ = multipletests(result["p_raw"], method=p_adjust)
    result[f"p_{p_adjust}"] = adj_p
    return result


def process_abiotic_data(abiotic_data_path: str, start_date: str, end_date: str):
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
    def load_and_filter(file_name: str, date_col: str, additional_filters: list = None):
        """
        Loads a CSV file, filters by date range and additional conditions, and cleans column names.

        Parameters:
        file_name (str): The name of the CSV file to load.
        date_col (str): The name of the column containing date-time information.
        additional_filters (list, optional): A list of additional filtering conditions (functions) to apply.

        Returns:
        pd.DataFrame: The cleaned and filtered DataFrame.
        """
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

    def calculate_seven_day_average(
        data: pd.DataFrame,
    ):
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

    def calculate_daily_average(data: pd.DataFrame):
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
        averaged_data_seven_day = calculate_seven_day_average(data)
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


def site_sample_area(site: str, data: xr.Dataset, start_time: str, end_time: str):
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
