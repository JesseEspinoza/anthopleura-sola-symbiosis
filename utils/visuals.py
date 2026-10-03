"""
Visualization utilities for data analysis.

This module provides plotting functions for analyzing and visualizing
symbiont density, chlorophyll a concentration, and environmental data across
intertidal zones and time periods.
"""

# load some library
import os
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union, Tuple
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from PIL import Image, ImageChops, ImageOps
from scipy import stats
from matplotlib.transforms import ScaledTranslation

PathLike = Union[str, Path]

import pandas as pd
from matplotlib.dates import DateFormatter
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
from calendar import Calendar

import matplotlib.dates as mdates
from scipy.stats import sem

c = Calendar()
import sys

from utils.functions import group_data, pull_data


def batch_bar(
    your_data: pd.DataFrame,
    yvar: str,
    zone: Optional[str],
    bar_color: str = "mediumseagreen",
    save_path: Optional[PathLike] = None,
) -> None:
    """
    Create a bar plot showing group statistics over time with error bars.

    This function generates a time-series bar plot displaying mean values
    with standard error bars for a specified variable, optionally filtered
    by intertidal zone.

    Parameters
    ----------
    your_data : pd.DataFrame
        Input dataframe containing the response variable and
        'intertidal_zone' column.
    yvar : str
        Column name in `your_data` to analyze. Expected values include:
        - 'num_cells_per_ug_protein'
        - 'ng_chlorophyll_per_ug_protein'
    zone : str or None
        Intertidal zone to filter by (e.g., 'low', 'mid', 'high').
        If None, data from all zones are included.
    bar_color : str, default "mediumseagreen"
        Color for the bar plots.
    save_path : str or Path, optional
        File path to save the plot. If None, the plot is not saved.

    Returns
    -------
    None
        Displays the plot using matplotlib.

    Notes
    -----
    This function assumes the existence of helper functions:
    - `group_data(data: pd.DataFrame, batch_size: int) -> pd.DataFrame`
    - `pull_data(data: pd.DataFrame, yvar: str) -> np.ndarray`
    """
    labels = [
        "Aug 27, 2022",
        "Sept 6, 2022",
        "Sept 23, 2022",
        "Oct 10, 2022",
        "Oct 27, 2022",
        "Nov 08, 2022",
        "Nov 23, 2022",
        "Dec 6, 2022",
        "Jan 06, 2023",
        "Jan 23, 2023",
        "Feb 6, 2023",
        "Feb 18, 2023",
        "Mar 17, 2023",
    ]
    batch_sizes = range(4, 17)

    batches = []
    means = []
    stds = []
    sems = []

    selected_data = (
        your_data[your_data["intertidal_zone"] == zone] if zone else your_data
    )

    for size in batch_sizes:
        batch = group_data(selected_data, size)
        batch = pull_data(batch, yvar)
        batches.append(batch)
        means.append(np.mean(batch))
        stds.append(np.std(batch))
        sems.append(sem(batch))

    x_pos = np.arange(len(labels))
    CTEs = means
    SEMs = sems
    error = stds

    fig, ax = plt.subplots(figsize=(27, 10))
    ax.set_facecolor("white")
    ax.bar(
        x_pos, CTEs, width=0.75, color=bar_color, zorder=2
    )  # Dynamically set bar color
    plt.errorbar(x_pos, CTEs, yerr=SEMs, fmt="o", color="black")

    ax.set_xlabel("Date", fontsize=20, color="black")
    ax.xaxis.set_label_coords(0.5, -0.134)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=15, rotation=28, color="black")
    ax.tick_params(axis="y", colors="black")
    plt.yticks(fontsize=17)

    if yvar == "num_cells_per_ug_protein":
        ax.set_ylabel("Symbiont Cells/ug Animal Protein", fontsize=33)
        ax.set_title("Average Population Symbiont Density over Time", fontsize=49)

    elif yvar == "ng_chlorophyll_per_ug_protein":
        ax.set_ylabel("ng Chl a/ug Animal Protein", fontsize=33)
        ax.set_title("Average Population Chlorophyll a over Time", fontsize=49)

    elif yvar == "ng_chlorophyll_per_hundred_cells":
        ax.set_ylabel("ng Chlorophyll a per 100 Cells", fontsize=25)

    legend_label = f"Intertidal zone: {zone}" if zone else "All intertidal zones"
    ax.legend([legend_label], loc="upper right", fontsize=30)

    ax.grid(axis="y", color="black", linestyle="--", linewidth=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()


def intertidal_box_plot(
    your_data: pd.DataFrame,
    yvar: str,
    yaxis: str,
    title: Optional[str] = None,
    save_path: Optional[PathLike] = None,
    box_color: str = "#4eb3d3",
) -> None:
    """
    Create a box plot comparing a variable across intertidal zones,
    using a consistent color for all boxes.

    This function:
    - Separates data by intertidal zone ('low', 'middle', 'high')
    - Extracts and cleans values for a specified variable
    - Creates a box plot with mean and median indicators
    - Applies a uniform box color for visual consistency
    - Annotates sample sizes for each zone
    - Optionally saves the resulting figure to disk

    Parameters
    ----------
    your_data : pandas.DataFrame
        Input dataframe containing the response variable and an
        'intertidal_zone' column.
    yvar : str
        Name of the column in `your_data` to visualize.
    yaxis : str
        Label for the y-axis.
    title : str, optional
        Title for the plot. If None, a default title is used.
    save_path : str or pathlib.Path, optional
        File path to save the plot. If None, the plot is not saved.
    box_color : str, default "#4eb3d3"
        Fill color applied uniformly to all box plots.

    Returns
    -------
    None

    Notes
    -----
    This function assumes the existence of helper functions:
    - `pull_data(data, yvar: str)`
    """
    intertidal_zones = ["low", "middle", "high"]
    data = []
    sample_sizes = []

    for zone in intertidal_zones:
        zone_data = your_data[your_data["intertidal_zone"] == zone]
        zone_data = pull_data(zone_data, yvar)
        zone_data = zone_data[~np.isnan(zone_data)]
        data.append(zone_data)
        sample_sizes.append(len(zone_data))

    labels = ["Low", "Middle", "High"]

    fig, ax = plt.subplots()
    ax.set_facecolor("white")

    box = ax.boxplot(
        data,
        tick_labels=labels,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "black"},
        medianprops={"color": "black", "linewidth": 1},
        patch_artist=True,
        boxprops=dict(facecolor=box_color, edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    ax.set_xlabel("Tidal Zone", fontsize=15, color="black")
    ax.xaxis.set_label_coords(0.5, -0.15)

    ax.set_ylabel(yaxis, fontsize=15, color="black")
    ax.set_title(title, fontsize=20)

    ax.set_xticklabels(labels, fontsize=13, color="black")
    ax.tick_params(axis="y", colors="black")

    # Sample size annotation
    legend_text = "\n".join(
        f"{zone}: n={size}" for zone, size in zip(intertidal_zones, sample_sizes)
    )

    ax.text(
        0.95,
        0.95,
        legend_text,
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        ha="right",
        color="black",
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="round,pad=0.3",
        ),
    )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()


def abiotic_plot(
    plot_type: str,
    data_dict: dict,
    xlabel: str,
    ylabel: str,
    xlim: tuple,
    ylim: tuple = None,
    title: str = None,
    save_path: str = None,
    show_drop_line: bool = False,
    split_date: str = "2022-11-15",
    drop_label: str = "Symbiont Drop",
    drop_label_y: float = 0.04,
    drop_label_fontsize: Optional[float] = None,
    month_ticks: bool = True,
    date_format: str = "%b %Y",
    edge_buffer_days: float = 5,
    edge_label_pad: float = 3,
    rain_color: str = "orange",
    rain_alpha: float = 1.0,
):
    """
    Function to create and save line or scatter plots with customization options.

    Parameters:
    -----------
    plot_type (str): 'line' or 'scatter'
        data_dict (dict): Dictionary of data with keys as labels and values as tuples (x, y, color, plot_type).
    title (str):
        Plot title.
    xlabel (str):
        Label for the x-axis.
    ylabel (str):
        Label for the y-axis.
    xlim (tuple):
        X-axis limits (start_date, end_date).
    ylim (tuple, optional):
        Y-axis limits.
    save_path (str, optional):
        File path to save the figure.
    show_drop_line (bool, default False):
        Draw a dashed vertical line at `split_date` with floating text
        (no box) beside it.
    split_date (str, default "2022-11-15"):
        Date of the line. The x-axis is a true date axis here, so no
        interpolation is needed.
    drop_label (str, default "Symbiont Drop"):
        Text beside the line.
    drop_label_y (float, default 0.04):
        Vertical position of the text in axes coordinates (0 = bottom).
        Values above 0.5 anchor the text from the top down instead.
    drop_label_fontsize (float, optional):
        Defaults to 16 for line plots and 12 for scatter plots.
    month_ticks (bool, default True):
        Put x ticks on the 1st of each month so "%Y-%m" labels mean what
        they say. False restores the original ticks every 4 weeks (on
        Mondays), where a "2022-11" label can sit on Nov 14.
    date_format (str, default "%b %Y"):
        strftime format for the x tick labels (e.g. "%Y-%m" for 2022-11).
    edge_buffer_days (float, default 5):
        With month_ticks, a label whose tick is within this many days of
        an axis end is left/right-aligned so it sits inside the axis
        instead of being centered on the axis line. Use 0 to disable.
        The tick mark at the axis end is hidden (it duplicates the axis line).
    edge_label_pad (float, default 3):
        Points to nudge those edge labels inward, which keeps the first x
        label from crowding the y-axis tick label at the corner.
    rain_color (str, default "orange"):
        Color of the rainfall bars (only used when "rain" is in data_dict).
    rain_alpha (float, default 1.0):
        Opacity of the rainfall bars.
    """
    fig, ax = plt.subplots(figsize=(14, 7) if plot_type == "line" else (12, 3.8))

    for label, (x, y, color, ptype) in data_dict.items():
        if ptype == "line":
            ax.plot(x, y, color=color, label=label, linewidth=2, zorder=3)
        elif ptype == "scatter":
            ax.scatter(x, y, color=color, label=label, s=15, zorder=3)

    if title:
        ax.set_title(title, fontsize=25 if plot_type == "scatter" else 30, pad=10)

    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xlim(xlim)
    if month_ticks:
        # a tick on the 1st of every month: the label is that month's start
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
    ax.xaxis.set_major_formatter(DateFormatter(date_format))

    if ylim:
        ax.set_ylim(ylim)

    ax.tick_params(axis="x", labelsize=12, rotation=0, labelbottom=True)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(plot_type == "scatter")
    ax.tick_params(axis="x", which="major", length=6)

    # Keep labels near the axis ends inside the axis (not centered on it)
    if month_ticks and edge_buffer_days > 0:
        lo, hi = ax.get_xlim()
        for loc, lab, tick in zip(
            ax.get_xticks(), ax.get_xticklabels(), ax.xaxis.get_major_ticks()
        ):
            if lo <= loc < lo + edge_buffer_days:
                side = 1
                lab.set_ha("left")
            elif hi - edge_buffer_days < loc <= hi:
                side = -1
                lab.set_ha("right")
            else:
                continue
            tick.tick1line.set_visible(False)  # tick mark would sit on the axis line
            lab.set_transform(
                lab.get_transform()
                + ScaledTranslation(side * edge_label_pad / 72, 0, fig.dpi_scale_trans)
            )

    # --- pre/post drop line (optional) ---
    if show_drop_line:
        split_ts = pd.Timestamp(split_date)
        split_num = mdates.date2num(split_ts)  # annotate needs a plain number
        lo, hi = ax.get_xlim()
        if lo <= split_num <= hi:
            ax.axvline(
                split_ts, color="#999999", linestyle="--", linewidth=1.5, zorder=2
            )
            ax.annotate(
                drop_label,
                xy=(split_num, drop_label_y),
                xycoords=ax.get_xaxis_transform(),  # x in data, y in axes fraction
                xytext=(6, 0),  # nudge right of the line (points)
                textcoords="offset points",
                rotation=-90,
                ha="left",
                va="top" if drop_label_y > 0.5 else "bottom",
                fontsize=drop_label_fontsize or (16 if plot_type == "line" else 12),
                color="#666666",
                zorder=4,
            )
        else:
            print(f"split_date {split_date} is outside xlim; drop line not drawn.")

    # Add twin axis if rain data is provided
    if "rain" in data_dict:
        ax2 = ax.twinx()
        ax2.bar(
            data_dict["rain"][0],
            data_dict["rain"][1],
            width=1.3,
            color=rain_color,
            alpha=rain_alpha,
            label="Rainfall",
        )
        ax2.set_ylabel("Rainfall (mm)", fontsize=12, rotation=270, va="bottom")
        ax2.set_xlim(xlim)
        ax2.tick_params(axis="y", labelsize=12)
        ax2.grid(False)

        # Combine legends for both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="lower left", fontsize=9)
    else:
        ax.legend(loc="best", fontsize=15)

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# New helpers
# ---------------------------------------------------------------------------


def load_ar_events(ar_data: Union[PathLike, pd.DataFrame]) -> pd.DataFrame:
    """Load the AR catalog (CSV path or DataFrame) and parse its dates.

    Expects the columns Start_Date, End_Date (YYYYMMDDHH) and AR_Scale.
    Adds datetime columns 'start' and 'end'.
    """
    if isinstance(ar_data, pd.DataFrame):
        df = ar_data.copy()
    else:
        df = pd.read_csv(ar_data, dtype={"Start_Date": str, "End_Date": str})
    df["start"] = pd.to_datetime(df["Start_Date"].astype(str), format="%Y%m%d%H")
    df["end"] = pd.to_datetime(df["End_Date"].astype(str), format="%Y%m%d%H")
    return df


def add_ar_shading(
    ax,
    x_dates: Sequence,
    ar_data: Union[PathLike, pd.DataFrame],
    color: str = "tab:blue",
    alpha: float = 0.2,
    min_width: float = 0.04,
    label_scale: bool = False,
    x_per_day: Optional[float] = None,
) -> Optional[Patch]:
    """
    Shade AR events on an axis whose x positions 0..n-1 correspond to
    `x_dates` (one date per categorical tick).

    Each AR is centered at its midpoint, interpolated between neighboring
    collection dates. Its width is proportional to its duration using one
    uniform scale (`x_per_day`) for all ARs, so equal durations look equal
    regardless of the gap between nearby collections. By default
    `x_per_day` is 1 / (median days between collections).

    Parameters
    ----------
    x_dates : sequence of date-like
        Collection dates for x positions 0..n-1 (must be increasing).
    min_width : float
        Minimum shaded width in x-axis units so short ARs stay visible.
    label_scale : bool
        If True, write the AR scale (1-5) at the top of each band.

    Returns
    -------
    A legend handle (Patch), or None if no AR fell within the plotted range.
    """
    ar = load_ar_events(ar_data)
    dates = pd.to_datetime(pd.Series(list(x_dates))).reset_index(drop=True)
    if len(dates) < 2:
        return None

    xp = np.array([t.timestamp() for t in dates])
    if np.any(np.diff(xp) <= 0):
        raise ValueError("x_dates must be strictly increasing to place AR shading.")
    fp = np.arange(len(dates))

    # Default scale: one tick spacing = the median gap between collections
    if x_per_day is None:
        x_per_day = 1.0 / (np.median(np.diff(xp)) / 86400.0)

    xlim = ax.get_xlim()  # axvspan can change limits; restore afterwards
    drawn = False
    for _, row in ar.iterrows():
        t0, t1 = row["start"].timestamp(), row["end"].timestamp()
        if t1 < xp[0] or t0 > xp[-1]:
            continue  # AR is outside the plotted date range
        # Center: interpolated position of the AR's midpoint between ticks.
        # Width: uniform scale, so equal durations look equal everywhere.
        mid = np.interp((t0 + t1) / 2, xp, fp)
        width = max((t1 - t0) / 86400.0 * x_per_day, min_width)
        x0, x1 = mid - width / 2, mid + width / 2
        ax.axvspan(x0, x1, color=color, alpha=alpha, linewidth=0, zorder=0.5)
        if label_scale and "AR_Scale" in ar.columns:
            ax.text(
                (x0 + x1) / 2,
                0.99,
                f"AR{int(row['AR_Scale'])}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=12,
                color=color,
            )
        drawn = True
    ax.set_xlim(xlim)

    if not drawn:
        return None
    return Patch(facecolor=color, alpha=max(alpha, 0.3), label="Atmospheric river")


# ---------------------------------------------------------------------------
# Updated functions
# ---------------------------------------------------------------------------


# Collection date for each batch size (position on the x-axis). Used to label
# batches that have no data in the selected subset, so they show as empty.
BATCH_DATES = {
    4: "2022-08-27",
    5: "2022-09-06",
    6: "2022-09-23",
    7: "2022-10-10",
    8: "2022-10-27",
    9: "2022-11-08",
    10: "2022-11-23",
    11: "2022-12-06",
    12: "2023-01-06",
    13: "2023-01-23",
    14: "2023-02-06",
    15: "2023-02-18",
    16: "2023-03-17",
}


def batch_box_plot(
    your_data: pd.DataFrame,
    yvar: str,
    yaxis: str,
    zone: Optional[str],
    title: Optional[str] = None,
    save_path: Optional[PathLike] = None,
    box_colors: Optional[Union[str, Sequence[str]]] = None,
    legend_loc: Union[str, Tuple[float, float]] = (0.98, 0.98),
    show_drop_line: bool = False,
    split_date: str = "2022-11-15",
    drop_label: Optional[str] = None,
    drop_label_y: float = 0.04,
    show_ar: bool = False,
    ar_data: Optional[Union[PathLike, pd.DataFrame]] = None,
    ar_color: str = "tab:blue",
    ar_alpha: float = 0.2,
    ar_label_scale: bool = False,
) -> None:
    """
    Create grouped box plots for a given variable and intertidal zone.

    (Original behavior unchanged; see original docstring.)

    Batches with no data (e.g. a zone that wasn't sampled on a date) are
    left as empty positions on the x-axis, labeled with the date from
    BATCH_DATES, so every plot has the same 13 slots.

    New parameters
    --------------
    legend_loc : str or (x, y), default (0.98, 0.98)
        Where the combined zone / n / AR legend box goes. Either a matplotlib
        location string ("upper left", "lower right", "center", ...) or an
        (x, y) tuple in axes coordinates (0-1) for the box's upper-right
        corner.
    show_drop_line : bool, default False
        If True, draw a dashed vertical line at `split_date` with floating
        text (no box) beside it.
    split_date : str, default "2022-11-15"
        Pre/post date for the line, placed by interpolation between
        collection dates.
    drop_label : str, optional
        Text beside the line. Defaults by `yvar`: "Symbiont Drop" for
        num_cells_per_ug_protein, "Chl a Drop" for chlorophyll variables.
    drop_label_y : float, default 0.04
        Vertical position of the text in axes coordinates (0 = bottom).
    show_ar : bool, default False
        If True, shade AR events behind the boxes.
    ar_data : str, pathlib.Path or DataFrame, optional
        AR catalog CSV (or DataFrame). Required when show_ar is True.
    ar_color, ar_alpha : styling of the shaded bands.
    ar_label_scale : bool, default False
        Label each band with its AR scale.
    """
    if show_ar and ar_data is None:
        raise ValueError("show_ar=True requires ar_data (CSV path or DataFrame).")

    batch_sizes = range(4, 17)

    selected_data = (
        your_data[your_data["intertidal_zone"] == zone] if zone else your_data
    )

    fig, ax = plt.subplots(figsize=(27, 10))
    ax.set_facecolor("white")

    box_data = {}  # x position -> values (only for batches that have data)
    batch_dates = []  # one date per x position, including empty batches

    for pos, size in enumerate(batch_sizes):
        batch = group_data(selected_data, size)
        if batch.empty:
            print(f"No data found for batch size {size}; leaving it empty.")
            batch_dates.append(BATCH_DATES[size])
            continue
        box_data[pos] = pull_data(batch, yvar)
        batch_dates.append(
            pd.Timestamp(batch["date_of_collection"].iloc[0]).strftime("%Y-%m-%d")
        )

    n_slots = len(batch_dates)
    if box_colors is None:
        box_colors = ["#4eb3d3"] * n_slots
    elif isinstance(box_colors, str):
        box_colors = [box_colors] * n_slots
    elif len(box_colors) < n_slots:
        print("Warning: Not enough colors provided. Using default for missing values.")
        box_colors = list(box_colors) + ["lightblue"] * (n_slots - len(box_colors))

    for pos, data in box_data.items():
        ax.boxplot(
            data,
            positions=[pos],
            patch_artist=True,
            showfliers=False,
            widths=0.5,
            boxprops=dict(facecolor=box_colors[pos]),
            medianprops={"color": "black"},
        )

    ax.set_xlabel("Collection Date", fontsize=25, color="black", labelpad=15)
    ax.set_ylabel(yaxis, fontsize=33, color="black", labelpad=15)
    ax.set_xticks(np.arange(n_slots))
    ax.set_xticklabels(batch_dates, fontsize=17, rotation=22, ha="right", color="black")
    ax.set_xlim(-0.5, n_slots - 0.5)  # keep empty slots at the edges visible
    ax.tick_params(axis="y", colors="black")
    plt.yticks(fontsize=17)
    if title:
        ax.set_title(title, fontsize=49, pad=10)

    # --- AR shading (optional) ---
    ar_handle = None
    if show_ar:
        ar_handle = add_ar_shading(
            ax,
            batch_dates,
            ar_data,
            color=ar_color,
            alpha=ar_alpha,
            label_scale=ar_label_scale,
        )

    # --- pre/post drop line (optional) ---
    if show_drop_line and len(batch_dates) >= 2:
        xp = np.array([pd.Timestamp(d).timestamp() for d in batch_dates])
        split_x = float(
            np.interp(pd.Timestamp(split_date).timestamp(), xp, np.arange(len(xp)))
        )
        ax.axvline(split_x, color="#999999", linestyle="--", linewidth=1.5, zorder=1)
        default_labels = {
            "num_cells_per_ug_protein": "Symbiont Drop",
        }
        ax.text(
            split_x + 0.16,
            drop_label_y,
            drop_label or default_labels.get(yvar, "Drop"),
            transform=ax.get_xaxis_transform(),
            rotation=-90,
            ha="right",
            va="bottom",
            fontsize=20,
            color="#666666",
            zorder=4,
        )

    zone_labels = {
        "low": "Lower Tidal Zone",
        "mid": "Middle Tidal Zone",
        "high": "Upper Tidal Zone",
    }

    if zone:
        zone_text = zone_labels.get(zone.lower(), f"{zone.title()} Tidal Zone")
    else:
        zone_text = "All Collection Zones"

    n_samples = len(selected_data)
    legend_text = f"{zone_text}\nn = {n_samples}"

    if ar_handle is not None:
        # One legend box: zone / n as the title, AR swatch as the entry
        if isinstance(legend_loc, str):
            loc_kwargs = dict(loc=legend_loc)
        else:
            loc_kwargs = dict(loc="upper right", bbox_to_anchor=legend_loc)
        leg = ax.legend(
            handles=[ar_handle],
            title=legend_text,
            fontsize=20,
            title_fontsize=20,
            fancybox=True,
            facecolor="white",
            edgecolor="black",
            framealpha=0.8,
            **loc_kwargs,
        )
        leg.get_title().set_multialignment("center")
    else:
        # No AR shown: original text box
        xy = (0.98, 0.98) if isinstance(legend_loc, str) else legend_loc
        ax.text(
            xy[0],
            xy[1],
            legend_text,
            transform=ax.transAxes,
            fontsize=20,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8
            ),
            multialignment="center",
        )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        full_save_path = f"{save_path}.png"
        plt.savefig(full_save_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved to {full_save_path}")

    plt.show()


def batch_bar_overlay(
    your_data: pd.DataFrame,
    yvar: str,
    save_path: Optional[PathLike] = None,
    colors: Sequence[str] = ("orange", "wheat", "red"),
    title: Optional[str] = None,
    show_ar: bool = False,
    ar_data: Optional[Union[PathLike, pd.DataFrame]] = None,
    ar_color: str = "tab:blue",
    ar_alpha: float = 0.2,
    ar_label_scale: bool = False,
) -> None:
    """
    Create a grouped bar plot with error bars for intertidal zone batch statistics.

    (Original behavior unchanged; see original docstring.)

    New parameters
    --------------
    show_ar : bool, default False
        If True, shade AR events behind the bars.
    ar_data : str, pathlib.Path or DataFrame, optional
        AR catalog CSV (or DataFrame). Required when show_ar is True.
    ar_color, ar_alpha : styling of the shaded bands.
    ar_label_scale : bool, default False
        Label each band with its AR scale.
    """
    if show_ar and ar_data is None:
        raise ValueError("show_ar=True requires ar_data (CSV path or DataFrame).")

    labels = [
        "2022-08-27",
        "2022-09-06",
        "2022-09-23",
        "2022-10-10",
        "2022-10-27",
        "2022-11-08",
        "2022-11-23",
        "2022-12-06",
        "2023-01-06",
        "2023-01-23",
        "2023-02-06",
        "2023-02-18",
        "2023-03-17",
    ]
    batch_sizes = range(4, 17)

    batches = []
    means = []
    stds = []
    sems = []

    selected_zones = ["low", "middle", "high"]

    for zone in selected_zones:
        selected_data = your_data[your_data["intertidal_zone"] == zone]

        for size in batch_sizes:
            batch = group_data(selected_data, size)
            batch = pull_data(batch, yvar)
            batches.append(batch)
            means.append(np.mean(batch))
            stds.append(np.std(batch))
            sems.append(sem(batch))

    x_pos = np.arange(len(labels))
    num_zones = len(selected_zones)
    width = 0.75 / num_zones

    fig, ax = plt.subplots(figsize=(27, 10))
    ax.set_facecolor("white")

    handles = []
    for i, zone in enumerate(selected_zones):
        start = i * width - (num_zones - 1) * width / 2
        CTEs = means[i * len(batch_sizes) : (i + 1) * len(batch_sizes)]
        SEMs = sems[i * len(batch_sizes) : (i + 1) * len(batch_sizes)]
        error = stds[i * len(batch_sizes) : (i + 1) * len(batch_sizes)]
        bar = ax.bar(x_pos + start, CTEs, width=width, color=colors[i], zorder=2)
        handles.append(bar)
        ax.errorbar(
            x_pos + start,
            CTEs,
            yerr=SEMs,
            fmt="o",
            color="black",
            capsize=4,
            elinewidth=1,
        )

    ax.set_xlabel("Date", fontsize=20, color="black")
    ax.xaxis.set_label_coords(0.5, -0.134)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=15, rotation=28, color="black")
    ax.tick_params(axis="y", colors="black")
    plt.yticks(fontsize=17)

    if yvar == "num_cells_per_ug_protein":
        ax.set_ylabel("Symbiont Cells/ug Animal Protein", fontsize=33)
        if title:
            ax.set_title(title, fontsize=49)

    if yvar == "ng_chlorophyll_per_ug_protein":
        ax.set_ylabel("ng Chl a/ug Animal Protein", fontsize=33)
        if title:
            ax.set_title(title, fontsize=49)

    if yvar == "ng_chlorophyll_per_hundred_cells":
        ax.set_ylabel("ng Chl a/100 Symbiont Cells", fontsize=25)

    n_value = len(your_data[your_data[yvar].notnull()])

    legend_labels = [
        f'Intertidal zone: {zone} (n={len(your_data[your_data["intertidal_zone"]==zone])})'
        for zone in selected_zones
    ]

    # --- AR shading (optional) ---
    if show_ar:
        ar_handle = add_ar_shading(
            ax,
            labels,
            ar_data,
            color=ar_color,
            alpha=ar_alpha,
            label_scale=ar_label_scale,
        )
        if ar_handle is not None:
            handles.append(ar_handle)
            legend_labels.append(ar_handle.get_label())

    ax.legend(handles, legend_labels, loc="upper right", fontsize=20)

    ax.grid(axis="y", color="black", linestyle="--", linewidth=0.5, alpha=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Line-plot version of batch_bar_overlay
# ---------------------------------------------------------------------------

# Pre/post (mid-November) % change in mean, from the GEE table.
PRE_POST_PCT_CHANGE = {
    "num_cells_per_ug_protein": {  # Symbiont density
        "all": -50.6,
        "low": -49.3,
        "middle": -51.8,
        "high": -44.8,
    },
    "ng_chlorophyll_per_ug_protein": {  # Chl a concentration
        "all": -58.1,
        "low": -59.7,
        "middle": -55.5,
        "high": -63.2,
    },
}


def batch_line_plot(
    your_data: pd.DataFrame,
    yvar: str,
    save_path: Optional[PathLike] = None,
    colors: Sequence[str] = ("#8fd0a8", "#3fb6cc", "#1b3a6b"),
    title: Optional[str] = None,
    # AR shading
    show_ar: bool = False,
    ar_data: Optional[Union[PathLike, pd.DataFrame]] = None,
    ar_color: str = "#a020a0",
    ar_alpha: float = 0.3,
    ar_label_scale: bool = False,
    # pre/post split + annotations
    split_date: Optional[str] = "2022-11-15",
    show_drop_line: bool = True,
    drop_label: Optional[str] = None,
    drop_label_y: float = 0.04,
    pct_change: Optional[dict] = None,
    pct_box_title: str = "Avg % change between\npre- and post-drop periods",
    footnote: Optional[str] = (
        "All pre/post zone contrasts significant at p < 0.001 (GEE, Holm-adjusted)"
    ),
    pct_box_loc: Tuple[float, float] = (0.98, 0.98),
    # look
    show_sem: bool = False,
    short_labels: bool = False,
    legend_loc: Union[str, Tuple[float, float]] = "below",
    bg_color: str = "white",
    figsize: Tuple[float, float] = (27, 10),
) -> None:
    """
    Line-plot version of `batch_bar_overlay`: one line per intertidal zone
    across collection dates, with an optional dashed pre/post split, per-zone
    % change annotations, and AR shading.

    Parameters
    ----------
    your_data, yvar, save_path, colors, title :
        As in `batch_bar_overlay` (colors are for low, middle, high).
    show_ar, ar_data, ar_color, ar_alpha, ar_label_scale :
        AR shading options, as in `batch_box_plot`.
    split_date : str or None
        Date of the dashed pre/post line (placed between collection dates by
        interpolation). None hides the line and the annotations.
    show_drop_line : bool, default True
        Draw the dashed vertical line at `split_date` with floating text
        beside it (no box). Turn off to hide both; the % change box (which
        also needs `split_date`) is unaffected.
    drop_label : str, optional
        Text beside the line. Defaults by `yvar`: "Symbiont Drop" for
        num_cells_per_ug_protein, "Chl a Drop" for the chlorophyll
        variables.
    drop_label_y : float
        Vertical position of the text in axes coordinates (0 = bottom);
        it is drawn rotated along the left side of the line.
    pct_change : dict, optional
        {"low": -49.3, "middle": -51.8, "high": -44.8}. Defaults to the
        values in PRE_POST_PCT_CHANGE for `yvar`, if any.
    pct_box_title : str
        Title of the boxed % change legend next to the dashed line.
    footnote : str or None
        Small text under the legend (only shown when annotations are shown).
    show_sem : bool
        Add SEM error bars.
    short_labels : bool, default False
        False: full YYYY-MM-DD labels, rotated, like batch_box_plot.
        True: short MM-DD labels (no year), horizontal.
    legend_loc : "below", a matplotlib loc string, or (x, y) axes coords
        "below" puts a horizontal legend under the axis, left-aligned.
    """
    if show_ar and ar_data is None:
        raise ValueError("show_ar=True requires ar_data (CSV path or DataFrame).")

    labels = [
        "2022-08-27",
        "2022-09-06",
        "2022-09-23",
        "2022-10-10",
        "2022-10-27",
        "2022-11-08",
        "2022-11-23",
        "2022-12-06",
        "2023-01-06",
        "2023-01-23",
        "2023-02-06",
        "2023-02-18",
        "2023-03-17",
    ]
    batch_sizes = range(4, 17)
    selected_zones = ["low", "middle", "high"]

    means, sems_ = {}, {}
    for zone in selected_zones:
        zone_data = your_data[your_data["intertidal_zone"] == zone]
        z_means, z_sems = [], []
        for size in batch_sizes:
            batch = pull_data(group_data(zone_data, size), yvar)
            z_means.append(np.mean(batch))
            z_sems.append(sem(batch))
        means[zone], sems_[zone] = np.array(z_means), np.array(z_sems)

    x_pos = np.arange(len(labels))
    grey = "black"  # text/tick color, matching batch_box_plot

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # --- lines ---
    handles, legend_labels = [], []
    for zone, color in zip(selected_zones, colors):
        (line,) = ax.plot(
            x_pos,
            means[zone],
            color=color,
            linewidth=3.5,
            marker="o",
            markersize=11,
            zorder=3,
        )
        if show_sem:
            ax.errorbar(
                x_pos,
                means[zone],
                yerr=sems_[zone],
                fmt="none",
                ecolor=color,
                elinewidth=1.5,
                capsize=4,
                alpha=0.7,
                zorder=2,
            )
        n_zone = len(your_data[your_data["intertidal_zone"] == zone])
        handles.append(line)
        legend_labels.append(f"{zone.title()} zone (n={n_zone})")

    # --- styling (matches batch_box_plot: black frame, ticks, big labels) ---
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
    ax.set_axisbelow(True)
    # ax.grid(axis="y", linestyle="--", color="#000000", linewidth=0.5, alpha=0.5, zorder=1)
    ax.tick_params(axis="both", colors=grey, length=5)
    ax.set_xticks(x_pos)
    tick_labels = [d[5:] for d in labels] if short_labels else labels
    ax.set_xticklabels(
        tick_labels,
        fontsize=17,
        color=grey,
        rotation=0 if short_labels else 22,
        ha="center" if short_labels else "right",
    )
    ax.set_xlabel("Collection Date", fontsize=25, color=grey, labelpad=15)
    ax.tick_params(axis="y", labelsize=17)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(bottom=0)

    ylabels = {
        "num_cells_per_ug_protein": "Symbiont Cells/ug Animal Protein",
        "ng_chlorophyll_per_ug_protein": "ng Chl a/ug Animal Protein",
        "ng_chlorophyll_per_hundred_cells": "ng Chl a/100 Symbiont Cells",
    }
    ax.set_ylabel(ylabels.get(yvar, yvar), fontsize=33, color=grey, labelpad=15)
    if title:
        ax.set_title(title, fontsize=49, pad=10, color="black")

    # --- AR shading ---
    if show_ar:
        ar_handle = add_ar_shading(
            ax,
            labels,
            ar_data,
            color=ar_color,
            alpha=ar_alpha,
            label_scale=ar_label_scale,
        )
        if ar_handle is not None:
            handles.append(ar_handle)
            legend_labels.append(ar_handle.get_label())

    # --- dashed pre/post split + % change annotations ---
    pcts = pct_change if pct_change is not None else PRE_POST_PCT_CHANGE.get(yvar)
    if split_date is not None:
        xp = np.array([pd.Timestamp(d).timestamp() for d in labels])
        split_x = float(np.interp(pd.Timestamp(split_date).timestamp(), xp, x_pos))
        if show_drop_line:
            ax.axvline(
                split_x, color="#999999", linestyle="--", linewidth=1.5, zorder=1
            )
            default_labels = {
                "num_cells_per_ug_protein": "Symbiont Drop",
                "ng_chlorophyll_per_ug_protein": "Chl a Drop",
                "ng_chlorophyll_per_hundred_cells": "Chl a Drop",
            }
            label = drop_label or default_labels.get(yvar, "Drop")
            ax.text(
                split_x - 0.08,
                drop_label_y,
                label,
                transform=ax.get_xaxis_transform(),
                rotation=90,
                ha="right",
                va="bottom",
                fontsize=20,
                color="#666666",
                zorder=4,
            )

        if pcts:
            zone_color = dict(zip(selected_zones, colors))
            short_name = {"low": "low", "middle": "mid", "high": "high"}
            ordered = sorted(
                [z for z in selected_zones if z in pcts], key=lambda z: pcts[z]
            )
            blank = [Line2D([], [], alpha=0) for _ in ordered]
            texts = [
                f"{short_name[z]} {pcts[z]:.0f}%".replace("-", "\u2212")
                for z in ordered
            ]
            pct_leg = ax.legend(
                blank,
                texts,
                title=pct_box_title,
                loc="upper left",
                bbox_to_anchor=(split_x + pct_box_loc[0], pct_box_loc[1]),
                bbox_transform=ax.get_xaxis_transform(),
                handlelength=0,
                handletextpad=0,
                fontsize=18,
                title_fontsize=15,
                fancybox=True,
                facecolor="white",
                edgecolor="black",
                framealpha=0.8,
            )
            for t, z in zip(pct_leg.get_texts(), ordered):
                t.set_color(zone_color[z])
                t.set_fontweight("bold")
            pct_leg.get_title().set_multialignment("center")
            ax.add_artist(pct_leg)  # keep it when the main legend is drawn

    # --- legend (+ footnote) ---
    if legend_loc == "below":
        # sits below the tick labels and the "Collection Date" x label
        legend_y = -0.17 if short_labels else -0.23
        ax.legend(
            handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(0.0, legend_y),
            ncol=len(handles),
            frameon=False,
            fontsize=20,
        )
        note_xy = (0.0, legend_y - 0.08)
    else:
        if isinstance(legend_loc, str):
            ax.legend(handles, legend_labels, loc=legend_loc, fontsize=20)
        else:
            ax.legend(
                handles,
                legend_labels,
                loc="upper right",
                bbox_to_anchor=legend_loc,
                fontsize=20,
            )
        note_xy = (0.0, -0.08)

    if footnote and pcts and split_date is not None:
        ax.text(
            note_xy[0],
            note_xy[1],
            footnote,
            transform=ax.transAxes,
            fontsize=15,
            color="#777777",
            va="top",
            ha="left",
        )

    if save_path:
        plt.savefig(
            save_path, bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor()
        )
        print(f"Plot saved to {save_path}")

    plt.show()


def regression_plot(
    your_data: pd.DataFrame,
    xvar: str,
    yvar: str,
    title: str,
    color: str = "Blue",
    save_path: str = None,
    ax: plt.Axes = None,
):
    """
    Create a regression plot with statistical annotations.
    Parameters
    ----------
    your_data : pandas.DataFrame
        Input dataframe containing the x and y variables.
    xvar : str
        Name of the column in `your_data` to use as the x variable.
    yvar : str
        Name of the column in `your_data` to use as the y variable.
    title : str
        Title for the plot.
    color : str, default "Blue"
        Color for the regression line and points.
    save_path : str, optional
        File path to save the plot. If None, the plot is not saved.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.
    Returns
    -------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
    """
    your_data = your_data[[xvar, yvar]].dropna()

    # Convert to float
    your_data[xvar] = your_data[xvar].astype(float)
    your_data[yvar] = your_data[yvar].astype(float)

    # Fit model with statsmodels
    X = sm.add_constant(your_data[xvar])
    y = your_data[yvar]
    model = sm.OLS(y, X).fit()

    # Durbin-Watson
    residuals = model.resid
    dw = sm.stats.durbin_watson(residuals)

    # Create axis if none passed
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        created_fig = True

    # Plot with seaborn
    sns.regplot(
        x=your_data[xvar],
        y=your_data[yvar],
        data=your_data,
        color=color,
        line_kws={"color": color},
        ax=ax,
    )

    ax.set_title(title, fontsize=15)

    if xvar == "temp_c_seven_day_average":
        ax.set_xlabel("Seven Day Avg. Temperature (°C)", fontsize=12)
    else:
        ax.set_xlabel("Seven Day Avg. Salinity (ppt)", fontsize=12)

    if yvar == "avg_num_cells_per_ug_protein":
        ax.set_ylabel(
            "Avg. Symbiont Cells/µg Animal Protein\nper Collection", fontsize=12
        )
    else:
        ax.set_ylabel("Avg. ng Chl a/µg Animal Protein\nper Collection", fontsize=12)

    # Add stats box
    r2 = model.rsquared
    p_val = model.pvalues[xvar]  # use variable name, not integer position
    se = model.bse[xvar]
    stats_text = f"$R^2$: {r2:.3f}\n$p$: {p_val:.3f}\nSE: {se:.3f}\nDW: {dw:.3f}"
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    # If standalone: save/show here
    if created_fig:
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    print(model.summary())
    return model


def export_plos_tiff(
    input_path: PathLike,
    output_path: Optional[PathLike] = None,
    dpi: Tuple[int, int] = (300, 300),
    max_width_px: int = 2250,
    max_height_px: int = 2625,
) -> None:
    """
    Process and export an image as a TIFF file suitable for PLOS submission.
    This function:
        - Opens the input image
        - Trims whitespace around the figure
        - Adds controlled padding back to the image
        - Resizes the image to fit within specified max dimensions
    Parameters
    ----------
    input_path : str or pathlib.Path
        Path to the input image file.
    output_path : str or pathlib.Path, optional
        Path to save the processed TIFF image. If None, saves
        with '_plos.tiff' suffix in the same directory as input.
    dpi : tuple of int, default (300, 300)
        DPI settings for the output TIFF image.
    max_width_px : int, default 2250
        Maximum width in pixels for the output image.
    max_height_px : int, default 2625
        Maximum height in pixels for the output image.
    Returns
    -------
    None
    """
    img = Image.open(input_path)

    if img.mode != "RGB":
        img = img.convert("RGB")

    # --- Trim whitespace ---
    bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()

    if bbox:
        img = img.crop(bbox)
        print("Cropped whitespace around figure.")

    # --- Add controlled padding back ---
    pad_frac = 0.02  # 2% padding
    pad_x = int(img.size[0] * pad_frac)
    pad_y = int(img.size[1] * 0.04)  # Slightly more vertical padding

    img = ImageOps.expand(
        img,
        border=(pad_x, pad_y, pad_x, pad_y),
        fill="white",
    )

    orig_width, orig_height = img.size
    print(f"Post-crop size: {orig_width} x {orig_height} px")

    width_ratio = max_width_px / orig_width
    height_ratio = max_height_px / orig_height
    resize_ratio = min(1.0, width_ratio, height_ratio)

    if resize_ratio < 1.0:
        new_size = (int(orig_width * resize_ratio), int(orig_height * resize_ratio))
        img = img.resize(new_size, Image.LANCZOS)

    if output_path is None:
        output_path = str(input_path).rsplit(".", 1)[0] + "_plos.tiff"

    img.save(output_path, format="TIFF", dpi=dpi, compression="tiff_lzw")


def inspect_image(path: PathLike) -> None:
    """
    Print basic metadata about an image file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the image file.

    Returns
    -------
    None
    """
    with Image.open(path) as img:
        print(f"Mode: {img.mode}")
        print(f"Size: {img.size} pixels")
        print(f"Info: {img.info}")


def show_tiff(path: PathLike) -> None:
    """
    Display a TIFF image using matplotlib.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the TIFF image.

    Returns
    -------
    None
    """
    img = Image.open(path)
    plt.imshow(img)
    plt.axis("off")  # Hide axes
    plt.show()


def intertidal_graph(your_data: pd.DataFrame, yvar: str) -> None:
    """
    Generate a bar graph comparing means of a specified variable across intertidal zones.

    Parameters
    ----------
    your_data : pandas.DataFrame
        Input dataframe containing the response variable and an 'intertidal_zone' column.
    yvar : str
        Name of the column in `your_data` to analyze and plot.
    Returns
    -------
    None
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
        ax.set_title("Middle Tidal Zone has \n Highest Symbiont Density", fontsize=20)
    elif yvar == "ng_chlorophyll_per_ug_protein":
        ax.set_ylabel(
            "ng Chlorophyll a \n  per Animal Protein", fontsize=17, color="black"
        )
        ax.set_title(
            "Middle Tidal Zone has \n Highest Chlorophyll a Production", fontsize=20
        )
    elif yvar == "ng_chlorophyll_per_hundred_cells":
        ax.set_ylabel("ng Chlorophyll a \n  per 100 Cells", fontsize=17, color="black")
        # ax.set_title('Tidal Zone on Chlorophyll a per Cell', fontsize = 20)

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
        label1 = "Symbiont Density"
    elif yvar1 == "ng_chlorophyll_per_ug_protein":
        label1 = "Chlorophyll a Concentration"
    elif yvar1 == "ng_chlorophyll_per_hundred_cells":
        label1 = "ng Chlorophyll a per 100 Cells"
    else:
        label1 = "Symbiont Density"

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
    # ax.set_title('Symbiont Density and Star Oddi Temp', fontsize =25)
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
        # ax.set_title('Merged symbiont and Fort Point salinity data overlayed', fontsize=20)

    if yvar1 == "ng_chlorophyll_per_ug_protein":
        ax.set_ylabel("ng Chlorophyll a per Animal Protein", fontsize=15)
        ax.set_title("Chlorophyll a with Fort Point Salinity Overlayed", fontsize=20)

    if yvar1 == "ng_chlorophyll_per_hundred_cells":
        ax.set_ylabel("ng Chlorophyll a per 100 Cells", fontsize=15)

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
