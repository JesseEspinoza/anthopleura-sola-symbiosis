# load some library
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import subprocess
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
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
#from google.colab.data_table import DataTable
#DataTable.max_columns = 40
import scikit_posthocs as sp
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import sys
#import msfunctions

from utils.functions import group_data, pull_data, get_y_label, get_title

def batch_bar(your_data, yvar, zone, save_path=None):
    labels = ['Aug 27, 2022', 'Sept 6, 2022', 'Sept 23, 2022', 'Oct 10, 2022',
              'Oct 27, 2022', 'Nov 08, 2022', 'Nov 23, 2022', 'Dec 6, 2022',
              'Jan 06, 2023', 'Jan 23, 2023', 'Feb 6, 2023', 'Feb 18, 2023',
              'Mar 17, 2023'
              ]
    batch_sizes = range(4, 17)

    batches = []
    means = []
    stds = []
    sems = []

    if zone:
        selected_data = your_data[your_data['intertidal_zone'] == zone]
    else:
        selected_data = your_data


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
    ax.bar(x_pos, CTEs, width=0.75, color='mediumseagreen', zorder=2)
    plt.errorbar(x_pos, CTEs, yerr=SEMs, fmt='o', color='black')
    ax.set_xlabel('Date', fontsize=20, color='black')
    ax.xaxis.set_label_coords(0.5, -.134)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=15, rotation=28, color='black')
    ax.tick_params(axis='y', colors='black')
    plt.yticks(fontsize=17)

    if yvar == 'num_cells_per_ug_protein':
        ax.set_ylabel('Algal Cells/ug Animal Protein', fontsize=33)
        ax.set_title('Average Population Algal Density over Time', fontsize=49)

    if yvar == 'ng_chlorophyll_per_ug_protein':
        ax.set_ylabel('ng Chl α/ug Animal Protein', fontsize=33)
        ax.set_title('Average Population Chlorophyll α over Time', fontsize=49)

    if yvar == 'ng_chlorophyll_per_hundred_cells':
        ax.set_ylabel('ng Chlorophyll per 100 Cells', fontsize=25)

    if zone:
        legend_label = f'Intertidal zone: {zone}'
    else:
        legend_label = 'all intertidal zones'
    ax.legend([legend_label], loc='upper right', fontsize=30)

    ax.grid(axis='y', color='black', linestyle='--', linewidth=0.5)

    '''
    for i in range(len(batches)):
        for j in range(i + 1, len(batches)):
            print(f'Kruskal testing batch_{batch_sizes[i]} and batch_{batch_sizes[j]}:', (stats.kruskal(batches[i], batches[j])))
    '''

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")    

    plt.show()

def intertidal_box_plot(your_data, yvar, yaxis, save_path=None):
    intertidal_zones = ['low', 'middle', 'high']
    data = []
    sample_sizes = []  # List to store sample sizes for each zone

    for zone in intertidal_zones:
        zone_data = your_data[your_data.intertidal_zone == zone]
        zone_data = pull_data(zone_data, yvar)
        zone_data = zone_data[~np.isnan(zone_data)]
        data.append(zone_data)
        sample_sizes.append(len(zone_data))  # Store the sample size for this zone

    labels = ['Low', 'Middle', 'High']

    fig, ax = plt.subplots()
    ax.set_facecolor("white")

    box = ax.boxplot(
        data, labels=labels, showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "black"},
        medianprops={"color": "black", "linewidth": 1},  # Make median line black
        patch_artist=True)

    # Set box colors to white
    for b in box['boxes']:
        b.set(facecolor='white')

    ax.set_xlabel('Tidal Zone', fontsize=15, color='black')
    ax.xaxis.set_label_coords(0.5, -.15)

    ax.set_ylabel(yaxis, fontsize=15, color='black')

    ax.set_title(get_title(yvar), fontsize=20)
    ax.set_xticklabels(labels, fontsize=13, color='black')
    ax.tick_params(axis='y', colors='black')
    ax.grid(axis='y', color='black', linestyle='--', linewidth=0.5)

    # Adding the sample size legend to the top right corner
    legend_text = "\n".join([f"{zone}: n={size}" for zone, size in zip(intertidal_zones, sample_sizes)])
    ax.text(0.95, 0.95, legend_text, transform=ax.transAxes, fontsize=12, va='top', ha='right', 
            color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()

def abiotic_plot(plot_type, data_dict, title, xlabel, ylabel, xlim, ylim=None, save_path=None):
    """
    Function to create and save line or scatter plots with customization options.

    Parameters:
        plot_type (str): 'line' or 'scatter'
        data_dict (dict): Dictionary of data with keys as labels and values as tuples (x, y, color, plot_type).
        title (str): Plot title.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        xlim (tuple): X-axis limits (start_date, end_date).
        ylim (tuple, optional): Y-axis limits.
        save_path (str, optional): File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(14, 7) if plot_type == 'line' else (12, 3.8))
    
    for label, (x, y, color, ptype) in data_dict.items():
        if ptype == 'line':
            ax.plot(x, y, color=color, label=label, linewidth=2, zorder=3)
        elif ptype == 'scatter':
            ax.scatter(x, y, color=color, label=label, s=15, zorder=3)

    ax.set_title(title, fontsize=25 if plot_type == 'scatter' else 30, pad=10)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    ax.tick_params(axis='x', labelsize=12, rotation=0, labelbottom=True)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(plot_type == 'scatter')

    # Add twin axis if rain data is provided
    if 'rain' in data_dict:
        ax2 = ax.twinx()
        ax2.bar(data_dict['rain'][0], data_dict['rain'][1], width=1.3, color='orange', label='Rainfall')
        ax2.set_ylabel('Rainfall (mm)', fontsize=12, rotation=270, va="bottom")
        ax2.set_xlim(xlim)
        ax2.tick_params(axis='y', labelsize=12)
        ax2.grid(False)

        # Combine legends for both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='lower left', fontsize=9)
    else:
        ax.legend(loc='best', fontsize=15)

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def batch_box_plot(your_data, yvar, yaxis, zone, save_path=None):
    batch_sizes = range(4, 17)

    valid_batch_sizes = []  # Keep track of valid batch sizes
    batch_dates = []  # Store the collection dates of batches

    if zone:
        selected_data = your_data[your_data['intertidal_zone'] == zone]
    else:
        selected_data = your_data

    fig, ax = plt.subplots(figsize=(27, 10))
    ax.set_facecolor("white")

    box_data = []  # Store boxplot data for each batch size

    for size in batch_sizes:
        batch = group_data(selected_data, size)
        if batch.empty:
            print(f"No data found for batch size {size}, skipping...")
            continue  # Skip this batch size and move to the next
        batch_data = pull_data(batch, yvar)
        box_data.append(batch_data)
        valid_batch_sizes.append(size)  # Store valid batch size
        batch_dates.append(batch['date_of_collection'].iloc[0])  # Store collection date

    # Create individual box plots for each batch size
    for i, data in enumerate(box_data):
        ax.boxplot(data, positions=[i], patch_artist=True, showfliers=False, widths = 0.5, boxprops=dict(facecolor="goldenrod"), medianprops={'color': 'black'})

    ax.set_xlabel('Collection Date', fontsize=25, color='black', labelpad=15)
    ax.set_ylabel(yaxis, fontsize=25, color='black', labelpad=15)
    ax.set_xticks(np.arange(len(box_data)))
    ax.set_xticklabels(batch_dates, fontsize=17, rotation=22, ha="right", color='black')
    ax.tick_params(axis='y', colors='black')
    plt.yticks(fontsize=17)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()


def batch_bar_overlay(your_data, yvar, save_path=None):
    labels = ['2022-08-27', '2022-09-06', '2022-09-23', '2022-10-10',
                '2022-10-27', '2022-11-08', '2022-11-23', '2022-12-06',
                '2023-01-06', '2023-01-23', '2023-02-06', '2023-02-18',
                '2023-03-17'
                ]
    batch_sizes = range(4, 17)

    batches = []
    means = []
    stds = []
    sems = []

    selected_zones = ['low', 'middle', 'high']

    for zone in selected_zones:
        selected_data = your_data[your_data['intertidal_zone'] == zone]

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
    colors = ['orange', 'wheat', 'red'][:num_zones]

    fig, ax = plt.subplots(figsize=(27, 10))
    ax.set_facecolor("white")

    handles = []
    for i, zone in enumerate(selected_zones):
        start = i * width - (num_zones - 1) * width / 2
        CTEs = means[i * len(batch_sizes): (i + 1) * len(batch_sizes)]
        SEMs = sems[i * len(batch_sizes): (i + 1) * len(batch_sizes)]
        error = stds[i * len(batch_sizes): (i + 1) * len(batch_sizes)]
        bar = ax.bar(x_pos + start, CTEs, width=width, color=colors[i], zorder=2)
        handles.append(bar)
        ax.errorbar(x_pos + start, CTEs, yerr=SEMs, fmt='o', color='black', capsize=4, elinewidth=1)

    ax.set_xlabel('Date', fontsize=20, color='black')
    ax.xaxis.set_label_coords(0.5, -.134)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=15, rotation=28, color='black')
    ax.tick_params(axis='y', colors='black')
    plt.yticks(fontsize=17)

    if yvar == 'num_cells_per_ug_protein':
        ax.set_ylabel('Algal Cells/ug Animal Protein', fontsize=33)
        ax.set_title('Sharp Algal Density Reduction in November', fontsize=49)

    if yvar == 'ng_chlorophyll_per_ug_protein':
        ax.set_ylabel('ng Chl α/ug Animal Protein', fontsize=33)
        ax.set_title('Gradual Chlorophyll α Reduction Following Seasonal Changes', fontsize=49)

    if yvar == 'ng_chlorophyll_per_hundred_cells':
        ax.set_ylabel('ng Chl α/100 Algae Cells', fontsize=25)

    n_value = len(your_data[your_data[yvar].notnull()])

    legend_labels = [f'Intertidal zone: {zone} (n={len(your_data[your_data["intertidal_zone"]==zone])})' for zone in selected_zones]

    ax.legend(handles, legend_labels, loc='upper right', fontsize=20)

    ax.grid(axis='y', color='black', linestyle='--', linewidth=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()