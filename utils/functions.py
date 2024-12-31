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

def normalcy(dataset, desired_variable):
    x = pull_data(dataset, desired_variable)
    xmask = x[np.logical_not(np.isnan(x))]
    shapiro_test = stats.shapiro(xmask)
    print(shapiro_test)

def kruskal_drops(your_data, yvar, zone):

  pre_drop_date_start = '2022-07-01T12:00:00'
  pre_drop_date_end = '2022-11-25T12:00:00'
  pre_drop = your_data[(your_data['date_time'] > pre_drop_date_start) & (your_data['date_time'] <= pre_drop_date_end)]

  post_drop_date_start = '2022-11-26T12:00:00'
  post_drop_date_end = '2023-04-06T12:00:00'
  post_drop = your_data[(your_data['date_time'] > post_drop_date_start) & (your_data['date_time'] <= post_drop_date_end)]

  print(f'Testing pre {stats.shapiro(pre_drop[yvar])} vs post {stats.shapiro(post_drop[yvar])} drop normalcy')

  if zone == 'all':
    print(f'Testing all intertidal zones before vs after drop:', stats.kruskal(pre_drop[yvar], post_drop[yvar]))
  else:
    pre_drop = pre_drop[pre_drop['intertidal_zone'] == zone]
    post_drop = post_drop[post_drop['intertidal_zone'] == zone]
    print(f'Testing {zone} intertidal zone before vs after drop:', stats.kruskal(pre_drop[yvar], post_drop[yvar]))

def group_data(your_data, group_num):
    group = your_data[your_data.collection_group == group_num]
    if group.empty:
        print(f"No data found for group {group_num}")
    return group

def pull_data(your_data, desired_variable):
    '''
    desried variable need to be in ''
    '''
    variable = your_data[desired_variable]
    if variable.empty:
        print(f"No data found for variable {desired_variable}")
    return variable

def intertidal_graph(your_data, yvar):
    intertidal_zones = ['low', 'medium', 'high']
    data = []

    for zone in intertidal_zones:
        zone_data = your_data[your_data.intertidal_zone == zone]
        zone_data = pull_data(zone_data, yvar)
        zone_data = zone_data[~np.isnan(zone_data)]

        zone_mean = np.mean(zone_data)
        zone_std = np.std(zone_data)
        zone_sem = sem(zone_data)

        data.append((zone_mean, zone_sem, zone_std))

    labels = ['Low', 'Middle', 'High']
    x_pos = np.arange(len(labels))
    CTEs = [mean for mean, _, _ in data]
    SEMs = [sem for _, sem, _ in data]
    error = [std for _, _, std in data]

    fig, ax = plt.subplots()
    ax.set_facecolor("white")
    ax.bar(x_pos, CTEs,
           width=0.7,
           zorder=2,
           color='goldenrod')
    plt.errorbar(x_pos, CTEs, yerr=SEMs, fmt='o', color='black')
    ax.set_xlabel('Tidal Zone', fontsize=20, color='black')
    ax.xaxis.set_label_coords(0.5, -.15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=17, color='black')
    ax.tick_params(axis='y', colors='black')

    if yvar == 'num_cells_per_ug_protein':
        ax.set_ylabel('Cells/ug Animal Protein', fontsize=20, color='black')
        ax.set_title('Middle Tidal Zone has \n Highest Algal Density', fontsize=20)
    elif yvar == 'ng_chlorophyll_per_ug_protein':
        ax.set_ylabel('ng Chlorophyll \n  per Animal Protein', fontsize=17, color='black')
        ax.set_title('Middle Tidal Zone has \n Highest Chlorophyll α Production', fontsize=20)
    elif yvar == 'ng_chlorophyll_per_hundred_cells':
        ax.set_ylabel('ng Chlorophyll \n  per 100 Cells', fontsize=17, color='black')
        # ax.set_title('Tidal Zone on Chlorophyll per Cell', fontsize = 20)

    ax.grid(axis='y', color='black', linestyle='--', linewidth=0.5)

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

        print(f'Kruskal testing {zone1}, {zone2}, and {zone3} zones:')
        print(f"Kruskal result: {stats.kruskal(zone1_data, zone2_data, zone3_data)}")
        print()
        print(f'Kruskal testing {zone1} and {zone2} zones:')
        print(f"Kruskal result: {stats.kruskal(zone1_data, zone2_data)}")
        print()
        print(f'Kruskal testing {zone1} and {zone3} zones:')
        print(f"Kruskal result: {stats.kruskal(zone1_data, zone3_data)}")
        print()
        print(f'Kruskal testing {zone2}, and {zone3} zones:')
        print(f"Kruskal result: {stats.kruskal(zone2_data, zone3_data)}")

    #return data

def linear_regression_plot(data, xvar, yvar):
    # Remove rows with missing or non-finite values
    data = data.dropna(subset=[xvar, yvar])
    data = data[np.isfinite(data[xvar])]
    data = data[np.isfinite(data[yvar])]

    sns.set(style="whitegrid")
    sns.regplot(x=xvar, y=yvar, data=data)

    plt.xlabel('Vertical Height (m)', fontsize=20, color='black')

    if yvar == 'num_cells_per_ug_protein':
      plt.ylabel('Cells/ug Animal Protein', fontsize=20, color='black')
      #plt.title("Linear Regression Plot")

    elif yvar == 'ng_chlorophyll_per_ug_protein':
      plt.ylabel('ng Chlorophyll \n  per Animal Protein', fontsize=17, color='black')
      #plt.title("Linear Regression Plot")

    elif yvar == 'ng_chlorophyll_per_hundred_cells':
      plt.ylabel('ng Chlorophyll \n  per 100 Cells', fontsize=17, color='black')
      #plt.title("Linear Regression Plot")


    x = data[xvar]
    y = data[yvar]

    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    regression_stats = {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err
    }

    plt.show()

    return regression_stats

def regression(your_data, xvar, yvar, title):

  if your_data[xvar].dtype != np.float64:
     your_data[xvar] = your_data[xvar].astype(np.float64)
  if your_data[yvar].dtype != np.float64:
     your_data[yvar] = your_data[yvar].astype(np.float64)

  sns.regplot(x=your_data[xvar], y=your_data[yvar], data=your_data)

  if yvar == 'num_cells_per_ug_protein':
      plt.ylabel('Cells/ug Animal Protein', fontsize=15)
  else:
      plt.ylabel('ng Chlorophyll per Animal Protein', fontsize=15)


  #plt.xlabel(x_label, fontsize=15, rotation= 0, va="bottom", labelpad=20)


  plt.title(title, fontsize=20)

  your_data.dropna(subset=[xvar, yvar], inplace=True)

  slope, intercept, r_value, p_value, std_err = linregress(x=your_data[xvar], y=your_data[yvar])

  regression_stats = {
  'slope': slope,
  'intercept': intercept,
  'r_value': r_value,
  'p_value': p_value,
  'std_err': std_err
      }


  if p_value < 0.05:
    p_text = 'p < 0.05'
  else:
    p_text = f'p = {p_value:.2f}'

  legend_text = (
                f'$R^2$ = {r_value:.2f}\n'
                f'{p_text}\n'
                f'SE = {std_err:.2f}'
                )

  plt.legend().remove()  # Remove the default legend
  plt.text(0.02, 0.98, legend_text, transform=plt.gca().transAxes, fontsize=12,
          verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

  return(regression_stats)

def multi_regression(data_list, xvar, yvar, label_keywords):
    regression_stats_list = []

    for i, data in enumerate(data_list):
        if data[xvar].dtype != np.float64:
            data[xvar] = data[xvar].astype(np.float64)
        if data[yvar].dtype != np.float64:
            data[yvar] = data[yvar].astype(np.float64)

        label = label_keywords.get(i, f'Dataset {i+1}')

        sns.regplot(x=xvar, y=yvar, data=data, label=label, color=colors[i])

        combined_data = data.dropna(subset=[xvar, yvar])
        slope, intercept, r_value, p_value, std_err = linregress(x=combined_data[xvar], y=combined_data[yvar])

        regression_stats = {
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_value,
            'std_err': std_err
        }
        regression_stats_list.append(regression_stats)

        label_text = f'{label} Regression Statistics:'
        print(label_text)
        print('-' * len(label_text))
        print(f'Slope: {slope}')
        print(f'Intercept: {intercept}')
        print(f'R-value: {r_value}')
        print(f'P-value: {p_value}')
        print(f'Standard Error: {std_err}')
        print()

    if yvar == 'num_cells_per_ug_protein':
        plt.ylabel('Cells/ug Animal Protein', fontsize=15)
    else:
        plt.ylabel('ng Chlorophyll per Animal Protein', fontsize=15)

    if xvar == 'temp(c)':
        plt.xlabel('Temp (c)', fontsize=15, rotation=0, va="bottom", labelpad=20)
    else:
        plt.xlabel('Salinity', fontsize=15, rotation=0, va="bottom", labelpad=20)

    plt.legend()
    plt.show()

    #return regression_stats_list

def merged_plot(your_data1, xvar, yvar1, your_data2, yvar2):
  start_date = '2022-08-01T00:00:00'
  end_date = '2023-03-31T00:00:00'
  f, (ax) = plt.subplots(figsize=(12,3.8))

  if yvar1 == 'num_cells_per_ug_protein':
      label1 = 'Algal Density'
  elif yvar1 == 'ng_chlorophyll_per_ug_protein':
      label1 = 'Chlorophyll Concentration'
  elif yvar1 == 'ng_chlorophyll_per_hundred_cells':
      label1 = 'ng Chlorophyll per 100 Cells'
  else:
      label1 = 'Algal Density'

  if yvar2 == 'temp(c)':
      label2 = 'Temperature'
  elif yvar2 == 'salinity(psu)' or 'salinity(ppt)':
      label2 = 'Salinity'
  else:
      label2 = 'skrt'

  ax.scatter(your_data1[xvar],your_data1[yvar1], color = 'royalblue', s =10, zorder = 3, label=label1)
  #ax.set(xlabel = "Date", ylabel='Num Cells')
  #ax.set_title('Algal Density and Star Oddi Temp', fontsize =25)
  ax.tick_params(axis='x', labelsize=11, rotation= 15, labelbottom=True, direction= 'out', pad=10)
  ax.tick_params(axis='y', labelsize=12)
  ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
  #ax.xaxis.set_major_formatter(DateFormatter("%m-%d-%y"))
  ax.xaxis.set_major_formatter(DateFormatter("%m-%d-%Y"))
  ax.set_xlim(start_date, end_date)
  #ax.grid(False)

  if yvar1 == 'num_cells_per_ug_protein':
      ax.set_ylabel('Cells/ug Animal Protein', fontsize=15)
      #ax.set_title('Merged algal and Fort Point salinity data overlayed', fontsize=20)

  if yvar1 == 'ng_chlorophyll_per_ug_protein':
      ax.set_ylabel('ng Chlorophyll per Animal Protein', fontsize=15)
      ax.set_title('Chlorophyll α with Fort Point Salinity Overlayed', fontsize=20)

  if yvar1 == 'ng_chlorophyll_per_hundred_cells':
      ax.set_ylabel('ng Chlorophyll per 100 Cells', fontsize=15)

  ax2 = ax.twinx() #to plot a second y axis
  ax2.scatter(your_data2[xvar], your_data2[yvar2], color = 'orange', label = label2, s = 15)
  #ax2.set(ylabel='Rainfall (mm)')
  ax2.tick_params(axis='y', labelsize=12)
  ax2.yaxis.label.set_size(20)
  ax2.set_ylim(5, 35)
  ax2.grid(False)

  if yvar2 == 'temp(c)':
      plt.ylabel('Temp (c)', fontsize=15, rotation= 270, va="bottom")

  if yvar2 == 'salinity(psu)' or 'salinity(ppt)':
      plt.ylabel('Salinity', fontsize=15, rotation= 270, va="bottom")

  lines, labels = ax.get_legend_handles_labels()
  lines2, labels2 = ax2.get_legend_handles_labels()
  ax2.legend(lines + lines2, labels + labels2, loc='best', fontsize = 9)


def kruskal_drops(your_data, yvar, zone):

  pre_drop_date_start = '2022-07-01T12:00:00'
  pre_drop_date_end = '2022-11-10T12:00:00'
  pre_drop = your_data[(your_data['date_time'] > pre_drop_date_start) & (your_data['date_time'] <= pre_drop_date_end)]

  post_drop_date_start = '2022-11-11T12:00:00'
  post_drop_date_end = '2023-04-06T12:00:00'
  post_drop = your_data[(your_data['date_time'] > post_drop_date_start) & (your_data['date_time'] <= post_drop_date_end)]

  print(f'Testing pre {stats.shapiro(pre_drop[yvar])} vs post {stats.shapiro(post_drop[yvar])} drop normalcy')

  if zone == 'all':
    print(f'Testing all intertidal zones before vs after drop:', stats.kruskal(pre_drop[yvar], post_drop[yvar]))
    #display(post_drop)
  else:
    pre_drop = pre_drop[pre_drop['intertidal_zone'] == zone]
    post_drop = post_drop[post_drop['intertidal_zone'] == zone]
    print(f'Testing {zone} intertidal zone before vs after drop:', stats.kruskal(pre_drop[yvar], post_drop[yvar]))
    #display(post_drop)

def get_y_label(yvar):
    if yvar == 'num_cells_per_ug_protein':
        return 'Cells/ug Animal Protein'
    elif yvar == 'ng_chlorophyll_per_ug_protein':
        return 'ng Chlorophyll per Animal Protein'
    elif yvar == 'ng_chlorophyll_per_hundred_cells':
        return 'ng Chlorophyll per 100 Cells'

def get_title(yvar):
    if yvar == 'num_cells_per_ug_protein':
        return 'Tidal Zone on Algal Density'
    elif yvar == 'ng_chlorophyll_per_ug_protein':
        return 'Tidal Zone on Chlorophyll α Production'
    elif yvar == 'ng_chlorophyll_per_hundred_cells':
        return 'Tidal Zone on Chlorophyll per Cell'