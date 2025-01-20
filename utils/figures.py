import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

def create_plot(plot_type, data_dict, title, xlabel, ylabel, xlim, ylim=None, save_path=None):
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