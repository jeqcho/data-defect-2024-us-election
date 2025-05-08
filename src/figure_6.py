# the goal of this file is to plot two plots
# input: ../data/figure_6.csv
# input: ../data/State-Pre-ElectionClassification.csv

# you, the AI assistant, should read the input files to understand the columns and the data types

# task:
# two candidates. one plot each.
# y-axis is log_10 Z_n_N, x-axis is log_10 of total_votes
# plot the regression as well for log Z=a+b log votes
# report b and its standard error in a box in the form of [b (std err)], use two sig figs
# use the classification CSV to color the dots, so red, blue, green. Green is swing states.
# inside plot, top of grid has text "Less accurate" and bottom has text "More accurate"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.patches as mpatches
import os
from decimal import Decimal, ROUND_HALF_UP

# Set the current working directory to the script directory
script_dir: str = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Read the data files
figure_data = pd.read_csv('../data/figure_6.csv')
classification = pd.read_csv('../data/State-Pre-ElectionClassification.csv')

# Merge the datasets
data = pd.merge(figure_data, classification, how='left', left_on='state', right_on='State')

# Create a color map for the states based on classification
color_map = {
    'Blue': 'blue',
    'Likely Blue': 'blue',
    'Red': 'red',
    'Swing': 'green'
}

# Map colors to each state
data['color'] = data['Pre-Election Classification'].map(color_map)

# Calculate logarithms
data['log_total_votes'] = np.log10(data['total_votes'])
data['log_trump_Z_n_N'] = np.log10(np.abs(data['trump_Z_n_N']))
data['log_harris_Z_n_N'] = np.log10(np.abs(data['harris_Z_n_N']))

# Set up the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Helper function to create each plot with regression
def create_plot(ax, x, y, colors, candidate_name):
    # Scatter plot
    ax.scatter(x, y, c=colors, s=30)
    
    # Linear regression
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask].values.reshape(-1, 1)
    y_clean = y[mask].values.reshape(-1, 1)
    
    reg = LinearRegression().fit(x_clean, y_clean)
    slope = reg.coef_[0][0]
    
    # Calculate standard error
    y_pred = reg.predict(x_clean)
    n = len(x_clean)
    se = np.sqrt(np.sum((y_clean - y_pred)**2) / (n - 2) / np.sum((x_clean - np.mean(x_clean))**2))
    
    # Plot regression line
    x_line = np.array([min(x_clean), max(x_clean)])
    y_line = reg.predict(x_line.reshape(-1, 1))
    ax.plot(x_line, y_line, 'gray', linewidth=2)
    
    # Format slope and standard error to 2 significant figures
    # Format to exactly 2 sig figs
    def format_to_2sig(value):
        # Use the %g format with precision=2 to get 2 significant digits
        formatted = f"{value:.2g}"
        
        # Check if we need to pad with zeros
        if 'e' not in formatted and '.' in formatted:
            int_part, dec_part = formatted.split('.')
            # If we have a single digit before the decimal, we need one decimal place
            if len(int_part) == 1 and int_part != '0':
                if len(dec_part) < 1:
                    formatted += '0'
            # If we have a decimal starting with 0, we need two decimal places
            elif int_part == '0':
                if len(dec_part) < 2:
                    formatted += '0' * (2 - len(dec_part))
        return formatted
    
    slope_text = f"{format_to_2sig(slope)}\n({format_to_2sig(se)})"
    
    # Add textbox with slope and standard error
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.95, 0.35, f"{slope_text}", transform=ax.transAxes, 
            verticalalignment='center', horizontalalignment='right', bbox=props)
    
    # Add "Less/More accurate" labels
    ax.text(0.05, 0.95, "Less accurate", transform=ax.transAxes, 
            horizontalalignment='left', fontsize=12)
    ax.text(0.05, 0.05, "More accurate", transform=ax.transAxes, 
            horizontalalignment='left', fontsize=12)
    
    # Add labels and title
    ax.set_xlabel('log$_{10}$ (Total Voters)')
    ax.set_ylabel(f'{candidate_name}' + ' log$_{10}$ |Z$_{n,N}$|')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return slope, se

# Create the plots
harris_slope, harris_se = create_plot(ax1, data['log_total_votes'], data['log_harris_Z_n_N'], 
                                   data['color'], 'Harris')
trump_slope, trump_se = create_plot(ax2, data['log_total_votes'], data['log_trump_Z_n_N'], 
                                  data['color'], 'Trump')

# Set y-axis limits for both plots
ax1.set_ylim(-2.2, 2.2)
ax2.set_ylim(-2.2, 2.2)

# Set specific y-axis ticks for both plots
y_ticks = [-2, -1, 0, 1, 2]
ax1.set_yticks(y_ticks)
ax2.set_yticks(y_ticks)

# Set specific x-axis ticks for both plots
x_ticks = [5.5, 6.0, 6.5, 7.0]
ax1.set_xticks(x_ticks)
ax2.set_xticks(x_ticks)

plt.tight_layout()

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)
plt.savefig('../figures/figure_6.png', dpi=300, bbox_inches='tight')
