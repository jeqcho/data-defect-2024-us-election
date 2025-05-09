# the goal of this file is to plot two figures
# input: ../data/figure_5.csv
# input: ../data/State-Pre-ElectionClassification.csv

# you, the AI assistant, should read the input files to understand the columns and the data types

# you will plot two figures, one for each candidate

# then plot the data defect correlation against total votes
# plot one for harris then trump
# x-axis is log 10 total_votes
# y-axis is data defect correlation

# color code the states using State-Pre-ElectionClassification.csv. blue, red, green. Green is for swing states.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set the current working directory to the script directory
script_dir: str = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def load_data():
    """Load and prepare the data."""
    # Read input data
    figure_5_data = pd.read_csv('../data/figure_5_likely.csv')
    state_classification = pd.read_csv('../data/State-Pre-ElectionClassification.csv')

    # Merge the datasets
    merged_data = pd.merge(
        figure_5_data, 
        state_classification, 
        how='left', 
        left_on='state', 
        right_on='State'
    )

    # Define color mapping for state classifications
    color_map = {
        'Blue': 'blue',
        'Likely Blue': 'blue',
        'Red': 'red',
        'Swing': 'green'
    }

    # Create a new column for colors based on state classification
    merged_data['color'] = merged_data['Pre-Election Classification'].map(color_map)
    
    return merged_data


def create_subplot(ax, data, x_transform, candidate, x_label):
    """Create a scatter plot for a candidate with the specified x-axis transformation."""
    correlation_column = f'{candidate.lower()}_data_defect_correlation'
    
    for idx, row in data.iterrows():
        x_value = x_transform(row['total_votes'])
        y_value = row[correlation_column]
        
        ax.scatter(
            x_value, 
            y_value, 
            color=row['color'], 
            s=70, 
            alpha=0.7
        )
        ax.annotate(row['state'], (x_value, y_value), fontsize=8)
    
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel('Data Defect Correlation', fontsize=14)
    ax.set_title(candidate, fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits from -0.01 to 0.01
    ax.set_ylim(-0.01, 0.01)


def create_figure(data, x_transform, x_label, title, filename):
    """Create a figure with subplots for both candidates using the specified x-axis transformation."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    
    # Create subplot for Harris
    create_subplot(axes[0], data, x_transform, 'Harris', x_label)
    
    # Create subplot for Trump
    create_subplot(axes[1], data, x_transform, 'Trump', x_label)
    
    # Add common legend
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Blue States')
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Red States')
    green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Swing States')
    
    fig.legend(handles=[blue_patch, red_patch, green_patch], loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01))
    
    fig.suptitle(title, fontsize=18)
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    
    # Create output directory if it doesn't exist
    os.makedirs('../figures', exist_ok=True)
    plt.savefig(f'../figures/{filename}.png', dpi=300)


def main():
    """Main function to create all figures."""
    data = load_data()
    
    # Create log10(N) plot
    create_figure(
        data=data,
        x_transform=lambda x: np.log10(x),
        x_label='Log10 Total Votes',
        title='Data Defect Correlation vs Log10(Total Votes)',
        filename='rho_logN'
    )
    
    # Create sqrt(N) plot
    create_figure(
        data=data,
        x_transform=lambda x: np.sqrt(x),
        x_label='sqrt(Total Votes)',
        title='Data Defect Correlation vs sqrt(Total Votes)',
        filename='rho_N'
    )
    
    print("Figures generated and saved to ../figures/ directory")


if __name__ == "__main__":
    main()