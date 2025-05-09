# this script reads ../data/figure_5.csv and plots two histograms. One for sample_size and one for sample_ratio

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set the current working directory to the script directory
script_dir: str = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Read the data
data_path = os.path.join('..', 'data', 'figure_5.csv')
df = pd.read_csv(data_path)

# Read the state classification data
classification_path = os.path.join('..', 'data', 'State-Pre-ElectionClassification.csv')
classification_df = pd.read_csv(classification_path)

# Merge the datasets
merged_df = pd.merge(df, classification_df, left_on='state', right_on='State', how='left')

# Create a color map
color_map = {
    'Swing': 'green',
    'Blue': 'blue',
    'Likely Blue': 'blue',
    'Red': 'red'
}

# Create figure
plt.figure(figsize=(6, 3))

# Create scatterplot with color coding
for classification, group in merged_df.groupby('Pre-Election Classification'):
    classification_str = str(classification)  # Convert to string to ensure type safety
    plt.scatter(
        group['sample_size'], 
        group['sample_ratio'], 
        color=color_map.get(classification_str, 'gray'),
        edgecolors='none',
        s=20  # slightly larger point size
    )


# Set log scale for x-axis (sample_size)
plt.xscale('log')

# Set axis labels
plt.xlabel('Sample Size')
plt.ylabel('Sample Ratio')

# Add grid for better readability but make it subtle
plt.grid(True, alpha=0.3, linestyle='--')

# Set clean background
plt.gca().set_facecolor('white')

plt.ylim(0,0.0007)

# Adjust layout
plt.tight_layout()

# Save the plot
output_dir = os.path.join('..', 'figures')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'sample_size_ratio_scatterplot.png'), dpi=300, bbox_inches='tight')

print("Scatterplot created and saved to ../figures/sample_size_ratio_scatterplot.png")