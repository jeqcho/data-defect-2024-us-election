# load the data from ../data/bias_correction.csv
# and plot trump_poll_corrected vs trump_share
# and color code it
# see figure_4.py for reference

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes  # Fix the linter error by importing Axes directly
import numpy as np
import os
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any
from matplotlib.lines import Line2D

# Set the current working directory to the script directory
script_dir: str = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Ensure figures directory exists
os.makedirs("../figures", exist_ok=True)

# Read the bias correction data
bias_correction_df: pd.DataFrame = pd.read_csv("../data/bias_correction.csv")

# Helper function to create scatter plots
def create_scatter(
    ax: Axes,  # Fix the linter error by using the directly imported Axes type
    x: pd.Series,  # Now x is actual vote share
    y: pd.Series,  # Now y is poll estimate
    title: str,
    xlabel: str,
    ylabel: str,
    states: pd.Series,
    state_classifications: pd.Series,
    candidate: str,  # Added candidate parameter to customize labels
) -> None:
    """
    Create a scatter plot with annotations and correlation info

    Args:
        ax: Matplotlib axes object to plot on
        x: x-axis data (actual vote share)
        y: y-axis data (poll support)
        title: Plot title
        xlabel: x-axis label
        ylabel: y-axis label
        states: Series containing state names for annotations
        state_classifications: Series containing state classifications for coloring
        candidate: Candidate name (Trump or Harris)
    """
    # Define colors based on state classification
    colors = []
    for classification in state_classifications:
        if classification == "Swing":
            colors.append("green")
        elif classification in ["Blue", "Likely Blue"]:
            colors.append("blue")
        else:  # Red
            colors.append("red")
    
    # Create scatter plot with colors
    ax.scatter(x, y, alpha=0.7, c=colors, s=5)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Set up percentage ticks
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    # Set axis limits to full range
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add a diagonal line for reference - changed from red to black
    ax.plot([0, 1], [0, 1], "k--")
    
    # Add overestimated/underestimated labels
    ax.text(0.05, 0.95, f"Polls overestimated\n{candidate} support", transform=ax.transAxes, 
            fontsize=11, horizontalalignment='left', verticalalignment='top')
    ax.text(0.95, 0.05, f"Polls underestimated\n{candidate} support", transform=ax.transAxes, 
            fontsize=11, horizontalalignment='right', verticalalignment='bottom')

    # Calculate and display Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((x - y) ** 2))
    # Position the RMSE text below the x-axis title
    ax.text(0.95, -0.25, f"RMSE: {rmse:.2f}", 
            transform=ax.transAxes,
            fontsize=11, horizontalalignment='right')

    # Add gridlines
    ax.grid(True, linestyle="--", alpha=0.6)
    

# %%
# Create a single plot
fig, ax = plt.subplots(figsize=(5, 4))

create_scatter(
    ax,  # Pass the single axes object directly
    bias_correction_df["trump_share"],  # x-axis is now the actual vote share (including third parties)
    bias_correction_df["trump_poll_corrected"],  # y-axis is now the poll estimate
    "Trump: Bias-Corrected Raw Poll Estimate vs. Actual Vote Share",
    "Final Trump Popular Vote Share",
    "Bias-Corrected Raw Poll Estimate,\nTrump Support",
    bias_correction_df["state"],
    bias_correction_df["Pre-Election Classification"],
    "Trump"
)
# Save the plot
plt.tight_layout()
plt.savefig("../figures/bias_correction_plot.png", dpi=300, bbox_inches="tight")
print("Plot saved to ../figures/bias_correction_plot.png and .pdf")