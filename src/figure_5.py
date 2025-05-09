# the goal of this script is to plot a figure called figure 5
# input: ../data/figure_5.csv

# read the dataset first to see for yourself and understand the columns and look at the rows to understand the data types

# the figure has two plots, one for each candidate. The candidates are Harris and Trump

# for each candidate, plot the histogram of the data defect correlation.
# include a small box at the top right corner which shows the mean \pm 2 std error (so you should compute the mean and the std error)

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

# Set the current working directory to the script directory
script_dir: str = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Read the dataset
suffix = "_likely"
# suffix = ""
df = pd.read_csv(f"../data/figure_5{suffix}.csv")

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(9, 2.5))

# Calculate statistics for each candidate
harris_data = df["harris_data_defect_correlation"]
trump_data = df["trump_data_defect_correlation"]

def to_2sf(i: float) -> str:
    """Convert a float to a string with 2 significant figures."""
    return f"{float(f'{i:.2g}'):g}"

def plot_histogram(ax, data: pd.Series, candidate: str) -> None:
    """
    Plot histogram for a candidate's data defect correlation.
    
    Args:
        ax: Matplotlib axis to plot on
        data: Series containing the data defect correlation values
        candidate: Name of the candidate (Harris or Trump)
    """
    # Calculate statistics
    mean = data.mean()
    std_err = stats.sem(data)  # Standard Error of the Mean
    
    # Set up axes
    ax.set_xlabel(f"{candidate} $\\hat{{\\rho}}_N$")
    ax.set_ylabel("Count")
    ax.set_xlim([-0.01, 0.01])
    ax.set_ylim([0, 12])
    ax.set_xticks([-0.010, -0.005, 0, 0.005, 0.010])
    ax.set_xticklabels(["-0.010", "-0.005", "0", "0.005", "0.010"])
    ax.set_yticks([0, 4, 8, 12])
    ax.grid(True, alpha=0.3, zorder=0)  # Set grid with low zorder
    
    # Plot histogram with higher zorder to appear on top of grid
    ax.hist(data, bins=15, color="gray", zorder=3)
    ax.axvline(x=0, color="black", linestyle=":", alpha=0.7, zorder=2)
    
    # Add text box with mean ± std_err
    props = dict(boxstyle="round", facecolor="white", alpha=0.7)
    mean_rounded = to_2sf(mean)
    std_err_rounded = to_2sf(std_err)
    
    textstr = f"{mean_rounded} ± {std_err_rounded}"
    ax.text(
        0.95,
        0.90,
        textstr,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )

# Plot histograms for both candidates
plot_histogram(axes[0], harris_data, "Harris")
plot_histogram(axes[1], trump_data, "Trump")

# Adjust layout and save figure
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig(f"../figures/figure_5{suffix}.png", dpi=300, bbox_inches="tight")
