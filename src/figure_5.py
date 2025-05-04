# the goal of this script is to plot a figure called figure 5
# input: ../data/figure_5.csv

# read the dataset first to see for yourself and understand the columns and look at the rows to understand the data types

# the figure has two plots, one for each candidate. The candidates are Harris and Trump

# for each candidate, plot the histogram of the data defect correlation.
# include a small box at the top right corner which shows the mean \pm 2 std error (so you should compute the mean and the std error)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from scipy import stats
import os

# Set the current working directory to the script directory
script_dir: str = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Read the dataset
df = pd.read_csv("../data/figure_5.csv")

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Figure 5: Data Defect Correlation Histograms", fontsize=16)

# Calculate statistics for Harris
harris_data = df["harris_data_defect_correlation"]
harris_mean = harris_data.mean()
harris_std_err = stats.sem(harris_data)  # Standard Error of the Mean

# Calculate statistics for Trump
trump_data = df["trump_data_defect_correlation"]
trump_mean = trump_data.mean()
trump_std_err = stats.sem(trump_data)  # Standard Error of the Mean

# Set up axes for Harris histogram (first subplot)
axes[0].set_xlabel(r"Harris $\hat{\rho}_N$")
axes[0].set_ylabel("Count")
axes[0].set_xlim([-0.01, 0.01])
axes[0].set_xticks([-0.010, -0.005, 0, 0.005, 0.010])
axes[0].set_xticklabels(["-0.010", "-0.005", "0", "0.005", "0.010"])
axes[0].grid(True, alpha=0.3, zorder=0)  # Set grid with low zorder
# Plot Harris histogram with higher zorder to appear on top of grid
axes[0].hist(harris_data, bins=15, color="gray", zorder=3)
axes[0].axvline(x=0, color="black", linestyle=":", alpha=0.7, zorder=2)

# Set up axes for Trump histogram (second subplot)
axes[1].set_xlabel(r"Trump $\hat{\rho}_N$")
axes[1].set_ylabel("Count")
axes[1].set_xlim([-0.01, 0.01])
axes[1].set_xticks([-0.010, -0.005, 0, 0.005, 0.010])
axes[1].set_xticklabels(["-0.010", "-0.005", "0", "0.005", "0.010"])
axes[1].grid(True, alpha=0.3, zorder=0)  # Set grid with low zorder
# Plot Trump histogram with higher zorder to appear on top of grid
axes[1].hist(trump_data, bins=15, color="gray", zorder=3)
axes[1].axvline(x=0, color="black", linestyle=":", alpha=0.7, zorder=2)


def to_2sf(i: float) -> str:
    return f"{float(f'{i:.2g}'):g}"


# Add text boxes with mean ± 2*std_err (2 significant figures but 4 decimal places)
# For Harris
props = dict(boxstyle="round", facecolor="white", alpha=0.7)
harris_mean_rounded = to_2sf(harris_mean)
harris_std_err_rounded = to_2sf(harris_std_err)

textstr = f"{harris_mean_rounded} ± {harris_std_err_rounded}"
axes[0].text(
    0.95,
    0.95,
    textstr,
    transform=axes[0].transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=props,
)

# For Trump
trump_mean_rounded = to_2sf(trump_mean)
trump_std_err_rounded = to_2sf(trump_std_err)

textstr = f"{trump_mean_rounded} ± {trump_std_err_rounded}"
axes[1].text(
    0.95,
    0.95,
    textstr,
    transform=axes[1].transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=props,
)

# Adjust layout and save figure
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("../figures/figure_5.png", dpi=300, bbox_inches="tight")
