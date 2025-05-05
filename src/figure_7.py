# the goal of this file is to compute a metric and plot it in two figures
# input: ../data/merged_all_voters.csv
# input: ../data/State-Pre-ElectionClassification.csv
# input: ../data/state_abbr.csv

# you, the AI assistant, should read the input files to understand the columns and the data types

# you will plot two figures, one for each candidate
# each row is a state
# compute Z_n for each row. Z_n=(phat-p)/sqrt(phat(phat-p)/n) where phat is the poll_share, p is the candidate_share, n is the number of respondents

# then plot it for harris then trump
# x-axis is log 10 total_votes
# y-axis is Z_n

# color code the states using State-Pre-ElectionClassification.csv. blue, red, green. Green is for swing states.
# use state_abbr.csv to label the dots with two letters each. The letters (and the line pointing to it, if any) should also be colored

# shade the y-axis range -2 to 2 in gray

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from matplotlib.ticker import FixedLocator
from adjustText import adjust_text  # Import the adjust_text method

# Set the current working directory to the script directory
script_dir: str = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def plot_z_scores(data, candidate, ax):
    """
    Plot Z-scores for a specific candidate on the given axis.

    Parameters:
    - data: DataFrame containing the data
    - candidate: String, either 'harris' or 'trump'
    - ax: Matplotlib axis to plot on
    """
    z_score_col = f"{candidate}_Z_n"
    poll_col = f"{candidate}_poll_all"
    share_col = f"{candidate}_share"

    # Plot the data points
    points = []
    
    for _, row in data.iterrows():
        x_pos = math.log10(row["total_votes"])
        y_pos = row[z_score_col]
        
        # Add scatter points
        ax.scatter(x_pos, y_pos, color=row["color"], s=50)
        
        # Create annotation with arrow
        ax.annotate(
            row["state_abbr"],
            xy=(x_pos, y_pos),
            xytext=(x_pos, y_pos + 0.5),  # Offset text slightly
            color=row["color"],
            fontsize=8,
            fontweight='bold',
            ha='center',
            va='center',
            arrowprops=dict(
                arrowstyle='->',
                color=row["color"],
                lw=0.5,
                shrinkA=5,  # Increase shrinkA to prevent arrows from striking through text
                shrinkB=5
            )
        )

    # Shade the y-axis range -2 to 2 in gray
    ax.axhspan(-2, 2, alpha=0.2, color="gray")

    # Set labels and title
    ax.set_xlabel("Log10 Total Votes")
    ax.set_ylabel("Z_n")
    ax.set_title(f"{candidate.capitalize()} Z-scores by State")
    ax.grid(True, alpha=0.3)

    # Set y-axis limits from -15 to 5
    ax.set_ylim(-17, 7)

    # Set custom y-axis ticks at -10, -5, -2, 0, 2, 5
    ax.yaxis.set_major_locator(FixedLocator([-10, -5, -2, 0, 2, 5]))
    # Format the tick labels
    ax.set_yticklabels(['-10', '-5', '-2', '0', '2', '5'])

    # Set custom x-axis ticks at 5.5, 6, 6.5, 7
    ax.xaxis.set_major_locator(FixedLocator([5.5, 6, 6.5, 7]))
    # Format the x-axis tick labels
    ax.set_xticklabels(['5.5', '6.0', '6.5', '7.0'])
    # Set x-axis limits to match the ticks
    ax.set_xlim(5.3, 7.4)


def main():
    # Read the input files
    merged_data = pd.read_csv("../data/merged_all_voters.csv")
    state_abbr = pd.read_csv("../data/state_abbr.csv")

    # Merge state abbreviations with the main data
    merged_data = pd.merge(merged_data, state_abbr, left_on="state", right_on="state")

    # Compute Z_n for Harris
    merged_data["harris_Z_n"] = (
        merged_data["harris_poll_all"] - merged_data["harris_share"]
    ) / np.sqrt(
        merged_data["harris_poll_all"]
        * (1 - merged_data["harris_poll_all"])
        / merged_data["num_respondents_all"]
    )

    # Compute Z_n for Trump
    merged_data["trump_Z_n"] = (
        merged_data["trump_poll_all"] - merged_data["trump_share"]
    ) / np.sqrt(
        merged_data["trump_poll_all"]
        * (1 - merged_data["trump_poll_all"])
        / merged_data["num_respondents_all"]
    )

    # Define color mapping for state classification
    color_map = {"Blue": "blue", "Likely Blue": "blue", "Red": "red", "Swing": "green"}

    # Convert the classification to colors
    print(merged_data.columns)
    merged_data["color"] = merged_data["Pre-Election Classification"].map(color_map)

    # Create figure and subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot for Harris on the left
    plot_z_scores(merged_data, "harris", axes[0])

    # Plot for Trump on the right
    plot_z_scores(merged_data, "trump", axes[1])

    plt.tight_layout()
    plt.savefig("../figures/figure_7.png", dpi=300)


if __name__ == "__main__":
    main()
