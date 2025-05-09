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

# Customize
# total_votes_or_sample_size = "total_votes"
total_votes_or_sample_size = "num_respondents_all"

# all_or_likely = "likely"
all_or_likely = "all"

# suffix = "_likely"
suffix = ""


def plot_z_scores(data, candidate, ax):
    """
    Plot Z-scores for a specific candidate on the given axis.

    Parameters:
    - data: DataFrame containing the data
    - candidate: String, either 'harris' or 'trump'
    - ax: Matplotlib axis to plot on
    """
    z_score_col = f"{candidate}_Z_n"

    # Filter blue and red states
    blue_states = data[data["color"] == "blue"].copy()
    red_states = data[data["color"] == "red"].copy()
    
    # Sort by total_votes to find top 3 and bottom 3 for each color
    blue_largest = blue_states.nlargest(3, total_votes_or_sample_size)
    blue_smallest = blue_states.nsmallest(3, total_votes_or_sample_size)
    red_largest = red_states.nlargest(3, total_votes_or_sample_size)
    red_smallest = red_states.nsmallest(3, total_votes_or_sample_size)
    
    # Combine states to label
    states_to_label = pd.concat([blue_largest, blue_smallest, red_largest, red_smallest])
    
    # Plot the data points
    texts = []
    for _, row in data.iterrows():
        x_pos = math.log10(row[total_votes_or_sample_size])
        y_pos = row[z_score_col]

        # Add scatter points
        ax.scatter(x_pos, y_pos, color=row["color"], s=10)

        # Only add text for states in our filtered list
        if row["state_abbr"] in states_to_label["state_abbr"].values:
            texts.append(
                ax.annotate(
                    row["state_abbr"],
                    xy=(x_pos, y_pos),
                    color=row["color"],
                    size=8,
                    arrowprops=dict(
                        arrowstyle="-",
                        color=row["color"],
                        lw=0.5,
                    ),
                )
            )
    

    # Shade the y-axis range -2 to 2 in gray
    ax.axhspan(-2, 2, alpha=0.2, color="gray")

    # Set labels and title
    if total_votes_or_sample_size == "total_votes":
        ax.set_xlabel("$\log_{10}$(Total Voters)")
    else:
        ax.set_xlabel("$\log_{10}$(Sample Size)")
    ax.set_ylabel(f"{candidate.capitalize()} " + "${Z_n}$")
    ax.grid(True, alpha=0.3)

    # Set y-axis limits from -15 to 5
    ax.set_ylim(-20, 7)

    # Set custom y-axis ticks at -10, -5, -2, 0, 2, 5
    ax.yaxis.set_major_locator(FixedLocator([-10, -5, -2, 0, 2, 5]))
    # Format the tick labels
    ax.set_yticklabels(["-10", "-5", "-2", "0", "2", "5"])

    # Set custom x-axis ticks at 5.5, 6, 6.5, 7
    if total_votes_or_sample_size == "total_votes":
        ax.xaxis.set_major_locator(FixedLocator([5.5, 6, 6.5, 7]))
        # Format the x-axis tick labels
        ax.set_xticklabels(["5.5", "6.0", "6.5", "7.0"])
        # Set x-axis limits to match the ticks
        ax.set_xlim(5.3, 7.4)
    else:
        ax.xaxis.set_major_locator(FixedLocator([2.0, 2.5, 3.0, 3.5]))
        # Format the x-axis tick labels
        ax.set_xticklabels(["2.0", "2.5", "3.0", "3.5"])
        # Set x-axis limits to match the ticks
        ax.set_xlim(1.6, 3.9)
    
    x = [math.log10(row[total_votes_or_sample_size]) for _,row in data.iterrows()]
    y = [row[z_score_col] for _,row in data.iterrows()]
    adjust_text(
        texts,
        objects=texts,
        x=x,
        y=y,
        ax=ax,
        expand=(2, 2),
    )


def main():
    # Read the input files
    merged_data = pd.read_csv(f"../data/merged_{all_or_likely}_voters.csv")
    state_abbr = pd.read_csv("../data/state_abbr.csv")

    # Merge state abbreviations with the main data
    merged_data = pd.merge(merged_data, state_abbr, left_on="state", right_on="state")

    # Compute Z_n for Harris
    merged_data["harris_Z_n"] = (
        merged_data[f"harris_poll_{all_or_likely}"] - merged_data["harris_share"]
    ) / np.sqrt(
        merged_data[f"harris_poll_{all_or_likely}"]
        * (1 - merged_data[f"harris_poll_{all_or_likely}"])
        / merged_data[f"num_respondents_{all_or_likely}"]
    )

    # Compute Z_n for Trump
    merged_data["trump_Z_n"] = (
        merged_data[f"trump_poll_{all_or_likely}"] - merged_data["trump_share"]
    ) / np.sqrt(
        merged_data[f"trump_poll_{all_or_likely}"]
        * (1 - merged_data[f"trump_poll_{all_or_likely}"])
        / merged_data[f"num_respondents_{all_or_likely}"]
    )

    # Define color mapping for state classification
    color_map = {"Blue": "blue", "Likely Blue": "blue", "Red": "red", "Swing": "green"}

    # Convert the classification to colors
    print(merged_data.columns)
    merged_data["color"] = merged_data["Pre-Election Classification"].map(color_map)

    # Save data to CSV file
    output_columns = [
        "state",
        "state_abbr",
        "harris_Z_n",
        "trump_Z_n",
        "total_votes",
        f"harris_poll_{all_or_likely}",
        "harris_share",
        f"trump_poll_{all_or_likely}",
        "trump_share",
        f"num_respondents_{all_or_likely}",
        "Pre-Election Classification",
    ]
    merged_data[output_columns].to_csv(f"../data/figure_7{suffix}.csv", index=False)

    # Create figure and subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    # Plot for Harris on the left
    plot_z_scores(merged_data, "harris", axes[0])

    # Plot for Trump on the right
    plot_z_scores(merged_data, "trump", axes[1])

    plt.tight_layout()
    plt.savefig(f"../figures/figure_7{suffix}.png", dpi=300)


if __name__ == "__main__":
    main()
