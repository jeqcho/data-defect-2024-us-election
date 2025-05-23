# this script is basically figure_4.py
# but those who chose "I am not sure" which is option 5 actually votes for trump

# the goal of this file is to generate 2 plots for Trump.
# we plot the actual popularity of Trump to the polled popularity of Trump
# it is a scatter plot, where each dot is a state
# the actual popularity of Trump is the actual vote share of Trump (including third parties).
# this data is at ../data/2024_us_election_results_by_state.csv
# the polled popularity of Trump is from ../data/CES24_Common.csv (pre-election data only)

# The plots differ by their y-axis:
# 1. The first plot uses all pre-election respondents (CC24_364b)
# 2. The second plot uses only likely voters (those who said they intend to vote in CC24_363)

# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes  # Fix the linter error by importing Axes directly
import numpy as np
import os
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

# Set the current working directory to the script directory
script_dir: str = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Ensure figures directory exists
os.makedirs("../figures", exist_ok=True)

# Read the actual election results data
election_results_path: str = "../data/2024_us_election_results_by_state.csv"
election_df: pd.DataFrame = pd.read_csv(election_results_path)

# Read state classification data
classification_path: str = "../data/State-Pre-ElectionClassification.csv"
classification_df: pd.DataFrame = pd.read_csv(classification_path)

# Calculate the actual vote share for Trump (including third parties)
election_df["trump_share"] = election_df["trump_votes"] / election_df["total_votes"]

# Read the poll data
poll_path: str = "../data/CES24_Common.csv"
poll_df: pd.DataFrame = pd.read_csv(poll_path)

# Define response code mappings
# CC24_364b: 1 = Harris, 2 = Trump, 3 = Other, 4 = Won't vote, 5 = Not sure
# CC24_363: 1 = Yes definitely, 2 = Probably, 3 = Already voted, 4 = Plan to vote, 5 = No, 6 = Undecided
likely_voter_codes: List[int] = [1, 2, 3, 4]  # Codes for respondents likely to vote

# Process the data
poll_df['is_likely_voter'] = poll_df['CC24_363'].isin(likely_voter_codes)

def get_trump_preference(code: Optional[Union[int, float]]) -> Optional[float]:
    """
    Convert preference code to numeric value for Trump
    
    Args:
        code: The preference response code
        
    Returns:
        1.0 if Trump preference (code=2) or Not sure (code=5), 0.0 if Harris preference (code=1), NaN otherwise
    """
    if pd.isna(code):
        return np.nan
    if code == 2 or code == 5:  # Trump or "Not sure" (counted as Trump)
        return 1.0
    return 0.0 # anything else

poll_df['trump_preference'] = poll_df['CC24_364b'].apply(get_trump_preference)

# Group by state to get poll percentages for the two scenarios
# 1. All respondents who expressed a preference in pre-election survey
all_respondents: pd.DataFrame = poll_df[~poll_df['trump_preference'].isna()]
state_polls_all: pd.DataFrame = all_respondents.groupby('inputstate').agg({
    'trump_preference': 'mean',
    'caseid': 'count'  # Count number of respondents per state
}).reset_index()

# 2. Only likely voters
likely_voters: pd.DataFrame = poll_df[poll_df['is_likely_voter'] & ~poll_df['trump_preference'].isna()]
state_polls_likely: pd.DataFrame = likely_voters.groupby('inputstate').agg({
    'trump_preference': 'mean',
    'caseid': 'count'  # Count number of respondents per state
}).reset_index()

# Rename columns for clarity
state_polls_all.rename(columns={
    'trump_preference': 'trump_poll_all',
    'caseid': 'num_respondents_all'
}, inplace=True)

state_polls_likely.rename(columns={
    'trump_preference': 'trump_poll_likely',
    'caseid': 'num_respondents_likely'
}, inplace=True)

# Create a mapping from FIPS state codes to state names
fips_to_state: Dict[int, str] = {
    1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 6: 'California',
    8: 'Colorado', 9: 'Connecticut', 10: 'Delaware', 11: 'District of Columbia',
    12: 'Florida', 13: 'Georgia', 15: 'Hawaii', 16: 'Idaho', 17: 'Illinois',
    18: 'Indiana', 19: 'Iowa', 20: 'Kansas', 21: 'Kentucky', 22: 'Louisiana',
    23: 'Maine', 24: 'Maryland', 25: 'Massachusetts', 26: 'Michigan',
    27: 'Minnesota', 28: 'Mississippi', 29: 'Missouri', 30: 'Montana',
    31: 'Nebraska', 32: 'Nevada', 33: 'New Hampshire', 34: 'New Jersey',
    35: 'New Mexico', 36: 'New York', 37: 'North Carolina', 38: 'North Dakota',
    39: 'Ohio', 40: 'Oklahoma', 41: 'Oregon', 42: 'Pennsylvania',
    44: 'Rhode Island', 45: 'South Carolina', 46: 'South Dakota',
    47: 'Tennessee', 48: 'Texas', 49: 'Utah', 50: 'Vermont', 51: 'Virginia',
    53: 'Washington', 54: 'West Virginia', 55: 'Wisconsin', 56: 'Wyoming'
}

# Apply FIPS state code mapping to convert numeric codes to state names
# Ensure inputstate is treated as int
state_polls_all['inputstate'] = state_polls_all['inputstate'].astype(int)
state_polls_likely['inputstate'] = state_polls_likely['inputstate'].astype(int)

# Map FIPS codes to state names
state_polls_all['state'] = state_polls_all['inputstate'].map(fips_to_state)
state_polls_likely['state'] = state_polls_likely['inputstate'].map(fips_to_state)

# Normalize case for state names to ensure proper matching
election_df['state'] = election_df['state'].str.title()
state_polls_all['state'] = state_polls_all['state'].str.title()
state_polls_likely['state'] = state_polls_likely['state'].str.title()

# Normalize case for state names
classification_df['State'] = classification_df['State'].str.strip('"').str.title()

# Merge election data with classification data
election_df = election_df.merge(classification_df, left_on='state', right_on='State', how='left')

# Now merge with poll data
merged_all = election_df.merge(state_polls_all, on='state', how='inner')
merged_likely = election_df.merge(state_polls_likely, on='state', how='inner')

# Save the merged DataFrames to CSV
merged_all.to_csv("../data/merged_all_voters_not_sure_is_trump.csv", index=False)
merged_likely.to_csv("../data/merged_likely_voters_not_sure_is_trump.csv", index=False)

# Scatter plot functions

# %%
# Create 2 plots for Trump only
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(wspace=0.3)

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
    ax.set_title(title, fontsize=14)
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
    ax.text(0.05, 0.95, "Polls overestimated\nTrump support", transform=ax.transAxes, 
            fontsize=11, horizontalalignment='left', verticalalignment='top')
    ax.text(0.95, 0.05, "Polls underestimated\nTrump support", transform=ax.transAxes, 
            fontsize=11, horizontalalignment='right', verticalalignment='bottom')

    # Calculate and display Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((x - y) ** 2))
    # Position the RMSE text below the x-axis title
    ax.text(0.95, -0.15, f"RMSE: {rmse:.2f}", 
            transform=ax.transAxes,
            fontsize=11, horizontalalignment='right')

    # Add gridlines
    ax.grid(True, linestyle="--", alpha=0.6)
    

# Trump plots
create_scatter(
    axes[0],
    merged_all["trump_share"],  # x-axis is now the actual vote share (including third parties)
    merged_all["trump_poll_all"],  # y-axis is now the poll estimate
    "Trump: Raw Poll Estimate vs. Actual Vote Share\n(Including 'Not Sure' as Trump)",
    "Final Trump Popular Vote Share",
    "Raw Poll Estimate,\nTrump Support",
    merged_all["state"],
    merged_all["Pre-Election Classification"]
)

create_scatter(
    axes[1],
    merged_likely["trump_share"],  # x-axis is now the actual vote share (including third parties)
    merged_likely["trump_poll_likely"],  # y-axis is now the poll estimate
    "Trump: Turnout-Adjusted Poll Estimate vs. Actual Vote Share\n(Including 'Not Sure' as Trump)",
    "Final Trump Popular Vote Share",
    "Turnout-Adjusted Poll Estimate,\nTrump Support",
    merged_likely["state"],
    merged_likely["Pre-Election Classification"]
)

plt.tight_layout()
plt.savefig("../figures/not_sure_is_trump.png", dpi=300, bbox_inches="tight")

print("Figure has been generated and saved to ../figures/not_sure_is_trump.png")

# %%