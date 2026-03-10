# Two-party vote share version of figure_4.py
# Both axes use only Harris vs Trump (excludes third parties, Other, Won't vote, Not sure)

# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import os
from typing import Dict, List, Optional, Union

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

# Calculate two-party vote share
two_party_total = election_df["harris_votes"] + election_df["trump_votes"]
election_df["trump_share"] = election_df["trump_votes"] / two_party_total
election_df["harris_share"] = election_df["harris_votes"] / two_party_total

# Read the poll data
poll_path: str = "../data/CCES24_Common_OUTPUT_vv_topost_final.csv"
poll_df: pd.DataFrame = pd.read_csv(poll_path)

# Define response code mappings
# CC24_364b: 1 = Harris, 2 = Trump, 3 = Other, 4 = Won't vote, 5 = Not sure
# CC24_363: 1 = Yes definitely, 2 = Probably, 3 = Already voted, 4 = Plan to vote, 5 = No, 6 = Undecided
likely_voter_codes: List[int] = [1, 2, 3, 4]  # Codes for respondents likely to vote

# Process the data
poll_df['is_likely_voter'] = poll_df['CC24_363'].isin(likely_voter_codes)
poll_df['is_validated_voter'] = (~poll_df['TS_g2024'].isna()) & (poll_df['TS_g2024'] < 7)

def get_harris_preference(code: Optional[Union[int, float]]) -> Optional[float]:
    if pd.isna(code):
        return np.nan
    if code == 1:  # Harris
        return 1.0
    if code == 2:  # Trump
        return 0.0
    return np.nan  # Exclude Other/Won't vote/Not sure

def get_trump_preference(code: Optional[Union[int, float]]) -> Optional[float]:
    if pd.isna(code):
        return np.nan
    if code == 2:  # Trump
        return 1.0
    if code == 1:  # Harris
        return 0.0
    return np.nan  # Exclude Other/Won't vote/Not sure

poll_df['harris_preference'] = poll_df['CC24_364b'].apply(get_harris_preference)
poll_df['trump_preference'] = poll_df['CC24_364b'].apply(get_trump_preference)

# Group by state to get poll percentages for the two scenarios
# 1. All respondents who expressed a Harris or Trump preference
all_respondents: pd.DataFrame = poll_df[~poll_df[['harris_preference', 'trump_preference']].isna().all(axis=1)]
state_polls_all: pd.DataFrame = all_respondents.groupby('inputstate').agg({
    'harris_preference': 'mean',
    'trump_preference': 'mean',
    'caseid': 'count'
}).reset_index()

# 2. Only likely voters
likely_voters: pd.DataFrame = poll_df[poll_df['is_likely_voter'] & ~poll_df[['harris_preference', 'trump_preference']].isna().all(axis=1)]
state_polls_likely: pd.DataFrame = likely_voters.groupby('inputstate').agg({
    'harris_preference': 'mean',
    'trump_preference': 'mean',
    'caseid': 'count'
}).reset_index()

# 3. Validated voters
validated_voters: pd.DataFrame = poll_df[poll_df['is_validated_voter'] & ~poll_df[['harris_preference', 'trump_preference']].isna().all(axis=1)]
state_polls_validated: pd.DataFrame = validated_voters.groupby('inputstate').agg({
    'harris_preference': 'mean',
    'trump_preference': 'mean',
    'caseid': 'count'
}).reset_index()

# Rename columns for clarity
state_polls_all.rename(columns={
    'harris_preference': 'harris_poll_all',
    'trump_preference': 'trump_poll_all',
    'caseid': 'num_respondents_all'
}, inplace=True)

state_polls_likely.rename(columns={
    'harris_preference': 'harris_poll_likely',
    'trump_preference': 'trump_poll_likely',
    'caseid': 'num_respondents_likely'
}, inplace=True)

state_polls_validated.rename(columns={
    'harris_preference': 'harris_poll_validated',
    'trump_preference': 'trump_poll_validated',
    'caseid': 'num_respondents_validated'
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

# Apply FIPS state code mapping
state_polls_all['inputstate'] = state_polls_all['inputstate'].astype(int)
state_polls_likely['inputstate'] = state_polls_likely['inputstate'].astype(int)
state_polls_validated['inputstate'] = state_polls_validated['inputstate'].astype(int)

state_polls_all['state'] = state_polls_all['inputstate'].map(fips_to_state)
state_polls_likely['state'] = state_polls_likely['inputstate'].map(fips_to_state)
state_polls_validated['state'] = state_polls_validated['inputstate'].map(fips_to_state)

# Normalize case
election_df['state'] = election_df['state'].str.title()
state_polls_all['state'] = state_polls_all['state'].str.title()
state_polls_likely['state'] = state_polls_likely['state'].str.title()
state_polls_validated['state'] = state_polls_validated['state'].str.title()

classification_df['State'] = classification_df['State'].str.strip('"').str.title()

# Merge
election_df = election_df.merge(classification_df, left_on='state', right_on='State', how='left')
merged_all = election_df.merge(state_polls_all, on='state', how='inner')
merged_likely = election_df.merge(state_polls_likely, on='state', how='inner')
merged_validated = election_df.merge(state_polls_validated, on='state', how='inner')

# %%
# Create 6 plots (2 rows x 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

def create_scatter(
    ax: Axes,
    x: pd.Series,
    y: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    states: pd.Series,
    state_classifications: pd.Series,
    candidate: str,
) -> None:
    colors = []
    for classification in state_classifications:
        if classification == "Swing":
            colors.append("green")
        elif classification in ["Blue", "Likely Blue"]:
            colors.append("blue")
        else:
            colors.append("red")

    ax.scatter(x, y, alpha=0.7, c=colors, s=5)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.plot([0, 1], [0, 1], "k--")

    ax.text(0.05, 0.95, f"Polls overestimated\n{candidate} support", transform=ax.transAxes,
            fontsize=11, horizontalalignment='left', verticalalignment='top')
    ax.text(0.95, 0.05, f"Polls underestimated\n{candidate} support", transform=ax.transAxes,
            fontsize=11, horizontalalignment='right', verticalalignment='bottom')

    rmse = np.sqrt(np.mean((x - y) ** 2))
    ax.text(0.95, -0.25, f"RMSE: {rmse:.2f}",
            transform=ax.transAxes,
            fontsize=11, horizontalalignment='right')

    ax.grid(True, linestyle="--", alpha=0.6)


# Harris plots
create_scatter(
    axes[0, 0],
    merged_all["harris_share"],
    merged_all["harris_poll_all"],
    "Harris: Raw Poll Estimate vs. Actual Vote Share (2-Party)",
    "Final Harris Popular Vote Share",
    "Raw Poll Estimate,\nHarris Support (2-Party)",
    merged_all["state"],
    merged_all["Pre-Election Classification"],
    "Harris"
)

create_scatter(
    axes[0, 1],
    merged_likely["harris_share"],
    merged_likely["harris_poll_likely"],
    "Harris: Turnout-Adjusted Poll Estimate vs. Actual Vote Share (2-Party)",
    "Final Harris Popular Vote Share",
    "Turnout-Adjusted Poll Estimate,\nHarris Support (2-Party)",
    merged_likely["state"],
    merged_likely["Pre-Election Classification"],
    "Harris"
)

create_scatter(
    axes[0, 2],
    merged_validated["harris_share"],
    merged_validated["harris_poll_validated"],
    "Harris: Validated Voters Poll Estimate vs. Actual Vote Share (2-Party)",
    "Final Harris Popular Vote Share",
    "Validated Voters Poll Estimate,\nHarris Support (2-Party)",
    merged_validated["state"],
    merged_validated["Pre-Election Classification"],
    "Harris"
)

# Trump plots
create_scatter(
    axes[1, 0],
    merged_all["trump_share"],
    merged_all["trump_poll_all"],
    "Trump: Raw Poll Estimate vs. Actual Vote Share (2-Party)",
    "Final Trump Popular Vote Share",
    "Raw Poll Estimate,\nTrump Support (2-Party)",
    merged_all["state"],
    merged_all["Pre-Election Classification"],
    "Trump"
)

create_scatter(
    axes[1, 1],
    merged_likely["trump_share"],
    merged_likely["trump_poll_likely"],
    "Trump: Turnout-Adjusted Poll Estimate vs. Actual Vote Share (2-Party)",
    "Final Trump Popular Vote Share",
    "Turnout-Adjusted Poll Estimate,\nTrump Support (2-Party)",
    merged_likely["state"],
    merged_likely["Pre-Election Classification"],
    "Trump"
)

create_scatter(
    axes[1, 2],
    merged_validated["trump_share"],
    merged_validated["trump_poll_validated"],
    "Trump: Validated Voters Poll Estimate vs. Actual Vote Share (2-Party)",
    "Final Trump Popular Vote Share",
    "Validated Voters Poll Estimate,\nTrump Support (2-Party)",
    merged_validated["state"],
    merged_validated["Pre-Election Classification"],
    "Trump"
)

plt.tight_layout()
plt.savefig("../figures/figure_4_2pvs.png", dpi=300, bbox_inches="tight")

print("Figure 4 (2-Party Vote Share) has been generated and saved to figures/figure_4_2pvs.png")

# %%
