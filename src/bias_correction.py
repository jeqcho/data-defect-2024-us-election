# this file computes the unbiased estimator given an estimator
# input: ../data/merged_all_voters.csv
# Turnout data: ../data/Turnout_2016G_v1.0.csv
# VEP data: ../data/Turnout_2024G_v0.3.csv
# columns: Unnamed: 0,state,trump_votes,harris_votes,total_votes,last_updated,trump_share,harris_share,State,Pre-Election Classification,inputstate,harris_poll_all,trump_poll_all,num_respondents_all
# task: compute trump_poll_all - rho * sqrt((1-f)/f) * sigma
# rho = -0.0045
# f = sample_ratio = num_respondents_all / estimated_votes
#     where estimated_votes = 2016_turnout * 2024_vep
# sigma = standard deviation of trump_poll_all
# output: ../data/bias_correction.csv

import pandas as pd
import numpy as np
import os
import re

# Constants
RHO = -0.0045

# Set the current working directory to the script directory
script_dir: str = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Helper function to clean percentage strings
def clean_percentage(pct_str):
    if pd.isna(pct_str):
        return np.nan
    if isinstance(pct_str, (int, float)):
        return pct_str / 100  # Already a number, just convert to proportion
    # Remove % sign and convert to float, then to proportion
    return float(pct_str.rstrip('%')) / 100

# Helper function to clean numeric strings with commas
def clean_numeric(num_str):
    if pd.isna(num_str):
        return np.nan
    if isinstance(num_str, (int, float)):
        return num_str  # Already a number
    # Remove commas and convert to float
    return float(re.sub(r'[,]', '', str(num_str)))

# Load data
df = pd.read_csv('../data/merged_all_voters.csv')

# Load turnout data
turnout_2016 = pd.read_csv('../data/Turnout_2016G_v1.0.csv')
turnout_2024 = pd.read_csv('../data/Turnout_2024G_v0.3.csv')

# Prepare turnout data
# Extract VEP_TURNOUT_RATE from 2016 data (as proportion)
turnout_2016['turnout_rate_2016'] = turnout_2016['VEP_TURNOUT_RATE'].apply(clean_percentage)

# Extract VEP from 2024 data
turnout_2024['vep_2024'] = turnout_2024['VEP'].apply(clean_numeric)

# Prepare data for merging
turnout_2016 = turnout_2016[['STATE', 'turnout_rate_2016']]
turnout_2024 = turnout_2024[['STATE', 'vep_2024']]

# Merge the turnout data frames together
turnout_data = pd.merge(turnout_2016, turnout_2024, on='STATE', how='inner')

# Rename the state column in the main dataframe to match turnout data for merging
df['STATE'] = df['State']

# Merge the turnout data with the main dataframe
df = pd.merge(df, turnout_data, on='STATE', how='left')

# Print some debugging information
print(f"States in merged_all_voters: {df['State'].nunique()}")
print(f"States in turnout data: {turnout_data['STATE'].nunique()}")
print(f"States successfully matched: {df['turnout_rate_2016'].notna().sum()}")

# Calculate estimated votes = 2016 turnout rate * 2024 vep
df['estimated_votes'] = df['turnout_rate_2016'] * df['vep_2024']

# Check if we have any NaN values in estimated_votes
if df['estimated_votes'].isna().any():
    print(f"Warning: {df['estimated_votes'].isna().sum()} states have missing estimated_votes")
    print("Using total_votes as a fallback for these states")
    # Use total_votes as a fallback
    df.loc[df['estimated_votes'].isna(), 'estimated_votes'] = df.loc[df['estimated_votes'].isna(), 'total_votes']

# Calculate sample ratio f = num_respondents_all / estimated_votes
df['f'] = df['num_respondents_all'] / df['estimated_votes']

# Calculate sigma for each row as the standard deviation of a Bernoulli distribution
# For a Bernoulli distribution with probability p, the standard deviation is sqrt(p*(1-p))
df['sigma'] = np.sqrt(df['trump_poll_all'] * (1 - df['trump_poll_all']))

# Calculate the bias correction term: rho * sqrt((1-f)/f) * sigma
df['bias_correction_term'] = RHO * np.sqrt((1 - df['f']) / df['f']) * df['sigma']

# Calculate the bias-corrected estimator
df['trump_poll_corrected'] = df['trump_poll_all'] - df['bias_correction_term']

# Calculate corresponding Harris corrected poll values
df['harris_poll_corrected'] = 1 - df['trump_poll_corrected']

# Save the results
output_columns = [
    'Unnamed: 0', 'state', 'trump_votes', 'harris_votes', 'total_votes', 
    'estimated_votes', 'last_updated', 'trump_share', 'harris_share', 'State', 
    'Pre-Election Classification', 'inputstate', 'harris_poll_all', 
    'trump_poll_all', 'num_respondents_all', 'f', 'sigma', 'bias_correction_term',
    'trump_poll_corrected', 'harris_poll_corrected'
]

df[output_columns].to_csv('../data/bias_correction.csv', index=False)

print("Bias correction completed and saved to '../data/bias_correction.csv'")