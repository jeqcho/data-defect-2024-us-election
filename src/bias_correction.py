# this file computes the unbiased estimator given an estimator
# input: ../data/merged_all_voters.csv
# columns: Unnamed: 0,state,trump_votes,harris_votes,total_votes,last_updated,trump_share,harris_share,State,Pre-Election Classification,inputstate,harris_poll_all,trump_poll_all,num_respondents_all
# task: compute trump_poll_all - rho * sqrt((1-f)/f) * sigma
# rho = -0.0045
# f = sample_ratio = num_respondents_all / total_votes
# sigma = standard deviation of trump_poll_all
# output: ../data/bias_correction.csv

import pandas as pd
import numpy as np
import os

# Constants
RHO = -0.0045

# Set the current working directory to the script directory
script_dir: str = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load data
df = pd.read_csv('../data/merged_all_voters.csv')

# Calculate sample ratio f = num_respondents_all / total_votes
df['f'] = df['num_respondents_all'] / df['total_votes']

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
    'last_updated', 'trump_share', 'harris_share', 'State', 
    'Pre-Election Classification', 'inputstate', 'harris_poll_all', 
    'trump_poll_all', 'num_respondents_all', 'f', 'sigma', 'bias_correction_term',
    'trump_poll_corrected', 'harris_poll_corrected'
]

df[output_columns].to_csv('../data/bias_correction.csv', index=False)

print("Bias correction completed and saved to '../data/bias_correction.csv'")