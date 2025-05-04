# the goal of this script is to compute a dataset
# input: ../data/merged_all_voters.csv
# output: ../data/figure_5.csv

# first you should read merged_all_voters.csv to understand the columns, and read the first few rows to understand the data types

# the output dataset has each row corresponding to each state

# the first quantity is the trump_error, which is trump_poll_all - trump_share. Do the same for Harris

# the second quantity is sigma_g, which calculates the standard deviation of the vote share
# Do so for both Trump and Harris (they are not exactly complements, because of other candidates)

# the third quantity is the sample_ratio, which is the ratio between num_respondents_all and total_votes

# the fourth quantity is the data_defect_correlation, which is error/(sigma_g * sqrt((1-f)/f))

import pandas as pd
import numpy as np
import os

# Read the input data
input_file = 'data/merged_all_voters.csv'
output_file = 'data/figure_5.csv'

# Read the data
df = pd.read_csv(input_file)

# Create a new dataframe to store the results
results = pd.DataFrame()
results['state'] = df['state']

# Calculate the first quantity: trump_error and harris_error
# error = poll - actual share
results['trump_error'] = df['trump_poll_all'] - df['trump_share']
results['harris_error'] = df['harris_poll_all'] - df['harris_share']

# Calculate the second quantity: sigma_g (standard deviation of vote share)
# For simplicity, we'll use the sample proportion as an estimate for sigma_g
# sigma_g for binomial proportion = sqrt(p * (1-p))
results['trump_sigma_g'] = np.sqrt(df['trump_share'] * (1 - df['trump_share']))
results['harris_sigma_g'] = np.sqrt(df['harris_share'] * (1 - df['harris_share']))

# Calculate the third quantity: sample_ratio (num_respondents_all / total_votes)
results['sample_ratio'] = df['num_respondents_all'] / df['total_votes']

# Calculate the fourth quantity: data_defect_correlation
# error/(sigma_g * sqrt((1-f)/f)) where f is the sample_ratio
# For Trump
f_trump = results['sample_ratio']
results['trump_data_defect_correlation'] = results['trump_error'] / (
    results['trump_sigma_g'] * np.sqrt((1 - f_trump) / f_trump)
)

# For Harris
f_harris = results['sample_ratio']
results['harris_data_defect_correlation'] = results['harris_error'] / (
    results['harris_sigma_g'] * np.sqrt((1 - f_harris) / f_harris)
)

# Save the output
results.to_csv(output_file, index=False)

print(f"Saved figure 5 data to {output_file}")