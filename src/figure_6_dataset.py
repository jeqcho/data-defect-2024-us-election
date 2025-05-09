#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

# Set the current working directory to the script directory
script_dir: str = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Read the input file
suffix = "_likely"
# suffix = ""
input_file = f"../data/figure_5{suffix}.csv"
print(f"Reading input file: {input_file}")
df = pd.read_csv(input_file)

# Display basic information about the dataset
print("\nColumns in the dataset:")
print(df.columns.tolist())
print("\nFirst few rows of the dataset:")
print(df.head())

# Compute the metrics as per instructions
print("\nComputing required metrics...")

# Step 1: Compute s_g_sq = N/(N-1)*sigma_g**2 where N is the total_votes
df['trump_s_g_sq'] = df['total_votes'] / (df['total_votes'] - 1) * df['trump_sigma_g']**2
df['harris_s_g_sq'] = df['total_votes'] / (df['total_votes'] - 1) * df['harris_sigma_g']**2

# Step 2: Compute var_srs = (1-f)/n * s_g_sq where f is sample_ratio and n is sample_size
df['trump_var_srs'] = (1 - df['sample_ratio']) / df['sample_size'] * df['trump_s_g_sq']
df['harris_var_srs'] = (1 - df['sample_ratio']) / df['sample_size'] * df['harris_s_g_sq']

# Step 3: Compute Z_n_N = data_defect_correlation/sqrt(var_srs)
df['trump_Z_n_N'] = df['trump_error'] / np.sqrt(df['trump_var_srs'])
df['harris_Z_n_N'] = df['harris_error'] / np.sqrt(df['harris_var_srs'])

# Step 4: Include total_votes as column (already included in original dataset)

# Selecting relevant columns for the final dataset
result_df = df[['state', 
                'trump_sigma_g', 'harris_sigma_g',
                'sample_ratio', 'sample_size', 'total_votes',
                'trump_error', 'harris_error',
                'trump_s_g_sq', 'harris_s_g_sq',
                'trump_var_srs', 'harris_var_srs',
                'trump_Z_n_N', 'harris_Z_n_N']]

# Save the result to a new CSV file
output_file = f"../data/figure_6{suffix}.csv"
result_df.to_csv(output_file, index=False)

print(f"\nComputation completed. Results saved to {output_file}")
print("\nSummary statistics for the computed variables:")
print(result_df[['trump_s_g_sq', 'harris_s_g_sq', 
                'trump_var_srs', 'harris_var_srs',
                'trump_Z_n_N', 'harris_Z_n_N']].describe())