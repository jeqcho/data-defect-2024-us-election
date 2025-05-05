# the goal of this script is to compute a dataset
# input: ../data/figure_5.csv

# you, the AI assistant, should read the input files to understand the columns and the data types

# each row is a state

# first compute the data_defect_index_lower_bound, which is data_defect_index ** 2
# second calculate effective_sample_size, which is data_defect_index_lower_bound * sample_ratio / (1-sample_ratio)
# third calculate percentage_reduction, which is (1-effective_sample_size/(sample_size))*100%

# output to ../data/output.csv

import pandas as pd
import os

# Set the current working directory to the script directory
script_dir: str = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Define input and output paths
input_path = os.path.join("../data", "figure_5.csv")
output_path = os.path.join("../data", "effective_sample_size.csv")

# Read the input data
print(f"Reading data from {input_path}")
df = pd.read_csv(input_path)

# Compute data_defect_index_lower_bound (data_defect_index ** 2)
# Assuming data_defect_index is the data_defect_correlation
df["trump_data_defect_index_lower_bound"] = df["trump_data_defect_correlation"] ** 2
df["harris_data_defect_index_lower_bound"] = df["harris_data_defect_correlation"] ** 2

# Calculate effective_sample_size (data_defect_index_lower_bound * sample_ratio / (1-sample_ratio))
df["trump_effective_sample_size"] = (
    df["sample_ratio"]
    / (1 - df["sample_ratio"])
    / df["trump_data_defect_index_lower_bound"]
)
df["harris_effective_sample_size"] = (
    df["sample_ratio"]
    / (1 - df["sample_ratio"])
    / df["harris_data_defect_index_lower_bound"]
)

# Calculate percentage_reduction ((1-effective_sample_size/sample_size)*100%)
df["trump_percentage_reduction"] = (
    1 - df["trump_effective_sample_size"] / df["sample_size"]
) * 100
df["harris_percentage_reduction"] = (
    1 - df["harris_effective_sample_size"] / df["sample_size"]
) * 100

# Print summary statistics
print(f"Number of states processed: {len(df)}")
print(
    f"Average Trump percentage reduction: {df['trump_percentage_reduction'].mean():.2f}%"
)
print(
    f"Average Harris percentage reduction: {df['harris_percentage_reduction'].mean():.2f}%"
)

# Save the results to the output file
print(f"Saving results to {output_path}")
df.to_csv(output_path, index=False)

print("Processing complete.")
