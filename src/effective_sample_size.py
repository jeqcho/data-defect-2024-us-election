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
fixed_rho_output_path = os.path.join("../data", "effective_sample_size_fixed_rho.csv")

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

# Now create a copy of the dataframe with fixed rho values
df_fixed = df.copy()

# Add fixed data defect correlation values
df_fixed["trump_data_defect_correlation"] = -0.0044
df_fixed["harris_data_defect_correlation"] = 0.00016

# Recompute values with the fixed correlation values
# Compute data_defect_index_lower_bound (data_defect_index ** 2)
df_fixed["trump_data_defect_index_lower_bound"] = df_fixed["trump_data_defect_correlation"] ** 2
df_fixed["harris_data_defect_index_lower_bound"] = df_fixed["harris_data_defect_correlation"] ** 2

# Calculate effective_sample_size (data_defect_index_lower_bound * sample_ratio / (1-sample_ratio))
df_fixed["trump_effective_sample_size"] = (
    df_fixed["sample_ratio"]
    / (1 - df_fixed["sample_ratio"])
    / df_fixed["trump_data_defect_index_lower_bound"]
)
df_fixed["harris_effective_sample_size"] = (
    df_fixed["sample_ratio"]
    / (1 - df_fixed["sample_ratio"])
    / df_fixed["harris_data_defect_index_lower_bound"]
)

# Calculate percentage_reduction ((1-effective_sample_size/sample_size)*100%)
df_fixed["trump_percentage_reduction"] = (
    1 - df_fixed["trump_effective_sample_size"] / df_fixed["sample_size"]
) * 100
df_fixed["harris_percentage_reduction"] = (
    1 - df_fixed["harris_effective_sample_size"] / df_fixed["sample_size"]
) * 100

# Print summary statistics for fixed values
print("\nFixed rho values:")
print(f"Trump data defect correlation: -0.0044")
print(f"Harris data defect correlation: 0.00016")
print(
    f"Average Trump percentage reduction (fixed): {df_fixed['trump_percentage_reduction'].mean():.2f}%"
)
print(
    f"Average Harris percentage reduction (fixed): {df_fixed['harris_percentage_reduction'].mean():.2f}%"
)

# Save the fixed results to the new output file
print(f"Saving fixed rho results to {fixed_rho_output_path}")
df_fixed.to_csv(fixed_rho_output_path, index=False)

print("Processing complete.")
