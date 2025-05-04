# explore ../data/CES24_Common.csv and show the column names
import pandas as pd
import os
import sys

# Set the correct working directory
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Set the working directory to the script directory
os.chdir(script_dir)

# Read the CSV file
file_path = '../data/CES24_Common.csv'
try:
    print(f"Current working directory: {os.getcwd()}")
    print(f"Attempting to read file: {os.path.abspath(file_path)}")
    
    # Read just the header to get column names
    df_headers = pd.read_csv(file_path, nrows=0)
    
    # Display column names
    print(f"The CSV file has {len(df_headers.columns)} columns:")
    for i, col in enumerate(df_headers.columns, 1):
        print(f"{i}. {col}")
    
    # Display basic info about the dataset
    print("\nReading full dataset for summary information...")
    df = pd.read_csv(file_path)
    print(f"\nDataset shape: {df.shape} (rows, columns)")
    print("\nDataset summary:")
    print(df.info())
    
    # Show first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Get counts of values in the 'tookpost' column
    if 'tookpost' in df.columns:
        print("\nCounts of values in 'tookpost' column:")
        tookpost_counts = df['tookpost'].value_counts(dropna=False)
        print(tookpost_counts)
        
        # Display percentage distribution
        print("\nPercentage distribution of 'tookpost' values:")
        tookpost_pct = df['tookpost'].value_counts(normalize=True, dropna=False) * 100
        print(tookpost_pct.round(2))
    else:
        print("\nColumn 'tookpost' not found in the dataset.")
    
except Exception as e:
    print(f"Error reading CSV file: {e}")
    print(f"File exists: {os.path.exists(file_path)}")