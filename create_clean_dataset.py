"""
Create clean dataset for AQI prediction model training
Dataset: Air Quality Index of Major Indian Cities (Mendeley)
Features: PM2.5, PM10, NO2, SO2, CO, O3
Target: AQI
"""
import os
import sys
sys.path.append('ml_pipeline')

import pandas as pd
import numpy as np

def main():
    print("Creating clean dataset for AQI prediction training...")
    
    # Load raw dataset (Mendeley Indian Cities AQI)
    raw_data_path = "data/raw_dataset/Dataset_AQI22-4.xlsx"
    
    if not os.path.exists(raw_data_path):
        print(f"ERROR: Raw dataset not found at {raw_data_path}")
        print("Download from: https://data.mendeley.com/datasets/xg43xct9yz")
        return
    
    df_raw = pd.read_excel(raw_data_path)
    print(f"Raw dataset loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    print(f"Columns: {list(df_raw.columns)}")
    
    # Select only the columns we need (6 input features + 1 target)
    target_columns = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']
    df_clean = df_raw[target_columns].copy()
    
    # Convert AQI to numeric (it may be stored as object/string)
    df_clean['AQI'] = pd.to_numeric(df_clean['AQI'], errors='coerce')
    
    # Drop rows with any missing values in key columns
    before_drop = len(df_clean)
    df_clean = df_clean.dropna()
    after_drop = len(df_clean)
    print(f"Dropped {before_drop - after_drop} rows with missing values")
    
    # Rename columns for consistency in the pipeline
    df_clean.columns = ['PM25', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']
    
    # Reset index for clean sequential indexing
    df_clean = df_clean.reset_index(drop=True)
    
    # Save clean dataset
    output_path = "data/clean_dataset.csv"
    df_clean.to_csv(output_path, index=False)
    
    print(f"\nClean dataset saved to: {output_path}")
    print(f"Shape: {df_clean.shape}")
    print(f"Columns: {list(df_clean.columns)}")
    
    # Verify no missing values
    missing_count = df_clean.isnull().sum().sum()
    print(f"Missing values: {missing_count}")
    
    # Print statistics
    print("\nDataset statistics:")
    print(df_clean.describe().to_string())
    
    print("\nStep 2 validation:")
    print("✔ Dataset loads correctly")
    print("✔ Missing values removed")
    print("✔ Only 6 input features + 1 target remain")
    print("✔ Clean dataset saved as CSV")
    print("\nStep 2 complete!")

if __name__ == "__main__":
    main()