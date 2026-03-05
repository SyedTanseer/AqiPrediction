"""
Dataset loader for UCI Air Quality Dataset
Loads, parses timestamps, and selects relevant sensor columns
"""
import pandas as pd
import numpy as np
from datetime import datetime

def load_air_quality_dataset(csv_path):
    """
    Load UCI Air Quality dataset and prepare for time-series analysis
    
    Args:
        csv_path: Path to AirQualityUCI.csv
        
    Returns:
        DataFrame with datetime index and selected sensor columns
    """
    # Read CSV with semicolon separator
    df = pd.read_csv(csv_path, sep=';', decimal=',')
    
    # Remove empty columns (the dataset has trailing semicolons)
    df = df.dropna(axis=1, how='all')
    
    # Combine Date and Time columns into datetime index
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], 
                                   format='%d/%m/%Y %H.%M.%S')
    
    # Set datetime as index
    df.set_index('datetime', inplace=True)
    
    # Select only the target columns that map to our sensors
    target_columns = ['CO(GT)', 'NO2(GT)', 'T', 'RH']
    df_clean = df[target_columns].copy()
    
    # Rename columns for consistency
    df_clean.columns = ['CO_GT', 'NO2_GT', 'T', 'RH']
    
    print(f"Dataset loaded: {len(df_clean)} samples")
    print(f"Date range: {df_clean.index.min()} to {df_clean.index.max()}")
    print(f"Columns: {list(df_clean.columns)}")
    
    return df_clean

if __name__ == "__main__":
    # Test the loader
    df = load_air_quality_dataset("../data/raw_dataset/AirQualityUCI.csv")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())