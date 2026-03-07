"""
Dataset loader for Mendeley Air Quality Index of Major Indian Cities
Loads, selects relevant pollutant columns and AQI target
"""
import pandas as pd
import numpy as np

def load_air_quality_dataset(csv_path):
    """
    Load Mendeley AQI dataset and prepare for model training
    
    Args:
        csv_path: Path to clean_dataset.csv (already preprocessed)
        
    Returns:
        DataFrame with selected columns: PM25, PM10, NO2, SO2, CO, O3, AQI
    """
    # Read CSV (standard comma-separated)
    df = pd.read_csv(csv_path)
    
    # Ensure expected columns are present
    expected_columns = ['PM25', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found. Available: {list(df.columns)}")
    
    df_clean = df[expected_columns].copy()
    
    print(f"Dataset loaded: {len(df_clean)} samples")
    print(f"Columns: {list(df_clean.columns)}")
    print(f"Input features: PM25, PM10, NO2, SO2, CO, O3")
    print(f"Target: AQI")
    
    return df_clean

if __name__ == "__main__":
    # Test the loader
    df = load_air_quality_dataset("../data/clean_dataset.csv")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())