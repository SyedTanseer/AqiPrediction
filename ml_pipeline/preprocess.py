"""
Data preprocessing module for air quality data
Handles missing values and normalization for the Mendeley Indian Cities AQI dataset
"""
import pandas as pd
import numpy as np
import yaml
import os

def handle_missing_values(df):
    """
    Drop rows with any NaN values in key columns
    
    Args:
        df: DataFrame with potential missing values
        
    Returns:
        DataFrame with missing values handled
    """
    before = len(df)
    df_clean = df.dropna()
    after = len(df_clean)
    
    if before != after:
        print(f"Dropped {before - after} rows with missing values. Remaining: {after}")
    else:
        print(f"No missing values found. All {after} samples retained.")
    
    return df_clean

def compute_normalization_stats(df, save_path):
    """
    Compute and save normalization statistics (z-score: mean, std)
    
    Args:
        df: DataFrame to compute stats for
        save_path: Path to save normalization.yaml
        
    Returns:
        Dictionary with normalization stats
    """
    stats = {}
    
    for column in df.columns:
        col_std = float(df[column].std())
        if col_std == 0:
            print(f"  ⚠️ Warning: {column} has std=0, using std=1.0 as fallback")
            col_std = 1.0
        stats[column] = {
            'mean': float(df[column].mean()),
            'std': col_std
        }
    
    # Create config directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save to YAML
    with open(save_path, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    
    print(f"Normalization stats saved to {save_path}")
    return stats

def normalize_features(df, stats=None):
    """
    Normalize features using z-score normalization
    
    Args:
        df: DataFrame to normalize
        stats: Optional pre-computed stats, if None computes from df
        
    Returns:
        Normalized DataFrame
    """
    if stats is None:
        stats = {col: {'mean': df[col].mean(), 'std': df[col].std()} 
                for col in df.columns}
    
    df_norm = df.copy()
    for column in df.columns:
        mean = stats[column]['mean']
        std = stats[column]['std']
        if std == 0:
            print(f"  ⚠️ Warning: {column} has std=0, setting normalized values to 0")
            df_norm[column] = 0.0
        else:
            df_norm[column] = (df[column] - mean) / std
    
    return df_norm

def preprocess_dataset(df, config_dir="../config"):
    """
    Complete preprocessing pipeline for Mendeley AQI dataset
    
    Args:
        df: Raw dataset DataFrame
        config_dir: Directory to save normalization stats
        
    Returns:
        Tuple of (processed_df, normalization_stats)
    """
    print("Starting preprocessing pipeline...")
    
    # Step 1: Handle missing values
    df_clean = handle_missing_values(df)
    
    # Step 2: Compute normalization stats on all data
    # (train-only split is handled in feature_engineering.py)
    norm_path = os.path.join(config_dir, "normalization.yaml")
    stats = compute_normalization_stats(df_clean, norm_path)
    
    print("Preprocessing complete!")
    return df_clean, stats

if __name__ == "__main__":
    from dataset_loader import load_air_quality_dataset
    
    # Load dataset
    df = load_air_quality_dataset("../data/clean_dataset.csv")
    
    # Preprocess
    df_processed, stats = preprocess_dataset(df)
    
    print("\nProcessed dataset info:")
    print(df_processed.info())
    print("\nFirst 5 rows:")
    print(df_processed.head())