"""
Data preprocessing module for air quality time-series data
Handles missing values, resampling, and normalization
"""
import pandas as pd
import numpy as np
import yaml
import os

def handle_missing_values(df):
    """
    Convert -200 values to NaN and apply forward fill
    
    Args:
        df: DataFrame with potential -200 missing values
        
    Returns:
        DataFrame with missing values handled
    """
    # Replace -200 with NaN
    df_clean = df.replace(-200.0, np.nan)
    
    # Forward fill missing values (limit=3 to avoid propagating across long outages)
    df_clean = df_clean.ffill(limit=3)
    
    # Drop any remaining NaN values at the beginning
    df_clean = df_clean.dropna()
    
    print(f"Missing values handled. Remaining samples: {len(df_clean)}")
    return df_clean

def resample_to_hourly(df):
    """
    Ensure hourly sampling and sort by timestamp
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame resampled to hourly frequency
    """
    # Sort by timestamp
    df_sorted = df.sort_index()
    
    # Resample to hourly (mean for any sub-hourly data)
    df_hourly = df_sorted.resample('h').mean()
    
    # Drop any NaN values from resampling
    df_hourly = df_hourly.dropna()
    
    print(f"Resampled to hourly: {len(df_hourly)} samples")
    return df_hourly

def compute_normalization_stats(df, save_path):
    """
    Compute and save normalization statistics
    
    Args:
        df: DataFrame to compute stats for
        save_path: Path to save normalization.yaml
        
    Returns:
        Dictionary with normalization stats
    """
    stats = {}
    
    for column in df.columns:
        stats[column] = {
            'mean': float(df[column].mean()),
            'std': float(df[column].std())
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
    Complete preprocessing pipeline
    
    Args:
        df: Raw dataset DataFrame
        config_dir: Directory to save normalization stats
        
    Returns:
        Tuple of (processed_df, normalization_stats)
    """
    print("Starting preprocessing pipeline...")
    
    # Step 1: Handle missing values
    df_clean = handle_missing_values(df)
    
    # Step 2: Resample to hourly
    df_hourly = resample_to_hourly(df_clean)
    
    # Step 3: Compute normalization stats
    norm_path = os.path.join(config_dir, "normalization.yaml")
    stats = compute_normalization_stats(df_hourly, norm_path)
    
    print("Preprocessing complete!")
    return df_hourly, stats

if __name__ == "__main__":
    from dataset_loader import load_air_quality_dataset
    
    # Load dataset
    df = load_air_quality_dataset("../data/raw_dataset/AirQualityUCI.csv")
    
    # Preprocess
    df_processed, stats = preprocess_dataset(df)
    
    print("\nProcessed dataset info:")
    print(df_processed.info())
    print("\nFirst 5 rows:")
    print(df_processed.head())