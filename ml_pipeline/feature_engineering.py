"""
Feature engineering for time-series air quality data
Creates sliding windows for GRU model training

FIX: Normalization stats are now computed on TRAINING data only
to prevent data leakage from validation/test sets.
"""
import numpy as np
import pandas as pd
import yaml
import os

def load_normalization_stats(config_path):
    """
    Load normalization statistics from YAML file
    
    Args:
        config_path: Path to normalization.yaml
        
    Returns:
        Dictionary with normalization stats
    """
    with open(config_path, 'r') as f:
        stats = yaml.safe_load(f)
    return stats

def save_normalization_stats(stats, config_path):
    """
    Save normalization statistics to YAML file
    
    Args:
        stats: Dictionary with per-column mean/std
        config_path: Path to save normalization.yaml
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    print(f"Normalization stats saved to {config_path}")

def compute_normalization_stats(df):
    """
    Compute normalization statistics from a DataFrame
    
    Args:
        df: DataFrame to compute stats for (should be TRAINING data only)
        
    Returns:
        Dictionary with per-column mean and std
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
    return stats

def normalize_dataset(df, stats):
    """
    Normalize dataset using pre-computed statistics
    
    Args:
        df: DataFrame to normalize
        stats: Normalization statistics dictionary
        
    Returns:
        Normalized DataFrame
    """
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

def create_time_series_windows(data, history_window=24, forecast_horizon=1, target_column='CO_GT'):
    """
    Create sliding windows for time-series prediction
    
    Args:
        data: DataFrame with time-series data
        history_window: Number of past timesteps to use as input (24 hours)
        forecast_horizon: Number of future timesteps to predict (1 hour)
        target_column: Column to predict
        
    Returns:
        Tuple of (X, y) where:
        X shape: (samples, history_window, n_features)
        y shape: (samples,)
    """
    # Convert to numpy array
    values = data.values
    target_idx = data.columns.get_loc(target_column)
    
    X, y = [], []
    
    # Create sliding windows
    for i in range(history_window, len(values) - forecast_horizon + 1):
        # Input window: past 24 hours of all features
        X.append(values[i-history_window:i, :])
        
        # Target: next hour's CO_GT value
        y.append(values[i + forecast_horizon - 1, target_idx])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} time-series windows")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y

def chronological_split(X, y, train_ratio=0.70, val_ratio=0.15):
    """
    Split data chronologically (no shuffling to prevent data leakage)
    
    Args:
        X: Input features array
        y: Target values array
        train_ratio: Proportion for training (0.70)
        val_ratio: Proportion for validation (0.15)
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"\nChronological split (no shuffling):")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/n:.1%})")
    print(f"  Val:   {len(X_val)} samples ({len(X_val)/n:.1%})")
    print(f"  Test:  {len(X_test)} samples ({len(X_test)/n:.1%})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_processed_arrays(X_train, X_val, X_test, y_train, y_val, y_test, data_dir):
    """Save all processed arrays to disk"""
    os.makedirs(data_dir, exist_ok=True)
    
    np.save(os.path.join(data_dir, "train_X.npy"), X_train)
    np.save(os.path.join(data_dir, "train_y.npy"), y_train)
    np.save(os.path.join(data_dir, "val_X.npy"), X_val)
    np.save(os.path.join(data_dir, "val_y.npy"), y_val)
    np.save(os.path.join(data_dir, "test_X.npy"), X_test)
    np.save(os.path.join(data_dir, "test_y.npy"), y_test)
    
    print(f"\nArrays saved to {data_dir}/")

def process_time_series_data(csv_path, config_path, data_dir="data", 
                           history_window=24, forecast_horizon=1):
    """
    Complete time-series processing pipeline
    
    FIX: Normalization stats are computed on TRAINING portion only
    to prevent data leakage from validation/test sets.
    
    Args:
        csv_path: Path to clean dataset CSV
        config_path: Path to normalization.yaml (will be overwritten with train-only stats)
        data_dir: Directory to save processed arrays
        history_window: Input sequence length
        forecast_horizon: Prediction horizon
        
    Returns:
        Tuple of processed arrays
    """
    print("Starting time-series processing pipeline...")
    
    # Load clean dataset
    df = pd.read_csv(csv_path, index_col='datetime', parse_dates=True)
    print(f"Loaded dataset: {df.shape}")
    
    # ===== FIX: Compute normalization stats on TRAINING portion only =====
    # Determine the training portion boundary on raw data
    train_end_idx = int(len(df) * 0.70)
    df_train_portion = df.iloc[:train_end_idx]
    
    print(f"\nComputing normalization stats on training data only ({train_end_idx} of {len(df)} rows)...")
    stats = compute_normalization_stats(df_train_portion)
    
    # Save train-only stats (overwrites any previously computed full-data stats)
    save_normalization_stats(stats, config_path)
    
    for col, s in stats.items():
        print(f"  {col}: mean={s['mean']:.4f}, std={s['std']:.4f}")
    # =====================================================================
    
    # Normalize ALL data using train-only stats
    df_norm = normalize_dataset(df, stats)
    print("Dataset normalized (using train-only statistics)")
    
    # Create time-series windows
    X, y = create_time_series_windows(df_norm, history_window, forecast_horizon)
    
    # Chronological split
    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X, y)
    
    # Save processed arrays
    save_processed_arrays(X_train, X_val, X_test, y_train, y_val, y_test, data_dir)
    
    print("\nTime-series processing complete!")
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    process_time_series_data(
        csv_path="../data/clean_dataset.csv",
        config_path="../config/normalization.yaml",
        data_dir="../data"
    )