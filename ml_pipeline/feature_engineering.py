"""
Feature engineering for air quality AQI prediction data
Handles normalization, train/val/test split, and array preparation

Adapted for cross-sectional Mendeley Indian Cities AQI dataset:
- No sliding windows (data is not time-series)
- Each row is a direct sample: 6 input features → AQI target
- Train-only normalization stats to prevent data leakage
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

def prepare_features_and_target(df, input_features=None, target_column='AQI'):
    """
    Split DataFrame into input features (X) and target (y)
    
    Args:
        df: DataFrame with all columns
        input_features: List of input feature column names
        target_column: Target column name
        
    Returns:
        Tuple of (X, y) as numpy arrays
    """
    if input_features is None:
        input_features = ['PM25', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    
    X = df[input_features].values.astype(np.float32)
    y = df[target_column].values.astype(np.float32)
    
    print(f"Prepared features: X{X.shape}, y{y.shape}")
    print(f"  Input features: {input_features}")
    print(f"  Target: {target_column}")
    
    return X, y

def random_split(X, y, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Randomly split data into train/val/test sets
    
    Args:
        X: Input features array
        y: Target values array
        train_ratio: Proportion for training (0.70)
        val_ratio: Proportion for validation (0.15)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n = len(X)
    
    # Shuffle indices
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"\nRandom split (seed={seed}):")
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

def process_data(csv_path, config_path, data_dir="data"):
    """
    Complete data processing pipeline for cross-sectional AQI data.
    
    Normalization stats are computed on TRAINING portion only
    to prevent data leakage from validation/test sets.
    
    Args:
        csv_path: Path to clean dataset CSV
        config_path: Path to normalization.yaml (will be overwritten with train-only stats)
        data_dir: Directory to save processed arrays
        
    Returns:
        Tuple of processed arrays
    """
    print("Starting data processing pipeline...")
    
    # Load clean dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {df.shape}")
    
    # Prepare features and target (before normalization)
    input_features = ['PM25', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    X_raw, y_raw = prepare_features_and_target(df, input_features, 'AQI')
    
    # Random split on raw data
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = random_split(X_raw, y_raw)
    
    # Compute normalization stats on TRAINING data only
    print(f"\nComputing normalization stats on training data only ({len(X_train_raw)} samples)...")
    
    # Build a DataFrame from training data for stats computation
    all_columns = input_features + ['AQI']
    train_df = pd.DataFrame(
        np.column_stack([X_train_raw, y_train]),
        columns=all_columns
    )
    stats = compute_normalization_stats(train_df)
    save_normalization_stats(stats, config_path)
    
    for col, s in stats.items():
        print(f"  {col}: mean={s['mean']:.4f}, std={s['std']:.4f}")
    
    # Normalize ALL features using train-only stats (input features only)
    input_stats = {col: stats[col] for col in input_features}
    
    # Normalize input features
    X_train = np.array([(X_train_raw[:, i] - input_stats[col]['mean']) / input_stats[col]['std'] 
                        for i, col in enumerate(input_features)]).T.astype(np.float32)
    X_val = np.array([(X_val_raw[:, i] - input_stats[col]['mean']) / input_stats[col]['std'] 
                      for i, col in enumerate(input_features)]).T.astype(np.float32)
    X_test = np.array([(X_test_raw[:, i] - input_stats[col]['mean']) / input_stats[col]['std'] 
                       for i, col in enumerate(input_features)]).T.astype(np.float32)
    
    # Normalize target (AQI) using train-only AQI stats
    aqi_mean = stats['AQI']['mean']
    aqi_std = stats['AQI']['std']
    y_train_norm = ((y_train - aqi_mean) / aqi_std).astype(np.float32)
    y_val_norm = ((y_val - aqi_mean) / aqi_std).astype(np.float32)
    y_test_norm = ((y_test - aqi_mean) / aqi_std).astype(np.float32)
    
    print(f"\nNormalized shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train_norm.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val_norm.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test_norm.shape}")
    
    # Save processed arrays
    save_processed_arrays(X_train, X_val, X_test, y_train_norm, y_val_norm, y_test_norm, data_dir)
    
    print("\nData processing complete!")
    return X_train, X_val, X_test, y_train_norm, y_val_norm, y_test_norm

if __name__ == "__main__":
    process_data(
        csv_path="../data/clean_dataset.csv",
        config_path="../config/normalization.yaml",
        data_dir="../data"
    )