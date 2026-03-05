"""
Feature engineering for time-series air quality data
Creates sliding windows for GRU model training
"""
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

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

def chronological_split(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split time-series data chronologically (no shuffling)
    
    Args:
        X: Input features
        y: Target values
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n_samples = len(X)
    
    # Calculate split indices
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    # Split chronologically
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]
    
    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_processed_arrays(X_train, X_val, X_test, y_train, y_val, y_test, data_dir="data"):
    """
    Save processed arrays as .npy files
    
    Args:
        X_train, X_val, X_test: Input arrays
        y_train, y_val, y_test: Target arrays
        data_dir: Directory to save files
    """
    import os
    
    # Save all arrays
    np.save(os.path.join(data_dir, "train_X.npy"), X_train)
    np.save(os.path.join(data_dir, "train_y.npy"), y_train)
    np.save(os.path.join(data_dir, "val_X.npy"), X_val)
    np.save(os.path.join(data_dir, "val_y.npy"), y_val)
    np.save(os.path.join(data_dir, "test_X.npy"), X_test)
    np.save(os.path.join(data_dir, "test_y.npy"), y_test)
    
    print(f"Processed arrays saved to {data_dir}/")

def process_time_series_data(csv_path, config_path, data_dir="data", 
                           history_window=24, forecast_horizon=1):
    """
    Complete time-series processing pipeline
    
    Args:
        csv_path: Path to clean dataset CSV
        config_path: Path to normalization.yaml
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
    
    # Load normalization stats
    stats = load_normalization_stats(config_path)
    
    # Normalize dataset
    df_norm = normalize_dataset(df, stats)
    print("Dataset normalized")
    
    # Create time-series windows
    X, y = create_time_series_windows(df_norm, history_window, forecast_horizon)
    
    # Chronological split
    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X, y)
    
    # Save processed arrays
    save_processed_arrays(X_train, X_val, X_test, y_train, y_val, y_test, data_dir)
    
    print("Time-series processing complete!")
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    # Process the dataset
    X_train, X_val, X_test, y_train, y_val, y_test = process_time_series_data(
        csv_path="../data/clean_dataset.csv",
        config_path="../config/normalization.yaml",
        data_dir="../data"
    )
    
    print("\nFinal shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"y_test: {y_test.shape}")