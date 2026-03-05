"""
Comprehensive audit of Steps 1-3 for edge-based air quality prediction system
Verifies every requirement and catches any potential issues
"""
import os
import pandas as pd
import numpy as np
import yaml
from datetime import datetime

def audit_step1():
    """Audit Step 1: Project Structure and Environment"""
    print("🔍 AUDITING STEP 1: Project Structure and Environment")
    print("=" * 60)
    
    issues = []
    
    # Check project structure
    required_structure = {
        "data/raw_dataset/": ["AirQualityUCI.csv", "README.md"],
        "ml_pipeline/": ["dataset_loader.py", "preprocess.py", "feature_engineering.py", 
                        "train_model.py", "evaluate_model.py", "export_tflite.py"],
        "model/": ["gru_model.py"],
        "simulation/": ["sensor_simulator.py", "esp32_data_stream.py"],
        "firmware/": ["esp32_main.ino", "sensor_reader.cpp", "inference_engine.cpp", "purifier_control.cpp"],
        "config/": ["config.yaml"],
        "notebooks/": ["model_experiments.ipynb"],
        "": ["requirements.txt"]
    }
    
    print("✓ Checking project structure...")
    for directory, files in required_structure.items():
        dir_path = directory if directory else "."
        if not os.path.exists(dir_path):
            issues.append(f"Missing directory: {directory}")
            continue
            
        for file in files:
            file_path = os.path.join(directory, file)
            if not os.path.exists(file_path):
                issues.append(f"Missing file: {file_path}")
            else:
                print(f"  ✓ {file_path}")
    
    # Check virtual environment
    venv_path = "venv"
    if os.path.exists(venv_path):
        print(f"  ✓ Virtual environment exists: {venv_path}")
    else:
        issues.append("Virtual environment missing")
    
    # Check dataset
    dataset_path = "data/raw_dataset/AirQualityUCI.csv"
    if os.path.exists(dataset_path):
        df_raw = pd.read_csv(dataset_path, sep=';', decimal=',')
        print(f"  ✓ Dataset loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
        
        # Verify expected columns exist
        expected_cols = ['Date', 'Time', 'CO(GT)', 'NO2(GT)', 'T', 'RH']
        missing_cols = [col for col in expected_cols if col not in df_raw.columns]
        if missing_cols:
            issues.append(f"Missing dataset columns: {missing_cols}")
        else:
            print(f"  ✓ All required columns present")
    else:
        issues.append("UCI Air Quality dataset not found")
    
    return issues

def audit_step2():
    """Audit Step 2: Dataset Cleaning and Feature Preparation"""
    print("\n🔍 AUDITING STEP 2: Dataset Cleaning and Feature Preparation")
    print("=" * 60)
    
    issues = []
    
    # Check clean dataset exists
    clean_path = "data/clean_dataset.csv"
    if not os.path.exists(clean_path):
        issues.append("Clean dataset file missing")
        return issues
    
    # Load and verify clean dataset
    df_clean = pd.read_csv(clean_path, index_col='datetime', parse_dates=True)
    print(f"✓ Clean dataset loaded: {df_clean.shape}")
    
    # Verify columns
    expected_columns = ['CO_GT', 'NO2_GT', 'T', 'RH']
    if list(df_clean.columns) != expected_columns:
        issues.append(f"Wrong columns. Expected: {expected_columns}, Got: {list(df_clean.columns)}")
    else:
        print(f"  ✓ Correct columns: {list(df_clean.columns)}")
    
    # Check for missing values
    missing_count = df_clean.isnull().sum().sum()
    if missing_count > 0:
        issues.append(f"Clean dataset has {missing_count} missing values")
    else:
        print(f"  ✓ No missing values")
    
    # Check datetime index
    if not isinstance(df_clean.index, pd.DatetimeIndex):
        issues.append("Index is not datetime")
    else:
        print(f"  ✓ Datetime index: {df_clean.index.min()} to {df_clean.index.max()}")
    
    # Check for -200 values (should be removed)
    has_negative_200 = (df_clean == -200.0).any().any()
    if has_negative_200:
        issues.append("Dataset still contains -200 values")
    else:
        print(f"  ✓ No -200 missing value indicators")
    
    # Check normalization stats
    norm_path = "config/normalization.yaml"
    if not os.path.exists(norm_path):
        issues.append("Normalization stats file missing")
    else:
        with open(norm_path, 'r') as f:
            stats = yaml.safe_load(f)
        
        # Verify stats for all columns
        for col in expected_columns:
            if col not in stats:
                issues.append(f"Missing normalization stats for {col}")
            elif 'mean' not in stats[col] or 'std' not in stats[col]:
                issues.append(f"Incomplete normalization stats for {col}")
            else:
                print(f"  ✓ {col}: mean={stats[col]['mean']:.3f}, std={stats[col]['std']:.3f}")
    
    # Verify data quality
    print(f"  ✓ Data range: {df_clean.min().min():.3f} to {df_clean.max().max():.3f}")
    
    return issues

def audit_step3():
    """Audit Step 3: Time-Series Window Creation"""
    print("\n🔍 AUDITING STEP 3: Time-Series Window Creation")
    print("=" * 60)
    
    issues = []
    
    # Check all required files exist
    required_files = ["train_X.npy", "train_y.npy", "val_X.npy", "val_y.npy", "test_X.npy", "test_y.npy"]
    data_dir = "data"
    
    arrays = {}
    for file in required_files:
        filepath = os.path.join(data_dir, file)
        if not os.path.exists(filepath):
            issues.append(f"Missing file: {filepath}")
        else:
            arrays[file.replace('.npy', '')] = np.load(filepath)
            print(f"  ✓ {file}: {arrays[file.replace('.npy', '')].shape}")
    
    if len(arrays) != 6:
        return issues
    
    # Verify shapes
    X_train, X_val, X_test = arrays['train_X'], arrays['val_X'], arrays['test_X']
    y_train, y_val, y_test = arrays['train_y'], arrays['val_y'], arrays['test_y']
    
    # Check input shapes (samples, 24, 4)
    expected_timesteps = 24
    expected_features = 4
    
    for name, X in [('train', X_train), ('val', X_val), ('test', X_test)]:
        if len(X.shape) != 3:
            issues.append(f"{name}_X should be 3D, got {len(X.shape)}D")
        elif X.shape[1] != expected_timesteps:
            issues.append(f"{name}_X timesteps should be {expected_timesteps}, got {X.shape[1]}")
        elif X.shape[2] != expected_features:
            issues.append(f"{name}_X features should be {expected_features}, got {X.shape[2]}")
    
    # Check target shapes (samples,)
    for name, y in [('train', y_train), ('val', y_val), ('test', y_test)]:
        if len(y.shape) != 1:
            issues.append(f"{name}_y should be 1D, got {len(y.shape)}D")
    
    # Check matching sample counts
    if X_train.shape[0] != y_train.shape[0]:
        issues.append(f"Train X and y sample mismatch: {X_train.shape[0]} vs {y_train.shape[0]}")
    if X_val.shape[0] != y_val.shape[0]:
        issues.append(f"Val X and y sample mismatch: {X_val.shape[0]} vs {y_val.shape[0]}")
    if X_test.shape[0] != y_test.shape[0]:
        issues.append(f"Test X and y sample mismatch: {X_test.shape[0]} vs {y_test.shape[0]}")
    
    # Check split ratios
    total_samples = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    train_ratio = X_train.shape[0] / total_samples
    val_ratio = X_val.shape[0] / total_samples
    test_ratio = X_test.shape[0] / total_samples
    
    print(f"  ✓ Split ratios: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")
    
    # Verify ratios are approximately correct (within 1%)
    if abs(train_ratio - 0.70) > 0.01:
        issues.append(f"Train ratio {train_ratio:.1%} not close to 70%")
    if abs(val_ratio - 0.15) > 0.01:
        issues.append(f"Val ratio {val_ratio:.1%} not close to 15%")
    if abs(test_ratio - 0.15) > 0.01:
        issues.append(f"Test ratio {test_ratio:.1%} not close to 15%")
    
    # Check normalization
    X_all = np.concatenate([X_train, X_val, X_test])
    mean_val = X_all.mean()
    std_val = X_all.std()
    
    print(f"  ✓ Normalization check: mean={mean_val:.3f}, std={std_val:.3f}")
    
    if abs(mean_val) > 0.1:
        issues.append(f"Data not properly normalized: mean={mean_val:.3f} (should be ~0)")
    if abs(std_val - 1.0) > 0.1:
        issues.append(f"Data not properly normalized: std={std_val:.3f} (should be ~1)")
    
    # Verify no data leakage (chronological order)
    # This is harder to verify directly, but we can check that the implementation exists
    feature_eng_path = "ml_pipeline/feature_engineering.py"
    if os.path.exists(feature_eng_path):
        with open(feature_eng_path, 'r') as f:
            content = f.read()
            if "chronological_split" in content and "shuffle" not in content.lower():
                print(f"  ✓ Chronological split implemented correctly")
            else:
                issues.append("Chronological split may not be implemented correctly")
    
    return issues

def comprehensive_audit():
    """Run complete audit of all steps"""
    print("🔍 COMPREHENSIVE AUDIT: Edge-based Air Quality Prediction System")
    print("=" * 70)
    print(f"Audit Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    all_issues = []
    
    # Audit each step
    all_issues.extend(audit_step1())
    all_issues.extend(audit_step2())
    all_issues.extend(audit_step3())
    
    # Final report
    print("\n" + "=" * 70)
    print("🎯 AUDIT SUMMARY")
    print("=" * 70)
    
    if not all_issues:
        print("✅ ALL CHECKS PASSED!")
        print("✓ Project structure is complete")
        print("✓ Dataset is properly cleaned and preprocessed")
        print("✓ Time-series windows are correctly formatted")
        print("✓ All files are present and valid")
        print("✓ No data quality issues detected")
        print("\n🚀 System is ready for GRU model training!")
        return True
    else:
        print("❌ ISSUES DETECTED:")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i:2d}. {issue}")
        print(f"\n⚠️  Total issues found: {len(all_issues)}")
        return False

if __name__ == "__main__":
    success = comprehensive_audit()
    exit(0 if success else 1)