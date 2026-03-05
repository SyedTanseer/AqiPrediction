"""
Inspect time-series windows to verify structure
"""
import numpy as np

# Load one sample to inspect
X_train = np.load("data/train_X.npy")
y_train = np.load("data/train_y.npy")

print("Time-Series Window Structure Inspection")
print("=" * 40)

# Show first window
print("First training sample:")
print(f"Input window shape: {X_train[0].shape}")
print(f"Target value: {y_train[0]:.3f}")

print("\nInput window (24 hours × 4 features):")
print("Hour | CO_GT   NO2_GT    T      RH")
print("-" * 35)
for i in range(min(5, X_train[0].shape[0])):  # Show first 5 hours
    row = X_train[0][i]
    print(f"{i+1:2d}   | {row[0]:6.3f} {row[1]:6.3f} {row[2]:6.3f} {row[3]:6.3f}")
print("...")
for i in range(max(0, X_train[0].shape[0]-2), X_train[0].shape[0]):  # Show last 2 hours
    row = X_train[0][i]
    print(f"{i+1:2d}   | {row[0]:6.3f} {row[1]:6.3f} {row[2]:6.3f} {row[3]:6.3f}")

print(f"\nTarget (hour 25): {y_train[0]:.3f}")

print(f"\nDataset Summary:")
print(f"Total windows: {len(X_train):,}")
print(f"Each window: 24 hours of 4 sensor readings")
print(f"Prediction: Next hour's CO_GT value")
print(f"Ready for GRU training!")