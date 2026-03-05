"""
Create fully TensorFlow Lite compatible model using Dense layers only
Flattens time-series input for ESP32 deployment

FIXES APPLIED:
- Random seeds for reproducibility (Flaw 9)
- EarlyStopping + ModelCheckpoint (Flaw 10)
- Training history saved to JSON (Flaw 19)
- Training curves plotted (Flaw 18)
- Proper evaluation on test set (Flaw 2)
"""
import tensorflow as tf
import numpy as np
import os
import json
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def create_dense_model(input_shape=(24, 4)):
    """
    Create Dense-only model that's fully TensorFlow Lite compatible
    
    Args:
        input_shape: Input tensor shape (timesteps, features)
        
    Returns:
        Compiled Keras model
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        # Flatten time-series input (trades temporal structure for TFLite compatibility)
        Flatten(input_shape=input_shape, name='flatten'),
        
        # Dense layers for pattern recognition
        Dense(32, activation='relu', name='dense1'),
        Dropout(0.2, name='dropout1'),
        
        Dense(16, activation='relu', name='dense2'),
        Dropout(0.2, name='dropout2'),
        
        Dense(8, activation='relu', name='dense3'),
        
        # Output layer
        Dense(1, activation='linear', name='output')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def save_training_history(history, save_path):
    """Save training history to JSON file"""
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(v) for v in values]
    history_dict['epochs'] = len(history.history['loss'])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"  Training history saved to {save_path}")

def plot_training_curves(history, save_dir):
    """Plot and save training curves for the Dense model"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Dense Model - Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dense_loss_curve.png'), dpi=150)
    plt.close()
    
    # MAE curve
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Dense Model - Training & Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dense_mae_curve.png'), dpi=150)
    plt.close()
    
    print(f"  Training curves saved to {save_dir}/")

def train_dense_model():
    """Train Dense model with proper callbacks and evaluation"""
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    
    print("🚀 Creating Dense Model for TensorFlow Lite")
    print("=" * 60)
    
    # Load training data
    print("Loading training data...")
    X_train = np.load("data/train_X.npy")
    y_train = np.load("data/train_y.npy")
    X_val = np.load("data/val_X.npy")
    y_val = np.load("data/val_y.npy")
    X_test = np.load("data/test_X.npy")
    y_test = np.load("data/test_y.npy")
    
    print(f"  Train: X{X_train.shape}, y{y_train.shape}")
    print(f"  Val:   X{X_val.shape}, y{y_val.shape}")
    print(f"  Test:  X{X_test.shape}, y{y_test.shape}")
    
    # Create Dense model
    print("\nCreating Dense model...")
    model = create_dense_model()
    
    print("\nModel Architecture:")
    model.summary()
    
    # Create callbacks (FIX: Flaw 10 - add early stopping and checkpointing)
    os.makedirs("model", exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath='model/best_dense_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Training (FIX: increased max epochs, added callbacks)
    print(f"\nTraining Dense model (max 50 epochs, patience=10)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history (FIX: Flaw 19)
    save_training_history(history, "reports/dense_training_history.json")
    
    # Plot training curves
    plot_training_curves(history, "reports/plots")
    
    # FIX: Flaw 18 - report best epoch metrics, not cherry-picked minimums
    best_epoch = np.argmin(history.history['val_loss'])
    best_val_loss = history.history['val_loss'][best_epoch]
    best_val_mae = history.history['val_mae'][best_epoch]
    
    print(f"\nTraining Results:")
    print(f"  Best Epoch: {best_epoch + 1}")
    print(f"  Best Val Loss (MSE): {best_val_loss:.6f}")
    print(f"  Best Val MAE: {best_val_mae:.6f}")
    
    # FIX: Flaw 2 - Evaluate the Dense model on the test set and save metrics
    print(f"\nEvaluating Dense model on TEST set...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss (MSE): {test_loss:.6f}")
    print(f"  Test MAE: {test_mae:.6f}")
    
    # Save evaluation metrics
    metrics = {
        'model': 'Dense-only (Flatten→32→16→8→1)',
        'parameters': int(model.count_params()),
        'epochs_trained': len(history.history['loss']),
        'best_epoch': int(best_epoch + 1),
        'best_val_loss': float(best_val_loss),
        'best_val_mae': float(best_val_mae),
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'seed': SEED
    }
    with open("reports/dense_model_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to reports/dense_model_metrics.json")
    
    return model

def convert_dense_to_tflite(model):
    """Convert Dense model to TensorFlow Lite"""
    
    print("\n🔄 Converting Dense Model to TensorFlow Lite...")
    
    # Load representative data
    X_train = np.load("data/train_X.npy")
    representative_data = X_train[:100]
    
    def representative_dataset():
        for i in range(100):
            yield [representative_data[i:i+1].astype(np.float32)]
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    
    # Convert
    print("  Converting model...")
    try:
        tflite_model = converter.convert()
        print("  ✓ Conversion successful!")
        
    except Exception as e:
        print(f"  ⚠️  Quantized conversion failed: {e}")
        print("  Trying without representative dataset...")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        print("  ✓ Basic conversion successful!")
    
    # Save model
    os.makedirs("model", exist_ok=True)
    tflite_path = "model/dense_model.tflite"
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    model_size = len(tflite_model)
    model_size_kb = model_size / 1024
    
    print(f"  ✓ Model saved: {tflite_path}")
    print(f"  ✓ Size: {model_size:,} bytes ({model_size_kb:.1f} KB)")
    
    return tflite_path, model_size_kb

def validate_tflite_model(tflite_path):
    """Validate the TFLite model"""
    
    print(f"\n🔍 Validating TensorFlow Lite Model...")
    
    try:
        # Load interpreter
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Input shape: {input_details[0]['shape']}")
        print(f"  ✓ Input type: {input_details[0]['dtype']}")
        print(f"  ✓ Output shape: {output_details[0]['shape']}")
        print(f"  ✓ Output type: {output_details[0]['dtype']}")
        
        # Test inference
        X_test = np.load("data/test_X.npy")
        test_sample = X_test[0:1].astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], test_sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"  ✓ Test inference successful")
        print(f"  ✓ Sample prediction: {output[0][0]:.6f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Validation failed: {e}")
        return False

def main():
    """Main pipeline for Dense TFLite model"""
    
    # Train Dense model
    model = train_dense_model()
    
    # Convert to TFLite
    tflite_path, model_size_kb = convert_dense_to_tflite(model)
    
    # Validate
    validation_success = validate_tflite_model(tflite_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 DENSE TFLITE MODEL SUMMARY")
    print("=" * 60)
    
    print(f"\nModel Details:")
    print(f"  • Architecture: Dense layers only (TFLite compatible)")
    print(f"  • Parameters: {model.count_params():,}")
    print(f"  • File: {tflite_path}")
    print(f"  • Size: {model_size_kb:.1f} KB")
    
    if model_size_kb <= 20:
        print(f"  ✓ Size within ESP32 limits ({model_size_kb:.1f} KB ≤ 20 KB)")
    else:
        print(f"  ⚠️  Size may be tight for ESP32 ({model_size_kb:.1f} KB)")
    
    if validation_success:
        print(f"  ✓ Model validation successful")
        print(f"  ✓ Ready for ESP32 deployment")
    
    return validation_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)